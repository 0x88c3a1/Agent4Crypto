import concurrent.futures
import re

import openai
from colorama import Fore, Style
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .experiment_utils import get_run_seed
from .prompts import ANALYST_PROMPT, DATA_TEAM_PROMPTS, SYSTEM_PROMPTS, TRADER_TEAM_PROMPTS


class LLMAgent:
    """Thin wrapper around chat completion calls, including multimodal input."""

    def __init__(self, model, api_key, base_url=None, seed=42):
        self.model = (model or "").strip()
        if not self.model:
            raise ValueError(
                "No model name configured. Set agent_params.model in config.yaml before running Agent4Crypto."
            )
        self.seed = seed
        normalized_base_url = (base_url or "").strip() or None
        self.client = openai.OpenAI(api_key=api_key, base_url=normalized_base_url)

    @retry(
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)
        ),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    def query(self, system_prompt, user_content, image_base64=None):
        """Send a request to the LLM, optionally with a chart image."""
        messages = [{"role": "system", "content": system_prompt}]

        if image_base64:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                    {"type": "image_url", "image_url": {"url": image_base64}},
                ],
            }
        else:
            user_message = {"role": "user", "content": user_content}

        messages.append(user_message)

        try:
            request_kwargs = dict(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            if self.seed is not None:
                request_kwargs["seed"] = self.seed

            response = self.client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content
        except Exception as exc:
            if "429" in str(exc) or "Rate limit" in str(exc):
                print(f"{Fore.YELLOW}[Rate Limit Hit] Retrying...{Style.RESET_ALL}")
                raise

            print(f"{Fore.RED}[LLM Error ({self.model})] {exc}{Style.RESET_ALL}")
            return "NEUTRAL/HOLD (LLM Error)"


class AgentFactory:
    """Build and orchestrate the agents used in the Agent4Crypto pipeline."""

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger

        api_key = config["api_keys"]["openai"]
        base_url = config["api_keys"].get("base_url")
        run_seed = get_run_seed(config)

        main_model_name = config["agent_params"].get("model")
        self.default_llm = LLMAgent(model=main_model_name, api_key=api_key, base_url=base_url, seed=run_seed)

        visual_model_name = config["agent_params"].get("visual_model") or main_model_name
        self.visual_llm = LLMAgent(model=visual_model_name, api_key=api_key, base_url=base_url, seed=run_seed)

        if self.logger:
            self.logger.info(
                f"Agents initialized | Main: {main_model_name} | Visual: {visual_model_name}",
                color=Fore.CYAN,
            )

    def get_data_team_reports(self, day_data):
        reports = {}
        dt_config = self.config.get("ablation", {}).get("data_team", {})

        def run_agent(name, prompt_key, content, is_visual=False):
            if is_visual:
                if self.logger:
                    self.logger.info("Sending chart to Visual Agent...", color=Fore.BLUE)
                return (
                    name,
                    self.visual_llm.query(
                        DATA_TEAM_PROMPTS[prompt_key],
                        "Analyze the provided candlestick chart image.",
                        image_base64=content,
                    ),
                )
            return name, self.default_llm.query(DATA_TEAM_PROMPTS[prompt_key], content)

        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            if dt_config.get("use_asset", True):
                content = f"Ticker: {day_data['ticker']}\nOn-chain Data: {day_data['onchain']}"
                tasks.append(executor.submit(run_agent, "asset", "Asset", content))

            if dt_config.get("use_technical", True):
                content = f"Technical Indicators: {day_data['technical']}"
                tasks.append(executor.submit(run_agent, "technical", "Technical", content))

            if dt_config.get("use_news", True):
                content = f"Recent Market News:\n{day_data['news']}"
                tasks.append(executor.submit(run_agent, "news", "News", content))

            if dt_config.get("use_visual", True):
                chart_img = day_data.get("chart_image")
                if chart_img:
                    tasks.append(executor.submit(run_agent, "visual", "Visual", chart_img, True))
                else:
                    reports["visual"] = "No chart image available."

            for future in concurrent.futures.as_completed(tasks):
                try:
                    name, result = future.result()
                    reports[name] = result
                except Exception as exc:
                    print(f"Agent execution failed: {exc}")

        return reports

    def get_unified_analysis(self, data_reports):
        ordered_agents = ["asset", "technical", "news", "visual"]
        display_names = {
            "asset": "Asset",
            "technical": "Technical",
            "news": "News",
            "visual": "Visual",
        }

        available_agents = [name for name in ordered_agents if name in data_reports]
        missing_agents = [name for name in ordered_agents if name not in data_reports]

        context_parts = [
            "Only use the evidence explicitly listed below.",
            "If a source is missing, disabled, or unavailable, treat it as unknown and do not infer its contents.",
            "Available reports: "
            + (", ".join(display_names[name] for name in available_agents) if available_agents else "None"),
        ]
        if missing_agents:
            context_parts.append(
                "Missing/disabled/unavailable sources: " + ", ".join(display_names[name] for name in missing_agents)
            )

        context_parts.append("Here are the reports from the Data Team:")
        for agent_name in available_agents:
            report = data_reports[agent_name]
            context_parts.append(f"--- {display_names[agent_name].upper()} AGENT REPORT ---\n{report}")

        if not available_agents:
            context_parts.append("No Data Team reports were provided.")

        context = "\n\n".join(context_parts)
        return self.default_llm.query(ANALYST_PROMPT, context)

    def get_risk_assessment(self, analyst_report, day_data):
        content = (
            f"Analyst's Market View: {analyst_report}\n"
            f"Current Market Data: {day_data['technical']}"
        )
        return self.default_llm.query(SYSTEM_PROMPTS["RiskManager"], content)

    def get_trader_decision(self, style, analyst_report, risk_report, memory_context, current_position):
        prompt = TRADER_TEAM_PROMPTS.get(style, TRADER_TEAM_PROMPTS["Neutral"])
        content = (
            f"--- CURRENT STATUS ---\n"
            f"Current Position: {current_position} (This is crucial!)\n\n"
            f"--- REPORTS ---\n"
            f"1. Chief Analyst Report: {analyst_report}\n"
            f"2. Risk Manager Assessment: {risk_report}\n"
            f"3. Memory: {memory_context}\n\n"
            f"--- INSTRUCTION ---\n"
            f"Based on your trading style ({style}), decide: BUY (1), SELL (-1), or HOLD (0).\n"
            f"Note:\n"
            f"- If I hold CASH and you want to enter, say BUY.\n"
            f"- If I hold ASSET and you want to exit, say SELL.\n"
            f"- If I hold ASSET and you are bearish, you MUST say SELL, do not say HOLD."
        )

        response = self.default_llm.query(prompt, content)
        decision = 0
        match = re.search(r"DECISION:\s*(-?1|0|BUY|SELL|HOLD)", response.upper())

        if match:
            result_str = match.group(1)
            if result_str in ["1", "BUY"]:
                decision = 1
            elif result_str in ["-1", "SELL"]:
                decision = -1
            elif result_str in ["0", "HOLD"]:
                decision = 0
        else:
            cleaned_res = response.upper()
            if "BUY" in cleaned_res and "SELL" not in cleaned_res:
                decision = 1
            elif "SELL" in cleaned_res and "BUY" not in cleaned_res:
                if "NOT SELL" not in cleaned_res and "premature" not in response.lower():
                    decision = -1

        return decision, response

    def get_reflection(self, trade_action, pnl, market_return, context_t1):
        """Summarize what yesterday's reasoning got right or wrong."""
        hold_state = ""
        if trade_action == "HOLD":
            if abs(pnl - market_return) < 0.1:
                hold_state = "(Held Asset)"
            elif abs(pnl) < 0.1 and abs(market_return) > 0.5:
                hold_state = "(Held Cash)"

        prompt = (
            f"--- TRADING REFLECTION ---\n"
            f"1. CONTEXT (Yesterday T-1): What we believed:\n   \"{context_t1}\"\n\n"
            f"2. ACTION (Yesterday T-1): We decided to {trade_action} {hold_state}.\n\n"
            f"3. OUTCOME (Today T): The market moved {market_return:+.2f}%, resulting in PnL {pnl:+.2f}%.\n\n"
            "TASK: Evaluate if the analysis (Context) justified the Action, given the Outcome.\n"
            "- Did the market behave as the Analyst predicted?\n"
            "- If we lost money, was it because of bad analysis or unexpected volatility?\n"
            "- If we missed out (Sold too early), was the analysis overly bearish?\n\n"
            "Output a concise, specific LESSON for the future."
        )
        return self.default_llm.query(SYSTEM_PROMPTS["Reflector"], prompt)
