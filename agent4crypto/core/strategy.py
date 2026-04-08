import collections
import re

import numpy as np
import pandas as pd
from colorama import Fore

try:
    from .rl_module import PPOAgent

    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print(f"{Fore.RED}[Warning] Torch not installed. PPO/RL mode will not work.")


class MemoryModule:
    """Store recent trades and reflections for the reflector and trader prompts."""

    def __init__(self, capacity=10):
        self.capacity = capacity
        self.trade_history = []
        self.reflections = []

    def add_trade(self, date, action, price, pnl):
        self.trade_history.append({"date": date, "action": action, "price": price, "pnl": pnl})
        if len(self.trade_history) > self.capacity:
            self.trade_history.pop(0)

    def add_reflection(self, reflection):
        self.reflections.append(reflection)
        if len(self.reflections) > self.capacity:
            self.reflections.pop(0)

    def get_context(self):
        if not self.reflections:
            return "No past reflections available."
        return "\n".join(self.reflections[-3:])


class BaseStrategy:
    def __init__(self, initial_capital, fee_rate=0.0, logger=None):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.fee_rate = fee_rate
        self.positions = 0
        self.portfolio_value = []
        self.dates = []
        self.logger = logger

    def reset_portfolio_state(self):
        """Reset execution state while keeping the configured capital and fee model."""
        self.capital = self.initial_capital
        self.positions = 0
        self.portfolio_value = []
        self.dates = []

    def execute(self, action, current_price, date):
        action_str = "HOLD"

        if action == 1 and self.capital > 0:
            effective_capital = self.capital * (1 - self.fee_rate)
            fee_cost = self.capital * self.fee_rate

            self.positions = effective_capital / current_price
            self.capital = 0
            action_str = "BUY"

            if self.logger:
                self.logger.info(f"  [COST] Fee paid: ${fee_cost:.2f}")

        elif action == -1 and self.positions > 0:
            gross_revenue = self.positions * current_price
            self.capital = gross_revenue * (1 - self.fee_rate)
            fee_cost = gross_revenue * self.fee_rate

            self.positions = 0
            action_str = "SELL"

            if self.logger:
                self.logger.info(f"  [COST] Fee paid: ${fee_cost:.2f}")

        current_wealth = self.capital + (self.positions * current_price)
        self.portfolio_value.append(current_wealth)
        self.dates.append(date)
        ret = (current_wealth - self.initial_capital) / self.initial_capital * 100

        if self.logger:
            self.logger.trade(action_str, current_price, current_wealth, ret)
        return current_wealth, action_str


class Agent4CryptoStrategy(BaseStrategy):
    def __init__(self, config, agent_factory, logger):
        fee = config["settings"].get("transaction_fee", 0.0)
        super().__init__(
            initial_capital=config["settings"]["initial_capital"],
            fee_rate=fee,
            logger=logger,
        )
        self.config = config
        self.factory = agent_factory
        self.memory = MemoryModule()
        self.mode = config["agent_params"].get("strategy_mode", "base").lower()

        init_w = np.array(config["agent_params"].get("initial_weights", [0.33, 0.33, 0.33]))
        self.initial_weights = init_w / init_w.sum()
        self.weights = self.initial_weights.copy()
        self.alpha = config["agent_params"].get("alpha", 0.0001)

        self.prev_decisions = None
        self.prev_price = None
        self.prev_open_price = None
        self.last_action_str = "HOLD"
        self.last_analysis_context = "Initial Day - No Context"

        self.ppo_agent = None
        self.price_history = collections.deque(maxlen=30)
        self.rl_state_buffer = None
        self.rl_training_enabled = False
        self.rl_rollout_horizon = config["agent_params"].get("rl_rollout_horizon", 50)

        if self.mode == "rl" and RL_AVAILABLE:
            if logger:
                logger.info("Initializing PPO agent for RL mode...", color=Fore.MAGENTA)
            self.state_dim = 14
            self.ppo_agent = PPOAgent(state_dim=self.state_dim, lr=0.0005, K_epochs=10)
            self.rl_training_enabled = True

    def _clean_text(self, text):
        if not text:
            return ""
        clean = re.sub(r"[\*#_`]", "", text)
        clean = re.sub(r"\n\s*\n", "\n", clean).strip()
        return clean

    def _log_input(self, agent_name, content):
        if not self.logger:
            return

        display_content = content
        if len(str(content)) > 150 and "News" in agent_name:
            display_content = str(content)[:150] + "... [Full content sent to Agent]"
        elif "Visual" in agent_name:
            display_content = "[Base64 Image Data] + Prompt"

        self.logger.info(f"  -> [INPUT to {agent_name}]: {display_content}", color=Fore.LIGHTBLACK_EX)

    def _calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        seed = deltas[: period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100.0

    def _calculate_macd_trend(self, prices):
        """Estimate MACD trend strength on a [-1, 1] scale."""
        if len(prices) < 26:
            return 0.0
        series = pd.Series(prices)
        exp12 = series.ewm(span=12, adjust=False).mean()
        exp26 = series.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        diff = macd.iloc[-1] - signal.iloc[-1]
        return np.tanh(diff)

    def _get_rl_state(self, candidate_actions=None):
        """Build the PPO state vector from weights, recent market features, and trader proposals."""
        state = list(self.weights)
        prices = list(self.price_history)

        if len(prices) < 6:
            returns = [0.0] * 5
            vol = 0.0
        else:
            price_series = pd.Series(prices)
            log_ret = np.log(price_series / price_series.shift(1)).dropna()
            returns = log_ret.values[-5:].tolist()
            while len(returns) < 5:
                returns.insert(0, 0.0)
            vol = log_ret[-5:].std() if len(log_ret) >= 5 else 0.0

        state.extend(returns)
        state.append(vol if not np.isnan(vol) else 0.0)
        state.append(self._calculate_rsi(prices))
        state.append(self._calculate_macd_trend(prices))
        if candidate_actions is None:
            candidate_actions = [0.0, 0.0, 0.0]
        state.extend([float(x) for x in candidate_actions[:3]])
        return np.array(state)

    def _reset_runtime_state(self, reset_price_history=True):
        """Clear execution-time state so validation training does not leak into test bookkeeping."""
        self.reset_portfolio_state()
        self.memory = MemoryModule()
        self.prev_decisions = None
        self.prev_price = None
        self.prev_open_price = None
        self.last_action_str = "HOLD"
        self.last_analysis_context = "Initial Day - No Context"
        self.rl_state_buffer = None
        self.weights = self.initial_weights.copy()
        if reset_price_history:
            self.price_history = collections.deque(maxlen=30)

    def freeze_rl_for_test(self, warm_prices=None):
        """Finalize PPO updates on validation data and switch to frozen inference for testing."""
        if self.mode != "rl" or not self.ppo_agent:
            return

        if self.ppo_agent.buffer:
            if self.logger:
                self.logger.info("[RL] Final PPO update on remaining validation transitions.", color=Fore.MAGENTA)
            self.ppo_agent.update()

        self.rl_training_enabled = False
        self._reset_runtime_state(reset_price_history=True)

        if warm_prices:
            self.price_history.extend(list(warm_prices)[-self.price_history.maxlen :])

        if self.logger:
            self.logger.info(
                "[RL] Validation training complete. PPO policy frozen for test-time inference.",
                color=Fore.MAGENTA,
            )

    def pretrain_on_validation(self, ticker, validation_df, loader):
        """Run the RL variant over the validation window, then freeze the learned policy for testing."""
        if self.mode != "rl" or not self.ppo_agent or validation_df.empty:
            return

        if self.logger:
            self.logger.section(f"RL VALIDATION TRAINING: {ticker}")
            self.logger.info(
                f"Training PPO on validation window {validation_df.index[0].date()} -> {validation_df.index[-1].date()}",
                color=Fore.MAGENTA,
            )

        self.rl_training_enabled = True
        for date in validation_df.index:
            day_data = loader.get_day_data(ticker, validation_df, pd.to_datetime(date))
            if day_data is None:
                continue
            self.decide_and_trade(day_data)

        warm_prices = validation_df["Close"].tail(self.price_history.maxlen).tolist()
        self.freeze_rl_for_test(warm_prices=warm_prices)

    def _update_weights_rl(self, current_price, current_decisions):
        self.price_history.append(current_price)
        curr_state = self._get_rl_state(current_decisions)

        if not self.rl_training_enabled:
            self.weights = self.ppo_agent.predict_action(curr_state)
            if self.logger:
                self.logger.info(
                    f"[RL Frozen Weights] Agg:{self.weights[0]:.2f} Con:{self.weights[1]:.2f} Neu:{self.weights[2]:.2f}",
                    color=Fore.MAGENTA,
                )
            return

        if self.rl_state_buffer is None:
            action, log_prob = self.ppo_agent.select_action(curr_state)
            self.weights = action
            self.rl_state_buffer = (curr_state, action, log_prob)
            return

        prev_s, prev_a, prev_log_prob = self.rl_state_buffer

        market_ret = 0.0
        if len(self.price_history) >= 2:
            market_ret = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]

        if self.prev_decisions:
            weighted_score = np.dot(prev_a, self.prev_decisions)
            final_sign = 1 if weighted_score > 0.1 else (-1 if weighted_score < -0.1 else 0)
            reward = final_sign * market_ret * 100
            if final_sign == 0 and market_ret > 0.02:
                reward -= 0.5
        else:
            reward = 0

        self.ppo_agent.store_transition((prev_s, prev_a, prev_log_prob, reward))

        if len(self.ppo_agent.buffer) >= self.rl_rollout_horizon:
            if self.logger:
                self.logger.info(f"[RL] Updating PPO... (Last reward: {reward:.2f})", color=Fore.MAGENTA)
            self.ppo_agent.update()

        new_weights, new_log_prob = self.ppo_agent.select_action(curr_state)
        self.weights = new_weights
        self.rl_state_buffer = (curr_state, new_weights, new_log_prob)

        if self.logger:
            self.logger.info(
                f"[RL Weights] Agg:{self.weights[0]:.2f} Con:{self.weights[1]:.2f} Neu:{self.weights[2]:.2f} | R:{reward:.2f}",
                color=Fore.MAGENTA,
            )

    def _update_weights_dr(self, current_open_price):
        if self.prev_decisions is None or self.prev_open_price is None:
            return

        market_ret = (current_open_price - self.prev_open_price) / self.prev_open_price
        market_dir = 1 if market_ret > 0.001 else (-1 if market_ret < -0.001 else 0)
        if market_dir == 0:
            return

        for idx, decision in enumerate(self.prev_decisions):
            if decision == market_dir:
                self.weights[idx] += self.alpha
            elif decision != 0:
                self.weights[idx] -= self.alpha

        self.weights = np.maximum(self.weights, 0.01)
        self.weights /= self.weights.sum()

        if self.logger:
            self.logger.info(
                f"[DR Weights] Agg: {self.weights[0]:.2f}, Cons: {self.weights[1]:.2f}, Neu: {self.weights[2]:.2f}",
                color=Fore.CYAN,
            )

    def decide_and_trade(self, day_data):
        current_price = day_data["price"]
        current_open_price = day_data.get("open_price", current_price)
        ablation = self.config.get("ablation", {})

        if self.logger:
            self.logger.section(f"DATE: {day_data['date']} | Price: ${current_price:.2f}")

        current_pos_str = (
            f"HOLDING ASSET ({day_data['ticker']})" if self.positions > 0 else "HOLDING CASH (USD)"
        )

        if self.logger:
            self.logger.info(f"[CURRENT POSITION]: {current_pos_str}", color=Fore.CYAN)

        if ablation.get("inter_team", {}).get("use_reflector", True) and self.portfolio_value:
            current_wealth_estimate = self.capital + (self.positions * current_price)
            last_wealth = self.portfolio_value[-1]
            realized_pnl_pct = 0.0
            if last_wealth > 0:
                realized_pnl_pct = (current_wealth_estimate - last_wealth) / last_wealth * 100

            market_return_pct = 0.0
            if self.prev_price and self.prev_price > 0:
                market_return_pct = (current_price - self.prev_price) / self.prev_price * 100

            should_reflect = (
                self.last_action_str != "HOLD"
                or abs(realized_pnl_pct) > 0.01
                or abs(market_return_pct) > 1.0
            )

            if should_reflect:
                if self.logger:
                    self.logger.sub_section("Phase 0: Reflection")
                reflection = self.factory.get_reflection(
                    self.last_action_str,
                    realized_pnl_pct,
                    market_return_pct,
                    self.last_analysis_context,
                )
                self.memory.add_reflection(reflection)
                if self.logger:
                    self.logger.info(f"[REFLECTOR]: {self._clean_text(reflection)}", color=Fore.LIGHTBLACK_EX)

        if self.mode == "dr":
            self._update_weights_dr(current_open_price)

        if self.logger:
            self.logger.sub_section("Phase 1: Data Team Analysis")
        self._log_input("ASSET", day_data["onchain"])
        self._log_input("TECHNICAL", day_data["technical"])
        self._log_input("NEWS", day_data["news"])
        self._log_input("VISUAL", "Chart Image Valid: " + str(bool(day_data["chart_image"])))

        data_reports = self.factory.get_data_team_reports(day_data)
        if self.logger:
            for agent, report in data_reports.items():
                self.logger.info(f"[{agent.upper()} AGENT]: {self._clean_text(report)}", color=Fore.BLUE)

        if self.logger:
            self.logger.sub_section("Phase 2: Analyst Synthesis")
        analyst_report = self.factory.get_unified_analysis(data_reports)
        if self.logger:
            self.logger.info(f"[ANALYST]: {self._clean_text(analyst_report)}", color=Fore.YELLOW)

        risk_report = "Risk Manager Disabled"
        if ablation.get("inter_team", {}).get("use_risk_manager", True):
            if self.logger:
                self.logger.sub_section("Phase 3: Risk Assessment")
            risk_report = self.factory.get_risk_assessment(analyst_report, day_data)
            if self.logger:
                self.logger.info(f"[RISK MANAGER]: {self._clean_text(risk_report)}", color=Fore.MAGENTA)

        mem_context = self.memory.get_context()
        styles = ["Aggressive", "Conservative", "Neutral"] if ablation.get("mechanism", {}).get(
            "use_competition", True
        ) else ["Neutral"]

        if self.logger:
            self.logger.sub_section("Phase 4: Trader Team Proposals")

        current_decisions = []
        for style in styles:
            decision, reasoning = self.factory.get_trader_decision(
                style, analyst_report, risk_report, mem_context, current_pos_str
            )
            current_decisions.append(decision)
            if self.logger:
                clean_reasoning = self._clean_text(reasoning)
                decision_color = Fore.GREEN if decision == 1 else (Fore.RED if decision == -1 else Fore.WHITE)
                self.logger.info(
                    f"[{style.upper()} TRADER]: {clean_reasoning} => Decision: {decision}",
                    color=decision_color,
                )

        if self.mode == "rl" and self.ppo_agent:
            self._update_weights_rl(current_price, current_decisions)

        effective_weights = self.weights if len(current_decisions) == 3 else np.array([1.0])
        weighted_score = np.dot(effective_weights, current_decisions)

        if self.logger:
            self.logger.info(f"Weighted Score: {weighted_score:.4f}", color=Fore.CYAN)

        final_action = 1 if weighted_score > 0.1 else (-1 if weighted_score < -0.1 else 0)
        _, action_str = self.execute(final_action, current_price, day_data["date"])

        self.last_action_str = action_str
        self.last_analysis_context = analyst_report
        self.prev_decisions = current_decisions if len(current_decisions) == 3 else [current_decisions[0], 0, 0]
        self.prev_price = current_price
        self.prev_open_price = current_open_price
