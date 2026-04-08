import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ..core.agents import AgentFactory
from ..core.experiment_utils import get_run_seed, get_run_seed_list, set_global_seed
from ..core.strategy import Agent4CryptoStrategy
from ..data.data_loader import DataLoader
from ..ui.utils import Logger
from ..ui.visualizer import SystemVisualizer

plt.rcParams["axes.unicode_minus"] = False


def calculate_metrics_raw(portfolio_values, market_prices):
    """Compute numeric backtest metrics and compare them with Buy & Hold."""
    if not portfolio_values or len(portfolio_values) < 2:
        return {}

    df = pd.DataFrame({"value": portfolio_values})
    df["daily_return"] = df["value"].pct_change().fillna(0)

    total_return = (df["value"].iloc[-1] - df["value"].iloc[0]) / df["value"].iloc[0]
    volatility = df["daily_return"].std() * np.sqrt(365)

    mean_ret = df["daily_return"].mean()
    std_ret = df["daily_return"].std()
    sharpe = (mean_ret / std_ret) if std_ret > 0 else 0

    downside_returns = df.loc[df["daily_return"] < 0, "daily_return"]
    downside_std = downside_returns.std()
    sortino = (mean_ret / downside_std) * np.sqrt(365) if downside_std > 0 else 0

    cummax = df["value"].cummax()
    drawdown = (df["value"] - cummax) / cummax
    max_drawdown = drawdown.min()
    calmar = abs(total_return / max_drawdown) if max_drawdown != 0 else 0

    benchmark_return = 0
    if len(market_prices) > 1:
        start_price = market_prices[0]
        end_price = market_prices[-1]
        benchmark_return = (end_price - start_price) / start_price

    return {
        "total_return": float(total_return),
        "benchmark_return": float(benchmark_return),
        "alpha": float(total_return - benchmark_return),
        "mean_ret": float(mean_ret),
        "std_ret": float(std_ret),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "volatility": float(volatility),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar),
    }


def calculate_metrics(portfolio_values, market_prices):
    """Format backtest metrics for terminal-friendly reporting."""
    raw = calculate_metrics_raw(portfolio_values, market_prices)
    if not raw:
        return {}

    return {
        "Total Return": f"{raw['total_return'] * 100:.2f}%",
        "Benchmark (B&H)": f"{raw['benchmark_return'] * 100:.2f}%",
        "Alpha": f"{raw['alpha'] * 100:.2f}%",
        "mean_ret": f"{raw['mean_ret'] * 100:.2f}%",
        "std_ret": f"{raw['std_ret'] * 100:.2f}%",
        "Sharpe Ratio": f"{raw['sharpe_ratio']:.2f}",
        "Sortino Ratio": f"{raw['sortino_ratio']:.4f}",
        "Volatility": f"{raw['volatility'] * 100:.2f}%",
        "Max Drawdown": f"{raw['max_drawdown'] * 100:.2f}%",
        "Calmar Ratio": f"{raw['calmar_ratio']:.4f}",
    }


def plot_equity_curve(strategy, market_df, ticker, regime):
    """Plot and save the strategy equity curve against Buy & Hold."""
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)

    dates = pd.to_datetime(strategy.dates)
    portfolio_vals = np.array(strategy.portfolio_value)
    market_prices = market_df["Close"].iloc[: len(portfolio_vals)].values

    normalized_portfolio = portfolio_vals / portfolio_vals[0]
    normalized_market = market_prices / market_prices[0]

    plt.figure(figsize=(12, 6))
    plt.plot(dates, normalized_portfolio, label="Agent4Crypto Strategy", color="blue", linewidth=2)
    plt.plot(dates, normalized_market, label=f"Buy & Hold ({ticker})", color="gray", linestyle="--", alpha=0.7)
    plt.fill_between(dates, normalized_portfolio, 1, where=(normalized_portfolio < 1), color="red", alpha=0.1)

    plt.title(f"Performance Analysis: {ticker} ({regime.upper()})", fontsize=14)
    plt.ylabel("Normalized Wealth (Start = 1.0)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    filename = f"{result_dir}/equity_{ticker}_{regime}_{int(time.time())}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Equity curve saved to: {filename}")


def run_experiment(
    target_regime,
    config,
    loader,
    specific_asset=None,
    show_visualizer=True,
    save_plot=True,
    logger=None,
):
    """Run a backtest for a specific regime."""
    if show_visualizer:
        viz = SystemVisualizer()
        viz.boot_sequence()

    log = logger if logger is not None else Logger(experiment_name=f"AGENT4CRYPTO_{target_regime}")
    run_seed = get_run_seed(config)
    configured_seed_list = get_run_seed_list(config)
    log.section(f"STARTING EXPERIMENT: {target_regime.upper()}")
    log.info(f"Ablation Settings: {config.get('ablation', {})}")
    if run_seed is None:
        log.info("Run seed: random / unset")
    elif configured_seed_list and config.get("backtest", {}).get("run_seed") in (None, ""):
        log.info(f"Run seed: {run_seed} (first entry from backtest.run_seeds)")
    else:
        log.info(f"Run seed: {run_seed}")
    results = []

    for ticker, regimes in config["assets"].items():
        if specific_asset and ticker != specific_asset:
            continue
        if target_regime not in regimes:
            continue

        set_global_seed(run_seed)

        date_range = regimes[target_regime]
        start_date, end_date = date_range["start"], date_range["end"]
        log.info(f"Testing {ticker} | Range: {start_date} -> {end_date}", bold=True)

        agent_factory = AgentFactory(config, logger=log)
        strategy = Agent4CryptoStrategy(config, agent_factory, logger=log)
        strategy_mode = config.get("agent_params", {}).get("strategy_mode", "base").lower()

        if strategy_mode == "rl" and target_regime != "validation" and "validation" in regimes:
            val_range = regimes["validation"]
            log.info(
                f"RL validation warm-up for {ticker} | Range: {val_range['start']} -> {val_range['end']}",
                bold=True,
            )
            validation_df = loader.fetch_market_data(ticker, val_range["start"], val_range["end"])
            if validation_df.empty:
                log.info("Validation data missing; skipping PPO warm-up and using the initialized policy.")
                strategy.freeze_rl_for_test()
            else:
                strategy.pretrain_on_validation(ticker, validation_df, loader)

        market_df = loader.fetch_market_data(ticker, start_date, end_date)
        if market_df.empty:
            log.info("No data found, skipping.")
            continue

        backtest_window = market_df.loc[start_date:end_date]

        for date in market_df.index:
            current_date = pd.to_datetime(date)
            day_data = loader.get_day_data(ticker, market_df, current_date)
            strategy.decide_and_trade(day_data)

        benchmark_prices = backtest_window["Close"].values
        metrics_raw = calculate_metrics_raw(strategy.portfolio_value, benchmark_prices)
        metrics = calculate_metrics(strategy.portfolio_value, benchmark_prices)

        log.section(f"SUMMARY: {ticker} ({target_regime.upper()})")
        for key, value in metrics.items():
            log.info(f"{key:<20}: {value}", bold=True)

        if save_plot and strategy.portfolio_value:
            plot_equity_curve(strategy, backtest_window, ticker, target_regime)

        results.append(
            {
                "ticker": ticker,
                "regime": target_regime,
                "metrics": metrics,
                "metrics_raw": metrics_raw,
                "run_seed": run_seed,
            }
        )

    return results


def main(config_path="config.yaml"):
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    loader = DataLoader(config_path)
    run_experiment(
        config["backtest"]["target_regime"],
        config,
        loader,
        specific_asset=config["backtest"]["target_asset"],
    )


if __name__ == "__main__":
    main()
