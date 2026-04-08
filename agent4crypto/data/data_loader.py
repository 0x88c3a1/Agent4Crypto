import base64
import io
import json
import os
from datetime import timedelta
from pathlib import Path

import matplotlib
import mplfinance as mpf
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")


class DataLoader:
    def __init__(self, config_path="config.yaml"):
        if isinstance(config_path, dict):
            self.config = config_path
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

        settings = self.config.get("settings", {})
        self.lookback_days = int(settings.get("data_lookback_days", 90))
        self.visual_lookback = 30
        self.max_news_items = int(settings.get("max_news_items", 3))

        self.data_dir = "data"
        self.market_dir = os.path.join(self.data_dir, "market")
        self.news_dir = os.path.join(self.data_dir, "news")
        self.onchain_dir = os.path.join(self.data_dir, "on-chain")

        os.makedirs(self.market_dir, exist_ok=True)
        os.makedirs(self.news_dir, exist_ok=True)
        os.makedirs(self.onchain_dir, exist_ok=True)

        self.full_history_cache = {}
        self.onchain_history_cache = {}
        self._preload_onchain_data()

    @staticmethod
    def _infer_onchain_symbol(path):
        stem = Path(path).stem
        if stem.endswith("_history"):
            stem = stem[: -len("_history")]
        parts = [part for part in stem.replace("-", "_").split("_") if part]
        for part in reversed(parts):
            if part.isalpha() and part.upper() == part:
                return part
        return parts[-1].upper() if parts else stem.upper()

    def _preload_onchain_data(self):
        for path in sorted(Path(self.onchain_dir).glob("*.csv")):
            symbol = self._infer_onchain_symbol(path)
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"  [Local Data] Failed to read an on-chain file for {symbol}: {exc}")
                continue

            if "date" not in df.columns:
                print(f"  [Local Data] Skipping one on-chain file for {symbol} because it has no 'date' column.")
                continue

            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
            df = df.dropna(subset=["date"])
            if df.empty:
                continue

            self.onchain_history_cache[symbol] = df
            print(f"  [Local Data] Loaded on-chain history for {symbol}")

    @staticmethod
    def _load_market_dataframe(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def _find_local_market_cache(self, ticker, buffer_start_dt, end_dt):
        prefix = f"{ticker}_"
        candidates = []
        if not os.path.isdir(self.market_dir):
            return None

        for filename in os.listdir(self.market_dir):
            if not filename.startswith(prefix) or not filename.endswith(".csv"):
                continue
            path = os.path.join(self.market_dir, filename)
            try:
                df = self._load_market_dataframe(path)
            except Exception:
                continue
            if df.empty:
                continue
            min_idx = pd.to_datetime(df.index.min())
            max_idx = pd.to_datetime(df.index.max())
            if min_idx <= buffer_start_dt and max_idx >= end_dt:
                span = (max_idx - min_idx).days
                candidates.append((span, path, df))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        _, path, df = candidates[0]
        return path, df

    def get_historical_onchain_data(self, ticker, target_date):
        symbol = ticker.split("-")[0].upper()
        df = self.onchain_history_cache.get(symbol)
        if df is None or df.empty:
            return "No local on-chain history available."

        target_day = target_date.normalize()

        try:
            row = df[df["date"].dt.date == target_day.date()]
        except AttributeError:
            row = df[df["date"] == target_day.strftime("%Y-%m-%d")]

        if row.empty:
            return f"On-chain data unavailable for {target_day.strftime('%Y-%m-%d')}."

        data_dict = row.iloc[0].to_dict()
        desc = f"On-Chain Data for {symbol} ({target_day.strftime('%Y-%m-%d')}):\n"
        for key, value in data_dict.items():
            if key == "date":
                continue
            desc += f"- {key}: {value}\n"
        return desc

    def fetch_market_data(self, ticker, start_date, end_date):
        start_dt = pd.to_datetime(start_date)
        buffer_start_dt = start_dt - timedelta(days=self.lookback_days)
        end_dt = pd.to_datetime(end_date)
        buffer_start_str = buffer_start_dt.strftime("%Y-%m-%d")

        print(f"Loading market data for {ticker} (buffer start: {buffer_start_str})...")

        cache_filename = f"{ticker}_{buffer_start_str}_{end_date}.csv"
        cache_path = os.path.join(self.market_dir, cache_filename)

        df = pd.DataFrame()

        if os.path.exists(cache_path):
            print(f"  > [Local Data] Loading market data from {cache_path}")
            df = self._load_market_dataframe(cache_path)
        else:
            local_match = self._find_local_market_cache(ticker, buffer_start_dt, end_dt)
            if local_match is not None:
                matched_path, df = local_match
                print(f"  > [Local Data] Reusing compatible local file {matched_path}")
            else:
                print("  > [Local Data] No compatible market file found.")

        if df.empty:
            print(f"Warning: No data found for {ticker}")
            return df

        df = df[~df.index.duplicated(keep="first")]

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                found = False
                for existing_col in df.columns:
                    if existing_col.lower() == col.lower():
                        df.rename(columns={existing_col: col}, inplace=True)
                        found = True
                        break
                if not found:
                    print(f"Error: Missing required column '{col}' for plotting.")
                    return pd.DataFrame()

        df["MA_20"] = df["Close"].rolling(window=20).mean()
        df["STD_20"] = df["Close"].rolling(window=20).std()
        df["Upper_Band"] = df["MA_20"] + (df["STD_20"] * 2)
        df["Lower_Band"] = df["MA_20"] - (df["STD_20"] * 2)

        exp12 = df["Close"].ewm(span=12, adjust=False).mean()
        exp26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp12 - exp26

        self.full_history_cache[ticker] = df.copy()

        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        sliced_df = df.loc[mask]

        print(f"  > Loop Range: {start_date} -> {end_date} ({len(sliced_df)} rows)")
        return sliced_df

    def _generate_chart_image(self, ticker, df_slice, current_date):
        if df_slice.empty or len(df_slice) < 5:
            return None

        market_colors = mpf.make_marketcolors(up="g", down="r", inherit=True)
        style = mpf.make_mpf_style(marketcolors=market_colors, gridstyle=":", y_on_right=True)
        buf = io.BytesIO()

        try:
            mpf.plot(
                df_slice,
                type="candle",
                style=style,
                volume=True,
                mav=(5, 10, 20),
                title=f"{ticker} Chart ({current_date.strftime('%Y-%m-%d')})",
                savefig=dict(fname=buf, dpi=100, bbox_inches="tight", format="png"),
                block=False,
                warn_too_much_data=1000,
            )
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            return f"data:image/png;base64,{img_str}"
        except Exception as exc:
            print(f"[Chart Error] Failed to generate chart: {exc}")
            return None
        finally:
            buf.close()

    def fetch_news(self, ticker, date_obj):
        symbol_raw = ticker.split("-")[0]
        date_str = date_obj.strftime("%Y-%m-%d")
        cache_filename = f"{symbol_raw}/{date_str}.json"
        cache_path = os.path.join(self.news_dir, cache_filename)

        if not os.path.exists(cache_path):
            print(f"{cache_filename} local news file not found")
            return "No local news found."

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                news_data_list = json.load(f)
        except Exception as exc:
            print(f"{cache_filename} failed to load: {exc}")
            return "No local news found."

        if not news_data_list:
            return "No local news found."

        output = []
        for item in news_data_list[: self.max_news_items]:
            output.append(f"- Title: {item['title']} \n Content: {item.get('content', '')[:300]}...")
        return "\n".join(output)

    def get_day_data(self, ticker, market_df_ignored, current_date):
        date_str = current_date.strftime("%Y-%m-%d")

        full_df = self.full_history_cache.get(ticker)
        if full_df is None or full_df.empty:
            full_df = market_df_ignored

        try:
            idx_loc = full_df.index.get_loc(current_date)
            if isinstance(idx_loc, slice):
                idx_loc = idx_loc.start
            elif isinstance(idx_loc, np.ndarray):
                idx_loc = np.where(idx_loc)[0][0]
        except KeyError:
            return None

        row = full_df.iloc[idx_loc]

        start_idx = max(0, idx_loc - self.visual_lookback)
        chart_slice = full_df.iloc[start_idx : idx_loc + 1].copy()
        chart_base64 = self._generate_chart_image(ticker, chart_slice, current_date)
        onchain_data = self.get_historical_onchain_data(ticker, current_date)

        def fmt(val):
            return f"{val:.2f}" if pd.notnull(val) else "N/A"

        tech_info = (
            f"Price: {fmt(row['Close'])}, "
            f"MA20: {fmt(row['MA_20'])}, "
            f"Bollinger Up: {fmt(row['Upper_Band'])}, "
            f"Bollinger Low: {fmt(row['Lower_Band'])}, "
            f"MACD: {fmt(row['MACD'])}"
        )

        return {
            "date": date_str,
            "ticker": ticker,
            "price": float(row["Close"]),
            "open_price": float(row["Open"]),
            "technical": tech_info,
            "news": self.fetch_news(ticker, current_date),
            "chart_image": chart_base64,
            "onchain": onchain_data,
        }
