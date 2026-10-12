import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


# ============================================================
# 0. 参数区：你主要改这里
# ============================================================

DATA_ROOT = Path(os.environ.get("MARKET_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data/market_data"))).expanduser()
ETF_FILE = DATA_ROOT / "merged_data/511090_20250701_20251231_merged.csv"

FUTURES_FILE = DATA_ROOT / "国债期货合约数据/快照/30年TL合约/TL_main_snapshot.csv"

OUTPUT_DIR = Path("task2b_511090_ou_pair_trading_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2025-07-01"
END_DATE = "2025-12-31"

INITIAL_CASH = 10_000_000.0

CAPITAL_LIMIT = 10_000_000
MAX_FUTURES_POSITION = 8

# TL 合约乘数。如果你们文档里有明确乘数，以文档为准。
FUTURES_MULTIPLIER = 10_000

# OU 动态更新频率：180分钟
UPDATE_INTERVAL_BARS = 180

# OU 信号阈值
ENTRY_Z = 2.0
EXIT_Z = 0.5

# z-score 到达 4 时使用最大仓位；低于 4 时按比例放大
Z_CAP_FOR_FULL_POSITION = 4.0

# 是否过滤期货交易时间
FILTER_FUTURES_TRADING_TIME = True


# ============================================================
# 1. OU 参数估计函数
# ============================================================

def estimate_ou_params(spread_series, dt=1.0):
    """
    用 AR(1) 离散形式估计 OU 过程参数。

    OU:
        dX_t = theta * (mu - X_t) dt + sigma dW_t

    离散化：
        X_t = a + b * X_{t-1} + eps_t

    b = exp(-theta * dt)
    mu = a / (1 - b)

    z-score 使用长期稳态标准差：
        stationary_sigma = sigma_diffusion / sqrt(2 * theta)
    """
    x = pd.Series(spread_series).dropna().astype(float).values

    if len(x) < 50:
        return None

    x_lag = x[:-1]
    x_now = x[1:]

    X = np.column_stack([np.ones(len(x_lag)), x_lag])

    try:
        beta = np.linalg.lstsq(X, x_now, rcond=None)[0]
    except Exception:
        return None

    a = beta[0]
    b = beta[1]

    residual = x_now - (a + b * x_lag)
    residual_var = np.var(residual, ddof=1)

    # OU 均值回复要求 0 < b < 1
    if b <= 0 or b >= 1:
        return None

    theta = -np.log(b) / dt

    if abs(1 - b) < 1e-12:
        return None

    mu = a / (1 - b)

    sigma_diffusion = np.sqrt(
        residual_var * 2 * theta / (1 - np.exp(-2 * theta * dt))
    )

    stationary_sigma = sigma_diffusion / np.sqrt(2 * theta)

    if (
        not np.isfinite(theta)
        or not np.isfinite(mu)
        or not np.isfinite(stationary_sigma)
        or theta <= 0
        or stationary_sigma <= 0
    ):
        return None

    half_life = np.log(2) / theta

    return {
        "theta": theta,
        "mu": mu,
        "sigma_diffusion": sigma_diffusion,
        "stationary_sigma": stationary_sigma,
        "half_life_minutes": half_life,
        "ar1_beta": b,
        "residual_std": np.sqrt(residual_var),
        "n_obs": len(x),
    }


# ============================================================
# 2. ETF 数据处理
# ============================================================

def load_etf_snapshot_csv(path, from_date=None, to_date=None):
    df = pd.read_csv(path)

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]

    if "trade_time" not in df.columns:
        raise ValueError("ETF 数据缺少 trade_time 列")

    df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
    df = df.dropna(subset=["trade_time"])
    df = df.sort_values("trade_time")

    if from_date is not None:
        df = df[df["trade_time"] >= pd.Timestamp(from_date)]

    if to_date is not None:
        to_dt = pd.Timestamp(to_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[df["trade_time"] <= to_dt]

    df = df.set_index("trade_time").sort_index()

    return df


def prepare_etf_minute_data(etf_df):
    """
    ETF价格优先使用 last，其次使用 close。
    最后重采样到 1分钟。
    """
    df = etf_df.copy()

    for col in ["last", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] <= 0, col] = np.nan

    if "last" in df.columns and "close" in df.columns:
        df["etf_price"] = df["last"].fillna(df["close"])
    elif "last" in df.columns:
        df["etf_price"] = df["last"]
    elif "close" in df.columns:
        df["etf_price"] = df["close"]
    else:
        raise ValueError("ETF 数据缺少 last / close，无法构造 ETF 价格")

    df = df.dropna(subset=["etf_price"])

    out = pd.DataFrame(index=df.index)
    out["etf_price"] = df["etf_price"]

    # ETF 正常连续竞价时间
    time_str = out.index.strftime("%H:%M:%S")
    morning = (time_str >= "09:30:00") & (time_str <= "11:30:00")
    afternoon = (time_str >= "13:00:00") & (time_str <= "15:00:00")
    out = out[morning | afternoon]

    out = out.resample("1min").last()
    out = out.dropna(subset=["etf_price"])

    return out


# ============================================================
# 3. TL 期货数据处理
# ============================================================

def parse_futures_datetime(df):
    """
    适配你的 TL 表：
    - 日期在 action_day 或 trading_day
    - trade_time 只有日内时间，例如 09:30:00.3
    - 有些被 Excel 显示成 30:00.3、29:00.3，这种说明小时显示被隐藏/破坏
    """
    df = df.copy()

    if "action_day" in df.columns:
        date_col = "action_day"
    elif "trading_day" in df.columns:
        date_col = "trading_day"
    else:
        raise ValueError("期货数据缺少 action_day / trading_day，无法构造完整时间戳")

    if "trade_time" not in df.columns:
        raise ValueError("期货数据缺少 trade_time 列")

    date_str = (
        df[date_col]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

    time_str = df["trade_time"].astype(str).str.strip()

    full_dt_str = date_str + " " + time_str
    dt = pd.to_datetime(full_dt_str, errors="coerce", utc=True).dt.tz_convert(None)

    fail_ratio = dt.isna().mean()

    if fail_ratio > 0.3:
        print("\n警告：较多期货 trade_time 无法直接解析。")
        print("这通常说明 trade_time 在 Excel 中被显示为 mm:ss.s，例如 30:00.3，而原始小时信息丢失。")
        print("建议直接读取原始 CSV，不要读取从 Excel 另存或复制后的文件。")
        print(f"解析失败比例: {fail_ratio:.2%}\n")

    df["datetime"] = dt
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime")

    return df


def prepare_futures_minute_data(fut_df, from_date=None, to_date=None):
    """
    TL 主力合约快照处理：
    1. action_day + trade_time 构造完整 datetime
    2. 价格优先使用 last，其次使用 bid/ask mid
    3. 过滤开盘前大量无效数据
    4. 重采样到 1分钟
    """
    df = fut_df.copy()

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]

    df = parse_futures_datetime(df)
    df = df.set_index("datetime").sort_index()
    df.index = pd.to_datetime(df.index, errors="coerce", utc=True).tz_convert(None)
    df = df[~df.index.isna()]
    df = df.sort_index()
    if from_date is not None:
        df = df[df.index >= pd.Timestamp(from_date)]

    if to_date is not None:
        to_dt = pd.Timestamp(to_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[df.index <= to_dt]

    numeric_cols = [
        "last", "bid_price1", "ask_price1",
        "volume", "amount", "open_interest"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["last", "bid_price1", "ask_price1"]:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan

    if {"bid_price1", "ask_price1"}.issubset(df.columns):
        bad_spread = df["bid_price1"] > df["ask_price1"]
        df.loc[bad_spread, ["bid_price1", "ask_price1"]] = np.nan
        df["mid_price"] = (df["bid_price1"] + df["ask_price1"]) / 2
    else:
        df["mid_price"] = np.nan

    if "last" in df.columns:
        df["futures_price"] = df["last"].fillna(df["mid_price"])
    else:
        df["futures_price"] = df["mid_price"]

    df = df.dropna(subset=["futures_price"])

    if FILTER_FUTURES_TRADING_TIME:
        time_str = df.index.strftime("%H:%M:%S")

        morning = (time_str >= "09:30:00") & (time_str <= "11:30:00")
        afternoon = (time_str >= "13:00:00") & (time_str <= "15:15:00")

        df = df[morning | afternoon]

    out = pd.DataFrame(index=df.index)
    out["futures_price"] = df["futures_price"]

    if "volume" in df.columns:
        out["volume"] = df["volume"]
    else:
        out["volume"] = 0

    if "open_interest" in df.columns:
        out["open_interest"] = df["open_interest"]
    else:
        out["open_interest"] = 0

    fut_min = out.resample("1min").agg({
        "futures_price": "last",
        "volume": "last",
        "open_interest": "last",
    })

    fut_min = fut_min.dropna(subset=["futures_price"])

    return fut_min


# ============================================================
# 4. 合并 ETF 和期货，构造价差
# ============================================================

def build_pair_panel():
    etf_raw = load_etf_snapshot_csv(ETF_FILE, START_DATE, END_DATE)
    fut_raw = pd.read_csv(FUTURES_FILE)

    etf = prepare_etf_minute_data(etf_raw)
    fut = prepare_futures_minute_data(fut_raw, START_DATE, END_DATE)

    etf = etf.sort_index()
    fut = fut.sort_index()

    print("\n=== ETF minute data ===")
    print(etf.head())
    print(etf.tail())
    print(etf.describe())
    print("ETF 时间范围:", etf.index.min(), "至", etf.index.max())
    print("ETF 样本量:", len(etf))

    print("\n=== Futures minute data ===")
    print(fut.head())
    print(fut.tail())
    print(fut.describe())
    print("期货时间范围:", fut.index.min(), "至", fut.index.max())
    print("期货样本量:", len(fut))

    etf_reset = etf.reset_index()
    etf_reset = etf_reset.rename(columns={etf_reset.columns[0]: "trade_time"})

    fut_reset = fut.reset_index()
    fut_reset = fut_reset.rename(columns={fut_reset.columns[0]: "trade_time"})

    panel = pd.merge_asof(
        etf_reset.sort_values("trade_time"),
        fut_reset.sort_values("trade_time"),
        on="trade_time",
        direction="backward",
        tolerance=pd.Timedelta("2min"),
    )

    panel = panel.dropna(subset=["etf_price", "futures_price"])
    panel = panel.set_index("trade_time").sort_index()

    if panel.empty:
        raise ValueError("ETF 和期货合并后为空，请检查时间戳解析、交易时间过滤、文件日期范围。")

    # 任务要求使用 ETF_Futures_Spread。
    # 如果你已有该列，可以改成直接读取。
    panel["ETF_Futures_Spread"] = panel["etf_price"] - panel["futures_price"]

    # ETF feed 的 OHLCV
    panel["open"] = panel["etf_price"]
    panel["high"] = panel["etf_price"]
    panel["low"] = panel["etf_price"]
    panel["close"] = panel["etf_price"]
    panel["volume"] = 0
    panel["openinterest"] = 0

    # 期货 feed 的 OHLCV
    fut_feed = pd.DataFrame(index=panel.index)
    fut_feed["open"] = panel["futures_price"]
    fut_feed["high"] = panel["futures_price"]
    fut_feed["low"] = panel["futures_price"]
    fut_feed["close"] = panel["futures_price"]
    fut_feed["volume"] = 0
    fut_feed["openinterest"] = 0

    print("\n=== Pair panel ===")
    print(panel[["etf_price", "futures_price", "ETF_Futures_Spread"]].head())
    print(panel[["etf_price", "futures_price", "ETF_Futures_Spread"]].describe())
    print("合并后时间范围:", panel.index.min(), "至", panel.index.max())
    print("合并后样本量:", len(panel))

    panel.to_csv(OUTPUT_DIR / "511090_ETF_TL_pair_panel.csv", encoding="utf-8-sig")

    return panel, fut_feed


# ============================================================
# 5. Backtrader DataFeed
# ============================================================

class PairETFData(bt.feeds.PandasData):
    lines = ("spread",)

    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", "openinterest"),
        ("spread", "ETF_Futures_Spread"),
    )


# ============================================================
# 6. OU 配对交易策略
# ============================================================

class OUPairTradingStrategy(bt.Strategy):
    params = dict(
        initial_theta=None,
        initial_mu=None,
        initial_sigma=None,
        train_bars=None,
        update_interval=180,
        rolling_window=None,

        entry_z=2.0,
        exit_z=0.5,
        z_cap=4.0,

        capital_limit=10_000_000,
        max_futures_position=8,
        futures_multiplier=10_000,

        printlog=False,
    )

    def __init__(self):
        self.etf = self.datas[0]
        self.fut = self.datas[1]

        self.theta = self.p.initial_theta
        self.mu = self.p.initial_mu
        self.sigma = self.p.initial_sigma

        self.spread_history = []
        self.z_records = []
        self.position_records = []
        self.param_records = []
        self.value_records = []

        # 0: 空仓
        # 1: 做多价差 = 买 ETF + 卖期货
        # -1: 做空价差 = 卖 ETF + 买期货
        self.spread_position = 0

        self.last_update_bar = None
        self.entry_count = 0

    def log(self, text):
        if self.p.printlog:
            dt = self.etf.datetime.datetime(0)
            print(f"{dt} | {text}")

    def _update_ou_params(self):
        if self.p.rolling_window is None:
            window = self.spread_history
        else:
            window = self.spread_history[-self.p.rolling_window:]

        params = estimate_ou_params(window, dt=1.0)

        if params is None:
            return False

        self.theta = params["theta"]
        self.mu = params["mu"]
        self.sigma = params["stationary_sigma"]

        dt = self.etf.datetime.datetime(0)

        self.param_records.append({
            "datetime": dt,
            "theta": params["theta"],
            "mu": params["mu"],
            "stationary_sigma": params["stationary_sigma"],
            "sigma_diffusion": params["sigma_diffusion"],
            "half_life_minutes": params["half_life_minutes"],
            "ar1_beta": params["ar1_beta"],
            "n_obs": params["n_obs"],
        })

        self.log(
            f"OU updated | theta={self.theta:.6f}, "
            f"mu={self.mu:.6f}, sigma={self.sigma:.6f}"
        )

        return True

    def _calc_target_contracts(self, abs_z):
        ratio = min(abs_z / self.p.z_cap, 1.0)
        contracts = int(round(self.p.max_futures_position * ratio))

        if abs_z >= self.p.entry_z:
            contracts = max(1, contracts)

        contracts = min(contracts, self.p.max_futures_position)

        return contracts

    def _set_pair_position(self, direction, contracts):
        """
        direction:
            1: 做多价差 = 买 ETF + 卖期货
           -1: 做空价差 = 卖 ETF + 买期货
            0: 平仓
        """
        etf_price = float(self.etf.close[0])
        fut_price = float(self.fut.close[0])

        if direction == 0 or contracts == 0:
            self.order_target_size(data=self.etf, target=0)
            self.order_target_size(data=self.fut, target=0)
            self.spread_position = 0
            return

        futures_notional = contracts * fut_price * self.p.futures_multiplier

        etf_shares_by_hedge = int(futures_notional / etf_price)
        etf_shares_by_capital = int(self.p.capital_limit / etf_price)

        etf_shares = min(etf_shares_by_hedge, etf_shares_by_capital)

        if etf_shares <= 0:
            return

        if direction == 1:
            # 做多价差：买 ETF + 卖期货
            target_etf = etf_shares
            target_fut = -contracts
        else:
            # 做空价差：卖 ETF + 买期货
            target_etf = -etf_shares
            target_fut = contracts

        self.order_target_size(data=self.etf, target=target_etf)
        self.order_target_size(data=self.fut, target=target_fut)

        if self.spread_position == 0:
            self.entry_count += 1

        self.spread_position = direction

    def next(self):
        dt = self.etf.datetime.datetime(0)
        spread = float(self.etf.spread[0])

        if np.isnan(spread):
            return

        self.spread_history.append(spread)

        self.value_records.append({
            "datetime": dt,
            "value": self.broker.getvalue(),
        })

        bar_no = len(self.spread_history)

        # 前 50% 数据作为训练期，不交易
        if bar_no < self.p.train_bars:
            self.z_records.append({
                "datetime": dt,
                "spread": spread,
                "z_score": np.nan,
                "mu": self.mu,
                "sigma": self.sigma,
                "theta": self.theta,
            })

            self.position_records.append({
                "datetime": dt,
                "etf_position": self.getposition(self.etf).size,
                "futures_position": self.getposition(self.fut).size,
                "spread_position": self.spread_position,
            })

            return

        # 每 180分钟动态更新 OU 参数
        should_update = False

        if self.last_update_bar is None:
            should_update = True
        elif bar_no - self.last_update_bar >= self.p.update_interval:
            should_update = True

        if should_update:
            updated = self._update_ou_params()
            if updated:
                self.last_update_bar = bar_no

        if self.sigma is None or self.sigma <= 0 or self.mu is None:
            return

        z = (spread - self.mu) / self.sigma
        abs_z = abs(z)

        self.z_records.append({
            "datetime": dt,
            "spread": spread,
            "z_score": z,
            "mu": self.mu,
            "sigma": self.sigma,
            "theta": self.theta,
        })

        # 交易规则
        if self.spread_position == 0:
            if z > self.p.entry_z:
                # 价差过高：卖空价差 = 卖 ETF + 买期货
                contracts = self._calc_target_contracts(abs_z)
                self._set_pair_position(direction=-1, contracts=contracts)
                self.log(f"SHORT SPREAD | z={z:.3f}, contracts={contracts}")

            elif z < -self.p.entry_z:
                # 价差过低：做多价差 = 买 ETF + 卖期货
                contracts = self._calc_target_contracts(abs_z)
                self._set_pair_position(direction=1, contracts=contracts)
                self.log(f"LONG SPREAD | z={z:.3f}, contracts={contracts}")

        else:
            if abs_z < self.p.exit_z:
                self._set_pair_position(direction=0, contracts=0)
                self.log(f"EXIT | z={z:.3f}")

            else:
                contracts = self._calc_target_contracts(abs_z)

                if self.spread_position == 1:
                    self._set_pair_position(direction=1, contracts=contracts)
                elif self.spread_position == -1:
                    self._set_pair_position(direction=-1, contracts=contracts)

        self.position_records.append({
            "datetime": dt,
            "etf_position": self.getposition(self.etf).size,
            "futures_position": self.getposition(self.fut).size,
            "spread_position": self.spread_position,
        })

    def stop(self):
        print("\n=== 策略内部统计 ===")
        print(f"入场次数: {self.entry_count}")

        if self.param_records:
            pd.DataFrame(self.param_records).to_csv(
                OUTPUT_DIR / "ou_dynamic_params.csv",
                index=False,
                encoding="utf-8-sig"
            )

            print("\n动态 OU 参数最后 5 条:")
            print(pd.DataFrame(self.param_records).tail())

        pd.DataFrame(self.z_records).to_csv(
            OUTPUT_DIR / "spread_zscore_records.csv",
            index=False,
            encoding="utf-8-sig"
        )

        pd.DataFrame(self.position_records).to_csv(
            OUTPUT_DIR / "position_records.csv",
            index=False,
            encoding="utf-8-sig"
        )

        pd.DataFrame(self.value_records).to_csv(
            OUTPUT_DIR / "portfolio_value_records.csv",
            index=False,
            encoding="utf-8-sig"
        )


# ============================================================
# 7. 绩效指标
# ============================================================

def calculate_metrics(value_records, initial_cash, final_value):
    value_df = pd.DataFrame(value_records)

    if value_df.empty:
        return {
            "absolute_return": final_value - initial_cash,
            "total_return": final_value / initial_cash - 1,
            "annual_return": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    value_df["datetime"] = pd.to_datetime(value_df["datetime"])
    value_df = value_df.drop_duplicates(subset=["datetime"])
    value_df = value_df.set_index("datetime").sort_index()

    daily_value = value_df["value"].resample("1D").last().dropna()
    total_return = final_value / initial_cash - 1

    if len(daily_value) >= 2:
        daily_ret = daily_value.pct_change().dropna()
        n_days = len(daily_ret)

        annual_return = (final_value / initial_cash) ** (252 / max(n_days, 1)) - 1

        if daily_ret.std() > 0:
            sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        else:
            sharpe = np.nan

        running_max = daily_value.cummax()
        drawdown = daily_value / running_max - 1
        max_drawdown = drawdown.min()

    else:
        annual_return = np.nan
        sharpe = np.nan
        max_drawdown = np.nan

    return {
        "absolute_return": final_value - initial_cash,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


# ============================================================
# 8. 绘图
# ============================================================

def plot_outputs():
    z_path = OUTPUT_DIR / "spread_zscore_records.csv"
    pos_path = OUTPUT_DIR / "position_records.csv"

    if not z_path.exists() or not pos_path.exists():
        print("没有找到绘图所需文件，跳过绘图")
        return

    z_df = pd.read_csv(z_path, parse_dates=["datetime"])
    pos_df = pd.read_csv(pos_path, parse_dates=["datetime"])

    z_df = z_df.dropna(subset=["z_score"])

    if z_df.empty:
        print("z_score 记录为空，跳过绘图")
        return

    plt.figure(figsize=(14, 6))
    plt.plot(z_df["datetime"], z_df["spread"], label="ETF-Futures Spread")
    plt.plot(z_df["datetime"], z_df["mu"], label="OU Long-term Mean")
    plt.title("ETF-Futures Spread and OU Mean")
    plt.xlabel("Time")
    plt.ylabel("Spread")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spread_series.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(z_df["datetime"], z_df["z_score"], label="z_score")
    plt.axhline(ENTRY_Z, linestyle="--", label="Entry +Z")
    plt.axhline(-ENTRY_Z, linestyle="--", label="Entry -Z")
    plt.axhline(EXIT_Z, linestyle=":", label="Exit +Z")
    plt.axhline(-EXIT_Z, linestyle=":", label="Exit -Z")
    plt.title("OU z-score")
    plt.xlabel("Time")
    plt.ylabel("z-score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "z_score_series.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(pos_df["datetime"], pos_df["etf_position"], label="ETF Position")
    plt.plot(pos_df["datetime"], pos_df["futures_position"], label="Futures Position")
    plt.title("Position Changes")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "position_changes.png", dpi=200)
    plt.close()

    print("图表已输出到:", OUTPUT_DIR)


# ============================================================
# 9. 主回测函数
# ============================================================

def run_backtest():
    panel, fut_feed_df = build_pair_panel()

    n_total = len(panel)
    train_bars = int(n_total * 0.5)

    if train_bars < 100:
        raise ValueError("训练样本太少，无法估计 OU 参数")

    train_spread = panel["ETF_Futures_Spread"].iloc[:train_bars]
    init_params = estimate_ou_params(train_spread, dt=1.0)

    if init_params is None:
        raise ValueError("初始 OU 参数估计失败。请检查价差序列是否近似均值回复，或样本是否足够。")

    print("\n=== 初始 OU 参数估计结果：前 50% 数据 ===")
    for k, v in init_params.items():
        print(f"{k}: {v}")

    pd.DataFrame([init_params]).to_csv(
        OUTPUT_DIR / "initial_ou_params.csv",
        index=False,
        encoding="utf-8-sig"
    )

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(INITIAL_CASH)

    # 允许空头现金处理
    cerebro.broker.set_shortcash(True)

    etf_data = PairETFData(
        dataname=panel,
        name="511090.SH",
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
    )

    fut_data = bt.feeds.PandasData(
        dataname=fut_feed_df,
        name="TL_main",
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
    )

    cerebro.adddata(etf_data)
    cerebro.adddata(fut_data)

    cerebro.addstrategy(
        OUPairTradingStrategy,
        initial_theta=init_params["theta"],
        initial_mu=init_params["mu"],
        initial_sigma=init_params["stationary_sigma"],
        train_bars=train_bars,
        update_interval=UPDATE_INTERVAL_BARS,
        rolling_window=train_bars,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        z_cap=Z_CAP_FOR_FULL_POSITION,
        capital_limit=CAPITAL_LIMIT,
        max_futures_position=MAX_FUTURES_POSITION,
        futures_multiplier=FUTURES_MULTIPLIER,
        printlog=False,
    )

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    print("\nStarting Portfolio Value:", cerebro.broker.getvalue())

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()

    print("Final Portfolio Value:", final_value)

    metrics = calculate_metrics(
        value_records=strat.value_records,
        initial_cash=INITIAL_CASH,
        final_value=final_value,
    )

    drawdown = strat.analyzers.drawdown.get_analysis()

    print("\n=== 回测绩效指标 ===")
    print(f"绝对收益: {metrics['absolute_return']:,.2f}")
    print(f"总收益率: {metrics['total_return']:.4%}")

    if pd.notna(metrics["annual_return"]):
        print(f"年化收益: {metrics['annual_return']:.4%}")
    else:
        print("年化收益: 无法计算")

    if pd.notna(metrics["sharpe"]):
        print(f"夏普比率: {metrics['sharpe']:.4f}")
    else:
        print("夏普比率: 无法计算")

    if pd.notna(metrics["max_drawdown"]):
        print(f"最大回撤: {metrics['max_drawdown']:.4%}")
    else:
        print("最大回撤: 无法计算")

    print(f"backtrader 最大回撤: {drawdown['max']['drawdown']:.4f}%")
    print(f"入场次数: {strat.entry_count}")

    pd.DataFrame([metrics]).to_csv(
        OUTPUT_DIR / "backtest_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plot_outputs()

    return strat, metrics


if __name__ == "__main__":
    run_backtest()