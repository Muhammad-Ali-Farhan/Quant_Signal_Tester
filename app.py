import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Quant Signal Tester", layout="wide")

DEFAULT_ETFS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "USO", "EEM"]

DEFAULT_STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V",
    "LLY", "UNH", "XOM", "AVGO", "MA", "COST", "HD", "MRK", "ABBV", "PEP",
    "KO", "WMT", "BAC", "CVX", "ADBE", "CRM", "MCD", "NFLX", "CSCO", "ACN"
]

BENCHMARK_OPTIONS = [
    "None",
    # US Equity
    "SPY", "IVV", "VOO", "VTI", "ITOT",
    "QQQ", "DIA", "IWM",
    "RSP", "VUG", "VTV",
    "XLF", "XLK", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU", "XLC", "XLRE",
    # International Equity
    "VEA", "EFA", "VXUS", "IXUS",
    "VWO", "EEM",
    # Bonds / Rates / Credit
    "AGG", "BND",
    "IEF", "IEI", "SHY",
    "TLT", "TIP",
    "LQD", "HYG",
    # Real assets / Commodities
    "GLD", "IAU",
    "USO", "DBC",
    # REITs
    "VNQ", "IYR",
]

DEFAULT_BENCHMARK = "SPY"

@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_prices(tickers: list[str], start: str) -> pd.DataFrame:
    
    if not tickers:
        return pd.DataFrame()

    tickers = [t.upper().strip() for t in tickers if t and t.strip()]
    tickers = sorted(set(tickers))

    def _download():
        data = yf.download(
            tickers,
            start=start,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
        )

        if isinstance(data, pd.DataFrame) and "Close" in data.columns:
            close = data["Close"]
        else:
            close = data.get("Adj Close", pd.DataFrame())

        if isinstance(close, pd.Series):
            close = close.to_frame()

        close = close.dropna(how="all").dropna(axis=1, how="all")
        return close

    px = _download()
    if px.empty:
        time.sleep(2.0)
        px = _download()

    return px

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def compute_signals(px: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mom = px.pct_change(252) - px.pct_change(21)

    ma50 = px.rolling(50).mean()
    ma200 = px.rolling(200).mean()
    ma_sig = (ma50 > ma200).astype(float)

    rsi14 = px.apply(rsi)
    rsi_sig = (rsi14 < 30).astype(float)

    score = (
        mom.rank(axis=1, pct=True) * 0.55
        + ma_sig * 0.30
        + rsi_sig * 0.15
    )

    return {
        "momentum": mom,
        "ma_signal": ma_sig,
        "rsi": rsi14,
        "rsi_signal": rsi_sig,
        "score": score,
    }

def resample_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "Daily":
        return index

    s = pd.Series(index=index, data=1)
    if freq == "Weekly":
        d = s.groupby(pd.Grouper(freq="W-MON")).apply(lambda x: x.index.min())
    elif freq == "Monthly":
        d = s.groupby(pd.Grouper(freq="MS")).apply(lambda x: x.index.min())
    else:
        return index

    d = pd.to_datetime(d.dropna().values)
    d = pd.DatetimeIndex(d)
    d = d[d.isin(index)]
    return d

def backtest_topk_long_only(
    px: pd.DataFrame,
    score: pd.DataFrame,
    k: int,
    rebalance: str,
    cost_bps: int,
    vol_target: float,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    rets = px.pct_change().fillna(0.0)

    dates = px.index
    reb_dates = resample_rebalance_dates(dates, rebalance)

    
    score_lag = score.shift(1)

    desired_w = pd.DataFrame(0.0, index=dates, columns=px.columns)
    for d in reb_dates:
        row = score_lag.loc[d].dropna()
        if row.empty:
            continue
        top = row.sort_values(ascending=False).head(k).index
        desired_w.loc[d, top] = 1.0 / len(top)

    
    w = desired_w.replace(0.0, np.nan).ffill().fillna(0.0)

    
    port_raw = (w * rets).sum(axis=1)
    vol = port_raw.rolling(20).std() * np.sqrt(252)
    scale = (vol_target / (vol.replace(0, np.nan))).clip(0, 2.0).fillna(0.0)
    w = w.mul(scale, axis=0)

    
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (cost_bps / 10000.0)

    port_ret = (w * rets).sum(axis=1) - costs
    equity = (1.0 + port_ret).cumprod()

    return port_ret, equity, w, turnover

def perf_stats(daily_ret: pd.Series) -> dict[str, float]:
    eps = 1e-12
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 2:
        return {"CAGR": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "WinRate": np.nan}

    equity = (1 + daily_ret).cumprod()
    years = len(daily_ret) / 252.0

    cagr = equity.iloc[-1] ** (1 / max(years, 1e-9)) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = cagr / (ann_vol + eps)

    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    win_rate = (daily_ret > 0).mean()

    return {
        "CAGR": float(cagr),
        "AnnVol": float(ann_vol),
        "Sharpe": float(sharpe),
        "MaxDD": float(max_dd),
        "WinRate": float(win_rate),
    }

st.markdown("# Quant Signal Tester")
st.caption("Signals • Backtest • Monitoring")

with st.sidebar:
    st.header("Controls")
    with st.form("controls_form"):
        universe_mode = st.selectbox("Universe", ["ETFs", "Large-cap Stocks"])
        default_universe = DEFAULT_ETFS if universe_mode == "ETFs" else DEFAULT_STOCKS

        tickers = st.multiselect("Tickers", default_universe, default=default_universe)

        start = st.text_input("Start date (YYYY-MM-DD)", "2018-01-01")

        rebalance = st.selectbox("Rebalance frequency", ["Daily", "Weekly", "Monthly"], index=1)
        k = st.slider("Top-K holdings", 1, min(10, max(2, len(tickers))), 3)

        cost_bps = st.slider("Transaction cost (bps per turnover)", 0, 25, 5)
        vol_target = st.slider("Vol target (annual)", 0.05, 0.30, 0.15, step=0.01)

        default_bench_index = BENCHMARK_OPTIONS.index(DEFAULT_BENCHMARK) if DEFAULT_BENCHMARK in BENCHMARK_OPTIONS else 1
        benchmark = st.selectbox("Benchmark (comparison)", BENCHMARK_OPTIONS, index=default_bench_index)

        apply = st.form_submit_button("Apply")

if not apply:
    st.info("Adjust settings in the sidebar, then click **Apply**.")
    st.stop()

if len(tickers) < 2:
    st.warning("Please select at least 2 tickers.")
    st.stop()

px = load_prices(tickers, start=start)

if px.empty or px.index.isna().all() or px.shape[1] < 2:
    st.error("Unable to load sufficient price data for the selected inputs. Please adjust the universe or start date and try again.")
    st.stop()

loaded_tickers = list(px.columns)
start_date = px.index.min().date()
end_date = px.index.max().date()

bench_eq = None
bench_label = None

if benchmark and benchmark != "None":
    bench_px = load_prices([benchmark], start=start)

    
    if bench_px is not None and not bench_px.empty and bench_px.shape[0] >= 50:
        bench_px = bench_px.reindex(px.index).ffill()
        b = bench_px.iloc[:, 0]
        if not b.isna().all():
            bench_ret = b.pct_change().fillna(0.0)
            bench_eq = (1 + bench_ret).cumprod()
            bench_label = benchmark

st.markdown("### Configuration")
st.markdown(
    f"""
**Universe:** `{universe_mode}`  
**Tickers ({len(loaded_tickers)}):** {", ".join(loaded_tickers)}  
**Range:** `{start_date}` → `{end_date}`  
**Rebalance:** `{rebalance}` &nbsp;&nbsp;|&nbsp;&nbsp; **Top-K:** `{k}` &nbsp;&nbsp;|&nbsp;&nbsp; **Costs:** `{cost_bps} bps` &nbsp;&nbsp;|&nbsp;&nbsp; **Vol target:** `{vol_target:.2f}`
"""
)

sig = compute_signals(px)

port_ret, equity, w, turnover = backtest_topk_long_only(
    px=px,
    score=sig["score"],
    k=k,
    rebalance=rebalance,
    cost_bps=cost_bps,
    vol_target=vol_target,
)

stats = perf_stats(port_ret)

st.markdown("### Performance Summary")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("CAGR", f"{stats['CAGR']*100:.1f}%" if np.isfinite(stats["CAGR"]) else "—")
c2.metric("Annual Vol", f"{stats['AnnVol']*100:.1f}%" if np.isfinite(stats["AnnVol"]) else "—")
c3.metric("Sharpe", f"{stats['Sharpe']:.2f}" if np.isfinite(stats["Sharpe"]) else "—")
c4.metric("Max Drawdown", f"{stats['MaxDD']*100:.1f}%" if np.isfinite(stats["MaxDD"]) else "—")
c5.metric("Win Rate", f"{stats['WinRate']*100:.1f}%" if np.isfinite(stats["WinRate"]) else "—")

st.markdown("### Equity Curve (Normalized)")
eq_df = pd.DataFrame({"Strategy": equity})
if bench_eq is not None and bench_label is not None:
    eq_df[f"Benchmark ({bench_label})"] = bench_eq
st.line_chart(eq_df)

st.markdown("### Signal Snapshot (Most Recent Trading Day)")

latest_date = px.index.max()
latest = pd.DataFrame(
    {
        "Price": px.loc[latest_date],
        "Momentum(12-1)": sig["momentum"].loc[latest_date],
        "MA(50>200)": sig["ma_signal"].loc[latest_date],
        "RSI(14)": sig["rsi"].loc[latest_date],
        "Score": sig["score"].loc[latest_date],
        "Weight": w.loc[latest_date],
    }
).sort_values("Score", ascending=False)

st.dataframe(
    latest.style.format(
        {
            "Price": "{:.2f}",
            "Momentum(12-1)": "{:.2%}",
            "RSI(14)": "{:.1f}",
            "Score": "{:.3f}",
            "Weight": "{:.2%}",
        }
    ),
    use_container_width=True,
)

st.markdown("### Turnover (Last 90 Trading Days)")
st.line_chart(turnover.tail(90))
