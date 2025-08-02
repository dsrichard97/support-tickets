# app.py – v3‑update on July 26th, 2025
###############################################################################
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import minimize
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# ─── Global Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Simulation Dashboard", layout="wide")
ALLOWED_TICKERS = ["SPY", "AGG", "VNQ", "DBC"]
TODAY = pd.Timestamp.today().normalize()
US_BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# ─── Title & Intro ────────────────────────────────────────────────────────────
st.title("📈 Portfolio Simulation Dashboard  (Last Updated 2025‑07‑26)")
st.markdown("""
Monte‑Carlo frontier, Markowitz optimizer, VaR / CVaR, and US‑holiday‑aware annualization.  
New: two **idea tabs** for future expansion (“Transaction Costs” & “Stress Testing”).
""")

# ─── Sidebar Inputs ───────────────────────────────────────────────────────────
st.sidebar.header("Inputs")

# -- Ticker whitelist
raw_tickers = st.sidebar.text_input("Tickers (allowed: SPY, AGG, VNQ, DBC)", value="SPY,AGG,VNQ,DBC").upper()
tickers = [t.strip() for t in raw_tickers.split(",") if t.strip()]
invalid = [t for t in tickers if t not in ALLOWED_TICKERS]
if invalid:
    st.sidebar.error(f"❌ Invalid ticker(s): {', '.join(invalid)}. Allowed: {ALLOWED_TICKERS}")
    st.stop()
if not tickers:
    st.sidebar.error("❌ Enter at least one ticker."); st.stop()

# -- Business-day-aware start/end selection
c1, c2 = st.sidebar.columns(2)
default_start = pd.date_range(end=TODAY, periods=5 * 252, freq=US_BDAY)[0]
default_end = TODAY

start = c1.date_input("Start date (business day)", default_start, max_value=TODAY)
end = c2.date_input("End date (business day)", default_end, min_value=start, max_value=TODAY)

start = pd.Timestamp(start)
end = pd.Timestamp(end)
if start not in pd.bdate_range(start, start):
    start = pd.bdate_range(start, periods=1)[0]
if end not in pd.bdate_range(end, end):
    end = pd.bdate_range(end, periods=1)[0]
if start >= end:
    st.sidebar.error("❌ End date must be after start date."); st.stop()

# -- Simulation size and controls
n_port = st.sidebar.slider("Number of random portfolios", 1_000, 20_000, 5_000, step=500)
rf = st.sidebar.slider("Risk‑free rate (annual %)", -1.0, 10.0, 0.0, step=0.25) / 100
run_opt = st.sidebar.button("🔧 Run Markowitz Optimizer")
show_tail = st.sidebar.checkbox("Tail‑risk metrics (VaR / CVaR)")
show_factor = st.sidebar.checkbox("Scenario & Factor Attribution (stub)")

# ─── Data Retrieval ───────────────────────────────────────────────────────────
@st.cache_data
def load_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Adj Close"] if "Adj Close" in df.columns.get_level_values(0) else df["Close"]
    else:
        prices = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    returns = prices.pct_change().dropna()
    return prices, returns

prices, returns = load_data(tickers, start, end)
if prices.empty or returns.empty:
    st.error("No price/return data retrieved. Try a broader date range."); st.stop()

# ─── US‑Holiday‑Aware Trading‑Day Count ───────────────────────────────────────
trading_days_index = pd.date_range(start, end, freq=US_BDAY)
trading_days = len(trading_days_index)
if trading_days == 0:
    st.error("No trading days in selected interval."); st.stop()

# ─── Annualized Statistics ────────────────────────────────────────────────────
mu_daily = returns.mean()
cov_daily = returns.cov()
mu_ann = mu_daily * trading_days
cov_ann = cov_daily * trading_days
ann_vol = np.sqrt(np.diag(cov_ann))

st.header("📊 Sample Statistics")
st.markdown(f"**Trading days (weekends & US holidays removed):** {trading_days}")
st.dataframe(pd.DataFrame({
    "Mean Ann. Return": mu_ann,
    "Volatility (Ann.)": ann_vol
}).style.format("{:.2%}"))

# ─── Monte‑Carlo Frontier ─────────────────────────────────────────────────────
st.header("🎲 Monte Carlo Frontier")
n_assets = len(mu_ann)
results = np.full((3, n_port), np.nan)  # vol, ret, sharpe
weights = []

for i in range(n_port):
    w = np.random.random(n_assets); w /= w.sum()
    vol = np.sqrt(w @ cov_ann @ w)
    if vol == 0: continue
    ret = w @ mu_ann
    shp = (ret - rf) / vol
    results[:, i] = [vol, ret, shp]
    weights.append(w)

mask = ~np.isnan(results).any(axis=0)
results = results[:, mask]
weights = [w for j, w in enumerate(weights) if mask[j]]
if results.size == 0:
    st.error("Simulation failed (all zero‑vol portfolios)."); st.stop()

fig, ax = plt.subplots()
pts = ax.scatter(results[0], results[1], c=results[2], cmap="viridis", s=10, alpha=0.5)
ax.set_xlabel("Annualized Volatility")
ax.set_ylabel("Annualized Return")
ax.set_title(f"Risk‑free {rf:.2%} • Trading days {trading_days}")
cbar = plt.colorbar(pts, ax=ax); cbar.set_label("Sharpe Ratio")

mc_idx = int(np.nanargmax(results[2]))
ax.scatter(results[0, mc_idx], results[1, mc_idx], marker="*", s=120, color="red", label="Max Sharpe (MC)")
ax.legend(); st.pyplot(fig)

st.markdown(
    f"**Monte Carlo Max‑Sharpe**  Return {results[1, mc_idx]:.2%} • Vol {results[0, mc_idx]:.2%} • Sharpe {results[2, mc_idx]:.2f}"
)
st.dataframe(pd.DataFrame({"Weight": weights[mc_idx]}, index=mu_ann.index).style.format("{:.2%}"))

# ─── Markowitz Optimizer (optional) ───────────────────────────────────────────
if run_opt:
    st.subheader("🔧 Markowitz Optimizer Result")

    def neg_sharpe(w): return -((w @ mu_ann - rf) / np.sqrt(w @ cov_ann @ w))
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    res = minimize(neg_sharpe, np.repeat(1 / n_assets, n_assets), bounds=bounds, constraints=cons)

    if not res.success:
        st.error("Optimization failed.")
    else:
        w_opt = res.x
        ret_opt = w_opt @ mu_ann
        vol_opt = np.sqrt(w_opt @ cov_ann @ w_opt)
        shp_opt = (ret_opt - rf) / vol_opt
        ax.scatter(vol_opt, ret_opt, marker="D", s=110, color="orange", label="Max Sharpe (OPT)")
        ax.legend(); st.pyplot(fig)

        st.markdown(
            f"**Optimizer Max‑Sharpe**  Return {ret_opt:.2%} • Vol {vol_opt:.2%} • Sharpe {shp_opt:.2f}"
        )
        st.dataframe(pd.DataFrame({"Weight": w_opt}, index=mu_ann.index).style.format("{:.2%}"))

# ─── Tail‑Risk Metrics (optional) ─────────────────────────────────────────────
if show_tail:
    st.subheader("📉 Tail‑Risk Metrics")
    conf = st.slider("Confidence level (%)", 90, 99, 95, step=1)
    w_ref = w_opt if run_opt and "w_opt" in locals() else weights[mc_idx]
    port_series = returns @ w_ref
    VaR = np.percentile(port_series, 100 - conf)
    CVaR = port_series[port_series <= VaR].mean()
    st.markdown(f"**Daily VaR ({conf}%)**: {VaR:.2%}    **Daily CVaR ({conf}%)**: {CVaR:.2%}")

# ─── Scenario / Factor Attribution stub (optional) ────────────────────────────
if show_factor:
    st.subheader("🧪 Scenario / Factor Attribution (coming soon)")
    st.info("Placeholder — integrate Fama‑French factors or macro‑shock stress scenarios here.")

# ─── Idea Tabs for Future Enhancements ────────────────────────────────────────
tab1, tab2 = st.tabs(["💸 Transaction Costs", "🌀 Stress Testing"])
with tab1:
    st.markdown("""
### Why consider transaction costs?
- **Slippage & bid–ask spreads** erode realized returns.  
- **Turnover constraints** can penalize frequent re‑balancing.

#### Possible implementation
1. **Estimate cost per trade** as a % of notional or a fixed ¢ per share.  
2. **Adjust expected returns**  
   $\\mu_i = \\mu_i - c_i$  
3. **Display cost‑adjusted frontier** versus the frictionless one.
""")

with tab2:
    st.markdown("""
### Stress‑test ideas
- **Historic drawdowns** (e.g., 2008, Mar‑2020) applied to current weights.  
- **Factor shocks** (∆Value, ∆Momentum) or **macro shifts** (rates ↑ 200 bp).

#### Possible implementation
1. Build a **shock vector Δr** and compute `new_ret = w · (μ + Δr)`.  
2. Re‑plot the portfolio point under stress, highlight change in Sharpe / VaR.  
3. Provide a **slider / dropdown** for user‑selectable scenarios.
""")

# ─── Data Peek ────────────────────────────────────────────────────────────────
with st.expander("🔍 Last five rows of raw data"):
    st.write("**Prices**");  st.dataframe(prices.tail())
    st.write("**Returns**"); st.dataframe(returns.tail())





