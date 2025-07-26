# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Simulation Dashboard",
    layout="wide"
)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Inputs")

# Tickers
tickers = [t.strip() for t in st.sidebar.text_input(
    "Tickers (Assume portfolio has the following)", 
    value="SPY,AGG,VNQ,DBC"
).upper().split(",") if t.strip()]

# Date range
col1, col2 = st.sidebar.columns(2)
start = col1.date_input("Start date", pd.to_datetime("2018-01-01"))
end   = col2.date_input("End date",   pd.to_datetime("2024-12-31"))

# Number of portfolios
n_portfolios = st.sidebar.slider(
    "Number of portfolios", 1000, 20_000, 5_000, step=500
)

# ─── Data Download ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Adj Close"] if "Adj Close" in df.columns.get_level_values(0) else df["Close"]
    else:
        prices = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    returns = prices.pct_change().dropna()
    return prices, returns

prices, returns = load_data(tickers, start, end)

# ─── Safety Check ───────────────────────────────────────────────────────────────
if returns.empty:
    st.error("No return data available for selected tickers and date range.")
    st.stop()

# ─── Compute Stats ─────────────────────────────────────────────────────────────
trading_days = len(pd.bdate_range(start=start, end=end))
mean_daily_ret = returns.mean()
cov_daily      = returns.cov()
mean_annual_ret = mean_daily_ret * trading_days
cov_annual      = cov_daily * trading_days

# Display stats
st.header("📊 Sample Statistics")
st.markdown(f"**Trading days in sample:** {trading_days}")
stats_df = pd.DataFrame({
    "Mean Ann. Return": mean_annual_ret,
    "Volatility (Ann.)": np.sqrt(np.diag(cov_annual))
})
st.dataframe(stats_df.style.format("{:.2%}"))

with st.expander("📥 Download Stats"):
    st.download_button("Download stats as CSV", stats_df.to_csv(), file_name="portfolio_stats.csv")

# ─── Monte Carlo Simulation ────────────────────────────────────────────────────
st.header("🎲 Monte Carlo Portfolio Simulation")

n_assets = len(mean_annual_ret)
results = np.zeros((3, n_portfolios))
weights_record = []

for i in range(n_portfolios):
    w = np.random.random(n_assets)
    w /= np.sum(w)
    port_ret = w @ mean_annual_ret
    port_vol = np.sqrt(w.T @ cov_annual @ w)
    results[0, i] = port_vol
    results[1, i] = port_ret
    results[2, i] = port_ret / port_vol
    weights_record.append(w)

# ─── Visualization ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots()
sc = ax.scatter(
    results[0], results[1],
    c=results[2], cmap="viridis", s=10, alpha=0.5
)
ax.set_xlabel("Annualized Volatility")
ax.set_ylabel("Annualized Return")
ax.set_title(f"Efficient Frontier ({trading_days} trading days)")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Sharpe Ratio")

# Highlight max Sharpe portfolio
max_sharpe_idx = results[2].argmax()
max_vol = results[0, max_sharpe_idx]
max_ret = results[1, max_sharpe_idx]
ax.scatter(max_vol, max_ret, color="red", marker="*", s=100, label="Max Sharpe")
ax.legend()

st.pyplot(fig)

# ─── Max Sharpe Portfolio Weights ──────────────────────────────────────────────
st.subheader("⭐ Max Sharpe Portfolio")
st.markdown(f"""
- **Annualized Return:** {max_ret:.2%}  
- **Annualized Volatility:** {max_vol:.2%}  
- **Sharpe Ratio:** {results[2, max_sharpe_idx]:.2f}
""")

opt_weights = weights_record[max_sharpe_idx]
opt_df = pd.DataFrame({
    "Ticker": mean_annual_ret.index,
    "Weight": opt_weights
}).set_index("Ticker")

st.dataframe(opt_df.style.format("{:.2%}"))

# ─── Show Data Samples ─────────────────────────────────────────────────────────
with st.expander("🔍 Show last 5 rows of prices & returns"):
    st.subheader("Prices")
    st.dataframe(prices.tail())
    st.subheader("Returns")
    st.dataframe(returns.tail())


