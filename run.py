import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(layout="wide")
st.title(" Forex Technical Analysis Prototype")

DATA_PATH = "data/raw"


# ==============================
# Indicator Functions
# ==============================

def add_indicators(df, ma_short, ma_long,
                   macd_fast, macd_slow, macd_signal,
                   rsi_period):

    price = df["Adj Close"]

    # Moving averages
    df["MA_short"] = price.rolling(ma_short).mean()
    df["MA_long"] = price.rolling(ma_long).mean()

    # MACD
    ema_fast = price.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = price.ewm(span=macd_slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()

    # RSI
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # --- Bollinger Bands (for bandwidth filter) ---
    n = 20      # standard BB window
    k = 2       # standard width multiplier

    mid = price.rolling(n).mean()
    std = price.rolling(n).std()

    df["BOLL_mid"] = mid
    df["BOLL_up"] = mid + k * std
    df["BOLL_low"] = mid - k * std

    # Bandwidth (normalized)
    df["BOLL_bw"] = (df["BOLL_up"] - df["BOLL_low"]) / df["BOLL_mid"]

    # Rolling median of bandwidth (regime threshold)
    df["BOLL_bw_med"] = df["BOLL_bw"].rolling(60).median()

    return df


def generate_signal(df):

    sig = pd.Series(0, index=df.index)
    vol_ok = df["BOLL_bw"] > df["BOLL_bw_med"]

    long_cond = (
        (df["MA_short"] > df["MA_long"]) &
        (df["MACD"] > df["MACD_signal"]) &
        (df["MACD"] > 0) &
        (df["RSI"] > 55)
    )

    short_cond = (
        (df["MA_short"] < df["MA_long"]) &
        (df["MACD"] < df["MACD_signal"]) &
        (df["MACD"] < 0) &
        (df["RSI"] < 45)
    )

    sig[long_cond] = 1
    sig[short_cond] = -1

    return sig


def sharpe_ratio(returns, freq=252):
    returns = returns.dropna()
    if returns.std() == 0:
        return 0
    return np.sqrt(freq) * returns.mean() / returns.std()


def max_drawdown(equity):
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd.min()


# ==============================
# Sidebar Controls
# ==============================

st.sidebar.header(" Strategy Parameters")

pairs = [f.replace(".csv", "") for f in os.listdir(DATA_PATH)]
selected_pair = st.sidebar.selectbox("Currency Pair", pairs)

ma_short = st.sidebar.slider("MA Short", 5, 50, 20)
ma_long = st.sidebar.slider("MA Long", 20, 200, 60)

macd_fast = st.sidebar.slider("MACD Fast", 5, 20, 12)
macd_slow = st.sidebar.slider("MACD Slow", 20, 40, 26)
macd_signal = st.sidebar.slider("MACD Signal", 5, 20, 9)

rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)

run_button = st.sidebar.button("▶ Run Backtest")


# ==============================
# Backtest Execution
# ==============================

if run_button:

    df = pd.read_csv(
        os.path.join(DATA_PATH, f"{selected_pair}.csv"),
        index_col=0,
        parse_dates=True
    )

    df = add_indicators(df,
                        ma_short, ma_long,
                        macd_fast, macd_slow, macd_signal,
                        rsi_period)

    df["Return"] = df["Adj Close"].pct_change()
    df["Signal"] = generate_signal(df)
    df["Position"] = df["Signal"].shift(1)

    df["Strategy_Return"] = df["Position"] * df["Return"]
    df.dropna(inplace=True)

    equity = (1 + df["Strategy_Return"]).cumprod()
    market = (1 + df["Return"]).cumprod()

    total_return = equity.iloc[-1] - 1
    market_return = market.iloc[-1] - 1
    sharpe = sharpe_ratio(df["Strategy_Return"])
    mdd = max_drawdown(equity)
    trades = int((df["Position"].diff().abs() > 0).sum())

    # ==============================
    # Display Metrics
    # ==============================

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Strategy Return", f"{total_return:.2%}")
    col2.metric("Market Return", f"{market_return:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col4.metric("Max Drawdown", f"{mdd:.2%}")
    col5.metric("Trades", trades)

    st.subheader(" Equity Curve")
    st.line_chart(pd.DataFrame({
        "Strategy": equity,
        "Market": market
    }))

    st.subheader("Detailed Data")
    st.dataframe(df)
