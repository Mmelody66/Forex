import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(layout="wide")
st.title(" Technical Analysis Prototype – Three Strategy Framework")

DATA_PATH = "data/raw"


# =====================================
# Indicator Calculations
# =====================================

def add_indicators(df,
                   rsi_period,
                   macd_fast, macd_slow, macd_signal,
                   bb_period, bb_k):

    price = df["Adj Close"]

    # ----- RSI -----
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ----- MACD -----
    ema_fast = price.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = price.ewm(span=macd_slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()

    # ----- Bollinger -----
    sma = price.rolling(bb_period).mean()
    std = price.rolling(bb_period).std()

    df["BOLL_mid"] = sma
    df["BOLL_up"] = sma + bb_k * std
    df["BOLL_low"] = sma - bb_k * std

    return df


# =====================================
# Signal Generation (5.2.2)
# =====================================

def generate_signal(df, strategy,
                    rsi_lower, rsi_upper):

    sig = pd.Series(0, index=df.index)

    if strategy == "RSI":

        sig[df["RSI"] < rsi_lower] = 1
        sig[df["RSI"] > rsi_upper] = -1

    elif strategy == "MACD":

        sig[df["MACD"] > df["MACD_signal"]] = 1
        sig[df["MACD"] < df["MACD_signal"]] = -1

    elif strategy == "Bollinger":

        sig[df["Adj Close"] < df["BOLL_low"]] = 1
        sig[df["Adj Close"] > df["BOLL_up"]] = -1

    return sig


# =====================================
# Performance Metrics
# =====================================

def sharpe_ratio(returns, freq=252):
    returns = returns.dropna()
    if returns.std() == 0:
        return 0
    return np.sqrt(freq) * returns.mean() / returns.std()


def max_drawdown(equity):
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd.min()


def annualized_return(equity, freq=252):
    n = len(equity)
    return equity.iloc[-1] ** (freq / n) - 1


# =====================================
# Sidebar Controls
# =====================================

st.sidebar.header("Strategy Selection")

pairs = [f.replace(".csv", "") for f in os.listdir(DATA_PATH)]
selected_pair = st.sidebar.selectbox("Currency Pair", pairs)

strategy = st.sidebar.radio(
    "Choose Strategy",
    ["RSI", "MACD", "Bollinger"]
)

split_date = st.sidebar.date_input(
    "Train/Test Split Date",
    value=pd.to_datetime("2023-01-01")
)

st.sidebar.header("RSI Parameters")

rsi_period = st.sidebar.slider("RSI Lookback", 5, 30, 14)
rsi_lower = st.sidebar.slider("RSI Lower Threshold", 10, 50, 30)
rsi_upper = st.sidebar.slider("RSI Upper Threshold", 50, 90, 70)

st.sidebar.header("MACD Parameters")

macd_fast = st.sidebar.slider("MACD Fast EMA", 5, 20, 12)
macd_slow = st.sidebar.slider("MACD Slow EMA", 20, 40, 26)
macd_signal = st.sidebar.slider("MACD Signal EMA", 5, 20, 9)

st.sidebar.header("Bollinger Parameters")

bb_period = st.sidebar.slider("BB Moving Average Period", 10, 50, 20)
bb_k = st.sidebar.slider("BB Std Multiplier", 1.0, 3.0, 2.0)

run_button = st.sidebar.button("▶ Run Backtest")


# =====================================
# Backtest
# =====================================

if run_button:

    df = pd.read_csv(
        os.path.join(DATA_PATH, f"{selected_pair}.csv"),
        index_col=0,
        parse_dates=True
    )

    df = add_indicators(df,
                        rsi_period,
                        macd_fast, macd_slow, macd_signal,
                        bb_period, bb_k)

    df["Return"] = df["Adj Close"].pct_change()
    df["Signal"] = generate_signal(df, strategy,
                                   rsi_lower, rsi_upper)

    # 5.2.3 Position Sizing: fixed ±1
    df["Position"] = df["Signal"].shift(1)
    df["Strategy_Return"] = df["Position"] * df["Return"]

    df.dropna(inplace=True)

    split_date = pd.to_datetime(split_date)

    df_train = df[df.index <= split_date]
    df_test = df[df.index > split_date]

    equity_train = (1 + df_train["Strategy_Return"]).cumprod()
    equity_test = (1 + df_test["Strategy_Return"]).cumprod()

    market_test = (1 + df_test["Return"]).cumprod()

    sharpe_train = sharpe_ratio(df_train["Strategy_Return"])
    sharpe_test = sharpe_ratio(df_test["Strategy_Return"])

    total_return = equity_test.iloc[-1] - 1
    ann_return = annualized_return(equity_test)
    mdd = max_drawdown(equity_test)
    calmar = ann_return / abs(mdd) if mdd != 0 else 0
    trades = int((df_test["Position"].diff().abs() > 0).sum())

    # =====================================
    # Display
    # =====================================

    st.subheader("Train Performance")
    st.line_chart(equity_train)

    st.subheader("Test Performance")
    st.line_chart(pd.DataFrame({
        "Strategy": equity_test,
        "Market": market_test
    }))

    st.subheader("Test Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Return", f"{total_return:.2%}")
    col2.metric("Annual Return", f"{ann_return:.2%}")
    col3.metric("Train Sharpe", f"{sharpe_train:.2f}")
    col4.metric("Test Sharpe", f"{sharpe_test:.2f}")
    col5.metric("Max Drawdown", f"{mdd:.2%}")

    st.write(f"Trades (Test): {trades}")



