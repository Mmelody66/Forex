import streamlit as st
import pandas as pd
import numpy as np
import os
import itertools

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

    df["MA_short"] = price.rolling(ma_short).mean()
    df["MA_long"] = price.rolling(ma_long).mean()

    ema_fast = price.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = price.ewm(span=macd_slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()

    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def generate_signal(df):

    sig = pd.Series(0, index=df.index)

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


# ==============================
# Performance Metrics
# ==============================

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
    total_periods = len(equity)
    return equity.iloc[-1]**(freq / total_periods) - 1


# ==============================
# Sidebar
# ==============================

st.sidebar.header(" Strategy Parameters")

# 
if "ma_short" not in st.session_state:
    st.session_state.ma_short = 20

if "ma_long" not in st.session_state:
    st.session_state.ma_long = 60

#
ma_short = st.sidebar.slider(
    "MA Short",
    5, 50,
    value=st.session_state.ma_short
)

ma_long = st.sidebar.slider(
    "MA Long",
    20, 200,
    value=st.session_state.ma_long
)
pairs = [f.replace(".csv", "") for f in os.listdir(DATA_PATH)]
selected_pair = st.sidebar.selectbox("Currency Pair", pairs)

split_date = st.sidebar.date_input(
    "Train/Test Split Date",
    value=pd.to_datetime("2023-01-01")
)


macd_fast = st.sidebar.slider("MACD Fast", 5, 20, 12)
macd_slow = st.sidebar.slider("MACD Slow", 20, 40, 26)
macd_signal = st.sidebar.slider("MACD Signal", 5, 20, 9)

rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)

run_button = st.sidebar.button("▶ Run Backtest")
opt_button = st.sidebar.button("🔍 Optimize on Train")


# ==============================
# Load Data
# ==============================

df = pd.read_csv(
    os.path.join(DATA_PATH, f"{selected_pair}.csv"),
    index_col=0,
    parse_dates=True
)

split_date = pd.to_datetime(split_date)


# ==============================
# Optimization (Train only)
# ==============================

if opt_button:

    st.sidebar.write("Optimizing...")

    best_sharpe = -np.inf
    best_params = None

    ma_short_grid = [10, 20, 30]
    ma_long_grid = [40, 60, 80]

    for s, l in itertools.product(ma_short_grid, ma_long_grid):
        if s >= l:
            continue

        temp = df.copy()
        temp = add_indicators(temp, s, l,
                              macd_fast, macd_slow,
                              macd_signal, rsi_period)

        temp["Return"] = temp["Adj Close"].pct_change()
        temp["Signal"] = generate_signal(temp)
        temp["Position"] = temp["Signal"].shift(1)
        temp["Strategy_Return"] = temp["Position"] * temp["Return"]
        temp.dropna(inplace=True)

        train = temp[temp.index <= split_date]
        equity_train = (1 + train["Strategy_Return"]).cumprod()

        s_train = sharpe_ratio(train["Strategy_Return"])

        if s_train > best_sharpe:
            best_sharpe = s_train
            best_params = (s, l)

    st.sidebar.success(f"Best MA: {best_params}, Train Sharpe: {best_sharpe:.2f}")

    ma_short, ma_long = best_params

    st.session_state.ma_short = best_params[0]
    st.session_state.ma_long = best_params[1]


# ==============================
# Backtest
# ==============================

if run_button or opt_button:

    df = add_indicators(df, ma_short, ma_long,
                        macd_fast, macd_slow,
                        macd_signal, rsi_period)

    df["Return"] = df["Adj Close"].pct_change()
    df["Signal"] = generate_signal(df)
    df["Position"] = df["Signal"].shift(1)
    df["Strategy_Return"] = df["Position"] * df["Return"]
    df.dropna(inplace=True)

    df_train = df[df.index <= split_date]
    df_test = df[df.index > split_date]

# ==============================
# Metrics
# ==============================

    sharpe_train = sharpe_ratio(df_train["Strategy_Return"])
    sharpe_test = sharpe_ratio(df_test["Strategy_Return"])

    equity_train = (1 + df_train["Strategy_Return"]).cumprod()
    equity_test = (1 + df_test["Strategy_Return"]).cumprod()

    market_train = (1 + df_train["Return"]).cumprod()
    market_test = (1 + df_test["Return"]).cumprod()



    # Metrics
    total_return = equity_test.iloc[-1] - 1
    ann_return = annualized_return(equity_test)
    mdd = max_drawdown(equity_test)
    calmar = ann_return / abs(mdd) if mdd != 0 else 0
    trades_train = int((df_train["Position"].diff().abs() > 0).sum())
    trades_test = int((df_test["Position"].diff().abs() > 0).sum())

    # ==============================
    # Display
    # ==============================

    st.subheader(" Train Performance")
    st.line_chart(equity_train)

    st.subheader(" Test Performance")
    st.line_chart(pd.DataFrame({
        "Strategy": equity_test,
        "Market": market_test
    }))

    st.subheader(" Test Summary Metrics")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Return", f"{total_return:.2%}")
    col2.metric("Annual Return", f"{ann_return:.2%}")
    col3.metric("Train Sharpe", f"{sharpe_train:.2f}")
    col4.metric("Test Sharpe", f"{sharpe_test:.2f}")
    col5.metric("Max Drawdown", f"{mdd:.2%}")
    col6.metric("Calmar", f"{calmar:.2f}")

    st.write(f"Trades: {trades_test}")

    st.subheader(" Detailed Test Data")
    st.dataframe(df_test.tail(50))


