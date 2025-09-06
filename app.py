# app.py
"""
Stock Buy Helper Agent (Beginner-Friendly)
------------------------------------------
A single-file Streamlit app that analyzes a stock using only:
- yfinance
- pandas
- numpy
- streamlit

How to run:
pip install streamlit yfinance pandas numpy
streamlit run app.py

Notes:
- All scoring is rule-based and deterministic (no black-box AI).
- Designed for explainability with tips, tooltips, and a pre-buy checklist.
- Gracefully handles missing data ("N/A") without crashing.

Educational only — not financial advice.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ------------------------------ #
# --------- UTILITIES ---------- #
# ------------------------------ #

APP_NAME = "Stock Buy Helper Agent"
DEFAULT_TICKER = "AAPL"
TZ = timezone.utc  # yfinance returns UTC-stamped data

# Streamlit page config
st.set_page_config(
    page_title=f"{APP_NAME}",
    page_icon="📈",
    layout="wide"
)

# ------------------------------ #
# ------ SIDEBAR: INPUTS ------- #
# ------------------------------ #

def init_session_state():
    """Initialize Streamlit session state for persistent UI values."""
    if "last_ticker" not in st.session_state:
        # Use new query params API (replaces deprecated experimental_get_query_params)
        qp = st.query_params
        last = qp.get("ticker", [DEFAULT_TICKER])[0] if "ticker" in qp else DEFAULT_TICKER
        st.session_state["last_ticker"] = last.upper().strip() or DEFAULT_TICKER
    if "weights" not in st.session_state:
        # Default equal-ish weights (user can change)
        st.session_state["weights"] = {
            "Growth": 3.0,
            "Profitability": 3.0,
            "Valuation": 3.0,
            "Dividend Quality": 2.0,
            "Risk": 3.0,
        }

init_session_state()

with st.sidebar:
    st.title("🧭 Controls")

    st.markdown("**Ticker Input**")
    ticker_input = st.text_input(
        "Enter a stock ticker",
        value=st.session_state["last_ticker"],
        help="Example: AAPL, MSFT, NVDA. US stocks preferred for best data coverage."
    )

    # Quick links to Yahoo Finance
    safe_ticker = (ticker_input or DEFAULT_TICKER).upper().strip()
    y_base = f"https://finance.yahoo.com/quote/{safe_ticker}"
    st.markdown(
        f"""
        **Quick Links**
        - [Quote]({y_base})
        - [Financials]({y_base}/financials)
        - [Analysis]({y_base}/analysis)
        - [Holders]({y_base}/holders)
        """,
        help="Open the same ticker on Yahoo Finance for extra context."
    )

    st.markdown("---")
    st.markdown("**Score Weights (0–5)**")
    w_growth = st.slider("Growth", 0.0, 5.0, st.session_state["weights"]["Growth"], 0.5,
                         help="Higher weight emphasizes growth metrics like revenue/EPS CAGR.")
    w_profit = st.slider("Profitability", 0.0, 5.0, st.session_state["weights"]["Profitability"], 0.5,
                         help="Higher weight emphasizes margins and returns (ROE/ROA).")
    w_valuation = st.slider("Valuation", 0.0, 5.0, st.session_state["weights"]["Valuation"], 0.5,
                            help="Higher weight emphasizes lower P/E, EV/EBITDA, PEG, Price/Book.")
    w_dividend = st.slider("Dividend Quality", 0.0, 5.0, st.session_state["weights"]["Dividend Quality"], 0.5,
                           help="Higher weight emphasizes yield, payout safety, and dividend growth.")
    w_risk = st.slider("Risk", 0.0, 5.0, st.session_state["weights"]["Risk"], 0.5,
                       help="Higher weight penalizes leverage, dilution, negative FCF, and high volatility.")

    # Save back to session state
    st.session_state["weights"] = {
        "Growth": w_growth,
        "Profitability": w_profit,
        "Valuation": w_valuation,
        "Dividend Quality": w_dividend,
        "Risk": w_risk,
    }

    st.markdown("---")
    with st.expander("📘 Learning Aid (Beginner Tips)", expanded=False):
        st.markdown(
            """
- **Start simple**: focus on one company at a time.
- **3 pillars**: *Business Quality*, *Valuation*, *Risk*.
- **Margin of safety**: even great companies can be bad investments if price is too high.
- **Diversify**: avoid putting too much in a single stock.
- **Time horizon**: long-term thinking helps reduce noise from daily price swings.
            """
        )

    if st.button("🔄 Analyze"):
        # Update the stored last ticker and URL query params (replaces experimental_set_query_params)
        st.session_state["last_ticker"] = safe_ticker
        st.query_params["ticker"] = safe_ticker

# ------------------------------ #
# ------- CACHING HELPERS ------ #
# ------------------------------ #

@st.cache_data(show_spinner=False, ttl=60 * 15)
def fetch_history_df(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch price history as a DataFrame with columns like Open/High/Low/Close/Volume.
    Returns empty DataFrame if failure.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            return df
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_dividends_series(ticker: str) -> pd.Series:
    """Fetch dividend history as a Series (Date index -> dividend per share)."""
    try:
        t = yf.Ticker(ticker)
        s = t.dividends  # pandas Series
        if isinstance(s, pd.Series) and not s.empty:
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_news_list(ticker: str) -> List[dict]:
    """
    Fetch latest news from yfinance. Returns list of dicts with keys like:
    'title','link','publisher','providerPublishTime'
    """
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        # Keep only needed keys to ensure cache-serializable and light
        cleaned = []
        for item in news[:10]:  # fetch a few more; we'll display up to 5
            cleaned.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "publisher": item.get("publisher"),
                "time": item.get("providerPublishTime"),
            })
        return cleaned
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_fast_info(ticker: str) -> Dict:
    """
    Fetch fast info (price, market cap, etc.) and some .get_info fallbacks.
    Returns dict (cache-friendly).
    """
    out = {}
    try:
        t = yf.Ticker(ticker)
        # fast_info is quick and structured
        try:
            fi = t.fast_info
            if fi:
                out.update(dict(fi))
        except Exception:
            pass
        # Add some fields from get_info if available (slower; optional)
        try:
            gi = t.get_info()
            for k in [
                "beta", "trailingPE", "forwardPE", "priceToBook", "pegRatio",
                "enterpriseValue", "enterpriseToEbitda", "trailingEps", "forwardEps",
                "dividendYield", "payoutRatio", "bookValue", "sharesOutstanding",
                "averageDailyVolume3Month"
            ]:
                if k in gi:
                    out[k] = gi.get(k)
        except Exception:
            pass
    except Exception:
        pass
    return out

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_income_stmt(ticker: str) -> pd.DataFrame:
    """Annual income statement (most recent years)."""
    try:
        t = yf.Ticker(ticker)
        df = t.get_income_stmt(freq="annual")
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_balance_sheet(ticker: str) -> pd.DataFrame:
    """Annual balance sheet."""
    try:
        t = yf.Ticker(ticker)
        df = t.get_balance_sheet(freq="annual")
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_cashflow_stmt(ticker: str) -> pd.DataFrame:
    """Annual cash flow statement."""
    try:
        t = yf.Ticker(ticker)
        df = t.get_cashflow(freq="annual")
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_shares_history(ticker: str) -> pd.Series:
    """
    Fetch shares outstanding history if available (may be sparse).
    Returns Series indexed by date.
    """
    try:
        t = yf.Ticker(ticker)
        # Try newer API first
        try:
            s = t.get_shares_full()
            if isinstance(s, pd.Series) and not s.empty:
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
            elif isinstance(s, pd.DataFrame) and not s.empty:
                col = None
                for c in s.columns:
                    if "share" in c.lower():
                        col = c
                        break
                if col:
                    ser = s[col].copy()
                    ser.index = pd.to_datetime(ser.index)
                    return ser.dropna().sort_index()
        except Exception:
            pass

        # Fallback to single point sharesOutstanding from info
        info = fetch_fast_info(ticker)
        so = info.get("sharesOutstanding")
        if so:
            today = pd.Timestamp.today(tz=TZ).normalize()
            return pd.Series({today: float(so)})
    except Exception:
        pass
    return pd.Series(dtype=float)


# ------------------------------ #
# ---- DATA EXTRACTION UTILS ---- #
# ------------------------------ #

def _normalize_is_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the Income Statement (or other statement) has rows as line items
    and columns as periods. Transpose if needed.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    test_keys = [k.lower() for k in ["Total Revenue", "Net Income", "Gross Profit", "Operating Income",
                                     "Diluted EPS", "Basic EPS"]]
    idx = [str(i).lower() for i in df.index]
    cols = [str(c).lower() for c in df.columns]
    if any(k in idx for k in test_keys):
        return df.copy()
    if any(k in cols for k in test_keys):
        return df.T.copy()
    return df.copy()

def _latest_value_from(df: pd.DataFrame, possible_rows: List[str]) -> Tuple[Optional[float], Optional[pd.Timestamp]]:
    """
    Find latest numeric value for any of the 'possible_rows' from a financial statement DataFrame.
    Returns (value, period) or (None, None) if not found.
    """
    if df is None or df.empty:
        return None, None
    df = _normalize_is_df(df)
    lower_index = {str(i).lower(): i for i in df.index}
    target_row = None
    for key in possible_rows:
        k = key.lower()
        if k in lower_index:
            target_row = lower_index[k]
            break
        for li_lower, li_orig in lower_index.items():
            if k in li_lower:
                target_row = li_orig
                break
        if target_row is not None:
            break
    if target_row is None:
        return None, None
    row = df.loc[target_row].dropna()
    if row.empty:
        return None, None
    try:
        cols = pd.to_datetime(row.index)
        latest_idx = cols.argmax()
        latest_col = row.index[latest_idx]
    except Exception:
        latest_col = row.index[-1]
    try:
        val = float(row[latest_col])
    except Exception:
        return None, None
    try:
        period = pd.to_datetime(latest_col)
    except Exception:
        period = None
    return val, period

def _series_from_row(df: pd.DataFrame, possible_rows: List[str]) -> pd.Series:
    """
    Return a time series (columns over time) for the given line item.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    df = _normalize_is_df(df)
    lower_index = {str(i).lower(): i for i in df.index}
    target_row = None
    for key in possible_rows:
        k = key.lower()
        if k in lower_index:
            target_row = lower_index[k]
            break
        for li_lower, li_orig in lower_index.items():
            if k in li_lower:
                target_row = li_orig
                break
        if target_row is not None:
            break
    if target_row is None:
        return pd.Series(dtype=float)
    s = df.loc[target_row]
    s = pd.to_numeric(s, errors="coerce").dropna()
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass
    return s.sort_index()

def fmt_money(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "N/A"
    try:
        abs_v = abs(v)
        sign = "-" if v < 0 else ""
        if abs_v >= 1e12:
            return f"{sign}${abs_v/1e12:,.2f}T"
        if abs_v >= 1e9:
            return f"{sign}${abs_v/1e9:,.2f}B"
        if abs_v >= 1e6:
            return f"{sign}${abs_v/1e6:,.2f}M"
        if abs_v >= 1e3:
            return f"{sign}${abs_v/1e3:,.0f}K"
        return f"{sign}${abs_v:,.2f}"
    except Exception:
        return "N/A"

def fmt_num(v: Optional[float], decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "N/A"
    try:
        return f"{v:,.{decimals}f}"
    except Exception:
        return "N/A"

def fmt_pct(v: Optional[float], decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "N/A"
    try:
        return f"{100*v:.{decimals}f}%"
    except Exception:
        return "N/A"

def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b is None:
            return None
        if b == 0:
            return None
        return a / b
    except Exception:
        return None

def cagr(series: pd.Series, years: int) -> Optional[float]:
    """
    Compute CAGR using first and last available points over 'years' horizon if possible.
    Returns None if insufficient data.
    """
    if series is None or series.empty:
        return None
    cutoff_start = series.index.max() - pd.DateOffset(years=years, days=60)  # buffer
    series = series[series.index >= cutoff_start]
    if series.shape[0] < 2:
        return None
    first = float(series.iloc[0])
    last = float(series.iloc[-1])
    delta_days = (series.index[-1] - series.index[0]).days
    years_exact = max(delta_days / 365.25, 0.5)
    if first <= 0 or years_exact <= 0:
        return None
    try:
        return (last / first) ** (1 / years_exact) - 1
    except Exception:
        return None

def rolling_rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    try:
        if close is None or close.empty or len(close) < period + 1:
            return None
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else None
    except Exception:
        return None

def decision_color(decision: str) -> str:
    if "BUY" in decision:
        return "#12B886"  # green
    if "WATCH" in decision or "NEUTRAL" in decision:
        return "#FAB005"  # yellow
    return "#FA5252"  # red


# ------------------------------ #
# --------- MAIN LOGIC --------- #
# ------------------------------ #

ticker = (ticker_input or DEFAULT_TICKER).upper().strip()
st.title(f"📈 {APP_NAME}")
st.caption("Explainable, beginner-friendly rules engine. Uses free Yahoo Finance data via `yfinance`.")

# Fetch data
hist_5y = fetch_history_df(ticker, period="5y", interval="1d")
hist_1y = fetch_history_df(ticker, period="1y", interval="1d")
dividends = fetch_dividends_series(ticker)
news_items = fetch_news_list(ticker)
fast_info = fetch_fast_info(ticker)
income_df = fetch_income_stmt(ticker)
balance_df = fetch_balance_sheet(ticker)
cashflow_df = fetch_cashflow_stmt(ticker)
shares_hist = fetch_shares_history(ticker)

# SNAPSHOT
with st.container():
    st.subheader("📸 Snapshot", help="Quick view of price, range, size, and volatility.")
    col1, col2, col3, col4, col5 = st.columns(5)

    # Current price & daily change
    last_price = fast_info.get("last_price")
    prev_close = fast_info.get("previous_close")
    if last_price is None and not hist_1y.empty:
        last_price = float(hist_1y["Close"].iloc[-1])
    if prev_close is None and not hist_1y.empty and len(hist_1y) >= 2:
        prev_close = float(hist_1y["Close"].iloc[-2])

    daily_change = safe_div((last_price - prev_close) if (last_price is not None and prev_close is not None) else None, prev_close)

    # 52-week range from 1Y history
    wk52_low, wk52_high = None, None
    if not hist_1y.empty:
        wk52_low = float(hist_1y["Low"].min())
        wk52_high = float(hist_1y["High"].max())

    # Market cap
    market_cap = fast_info.get("market_cap")

    # Beta (volatility proxy)
    beta = fast_info.get("beta")

    with col1:
        st.metric("Price", fmt_money(last_price), fmt_pct(daily_change) if daily_change is not None else None,
                  help="Latest trading price.")
    with col2:
        if wk52_low is not None and wk52_high is not None:
            st.metric("52-Week Range", f"{fmt_money(wk52_low)} – {fmt_money(wk52_high)}",
                      help="Lowest and highest prices in the last 52 weeks.")
        else:
            st.metric("52-Week Range", "N/A")
    with col3:
        st.metric("Market Cap", fmt_money(market_cap),
                  help="Total market value of all outstanding shares.")
    with col4:
        st.metric("Beta", fmt_num(beta, 2) if beta is not None else "N/A",
                  help="Volatility vs. market (≈1 = market-like, >1 = more volatile).")
    with col5:
        near_flag = ""
        if last_price and wk52_high and wk52_low:
            dist_high = (wk52_high - last_price) / wk52_high if wk52_high != 0 else None
            dist_low = (last_price - wk52_low) / wk52_low if wk52_low != 0 else None
            if dist_high is not None and dist_high <= 0.05:
                near_flag = "🟢 Near 52W High"
            elif dist_low is not None and dist_low <= 0.05:
                near_flag = "🔴 Near 52W Low"
        st.metric("Range Position", near_flag or "—",
                  help="Highlights if price is within ~5% of 52-week extremes.")

# FUNDAMENTALS
st.subheader("🏗️ Fundamentals", help="Revenue, earnings, margins, leverage, cash & free cash flow, ROE/ROA.")

# Extract core items
rev_latest, _ = _latest_value_from(income_df, ["Total Revenue"])
net_income_latest, _ = _latest_value_from(income_df, ["Net Income"])
eps_diluted_latest, _ = _latest_value_from(income_df, ["Diluted EPS", "Basic EPS"])
if eps_diluted_latest is None:
    eps_diluted_latest = fast_info.get("trailingEps")

gross_profit_latest, _ = _latest_value_from(income_df, ["Gross Profit"])
op_income_latest, _ = _latest_value_from(income_df, ["Operating Income"])

# Margins
gross_margin = safe_div(gross_profit_latest, rev_latest)
oper_margin = safe_div(op_income_latest, rev_latest)
net_margin = safe_div(net_income_latest, rev_latest)

# Debt/Equity, Cash
total_debt, _ = _latest_value_from(balance_df, ["Total Debt", "Short Long Term Debt", "Long Term Debt"])
if total_debt is None:
    s_debt, _ = _latest_value_from(balance_df, ["Short Long Term Debt", "Short Long-Term Debt"])
    l_debt, _ = _latest_value_from(balance_df, ["Long Term Debt", "Long-Term Debt"])
    if s_debt is not None or l_debt is not None:
        total_debt = (s_debt or 0.0) + (l_debt or 0.0)

total_equity, _ = _latest_value_from(balance_df, ["Total Stockholder Equity", "Total Equity", "Total Equity Gross Minority Interest"])
total_assets, _ = _latest_value_from(balance_df, ["Total Assets"])

# Cash & Equivalents (best-effort search)
cash_candidates = [
    "Cash And Cash Equivalents",
    "Cash And Cash Equivalents, at Carrying Value",
    "Cash",
    "Cash Cash Equivalents And Short Term Investments"
]
cash_latest, _ = _latest_value_from(balance_df, cash_candidates)

# Free Cash Flow (Operating CF + CapEx)
op_cf, _ = _latest_value_from(cashflow_df, ["Operating Cash Flow", "Net Cash Provided by Operating Activities", "Cash Flow From Continuing Operating Activities"])
capex, _ = _latest_value_from(cashflow_df, ["Capital Expenditure", "Investments In Property Plant And Equipment"])
free_cash_flow = None
if op_cf is not None and capex is not None:
    free_cash_flow = op_cf + capex  # typical: capex is negative, so add

# ROE & ROA (simple)
roe = safe_div(net_income_latest, total_equity)
roa = safe_div(net_income_latest, total_assets)

# Fundamentals table
fund_cols = st.columns(3)
with fund_cols[0]:
    st.markdown("**Income**")
    st.write("Revenue:", fmt_money(rev_latest))
    st.caption("ℹ️ Total sales over the last fiscal year.")
    st.write("Net Income:", fmt_money(net_income_latest))
    st.caption("ℹ️ Profit after all expenses and taxes.")
    st.write("EPS (Diluted):", fmt_num(eps_diluted_latest, 2) if eps_diluted_latest is not None else "N/A")
    st.caption("ℹ️ Earnings per share. Diluted accounts for stock-based comp/options.")

with fund_cols[1]:
    st.markdown("**Margins**")
    st.write("Gross Margin:", fmt_pct(gross_margin) if gross_margin is not None else "N/A")
    st.write("Operating Margin:", fmt_pct(oper_margin) if oper_margin is not None else "N/A")
    st.write("Net Margin:", fmt_pct(net_margin) if net_margin is not None else "N/A")
    st.caption("ℹ️ Higher margins indicate stronger pricing power/efficiency.")

with fund_cols[2]:
    st.markdown("**Balance & Cash Flow**")
    d_e = safe_div(total_debt, total_equity)
    st.write("Debt/Equity:", fmt_num(d_e, 2) if d_e is not None else "N/A")
    st.write("Cash & Equivalents:", fmt_money(cash_latest))
    st.write("Free Cash Flow:", fmt_money(free_cash_flow))
    st.caption("ℹ️ FCF ≈ Operating Cash Flow + Capital Expenditures (capex is usually negative).")

# VALUATION
st.subheader("💲 Valuation", help="How expensive is the stock relative to earnings, cash flow, and book value?")
trailing_eps = fast_info.get("trailingEps")
forward_eps = fast_info.get("forwardEps")
trailing_pe = None
forward_pe = None

if last_price and trailing_eps:
    trailing_pe = safe_div(last_price, trailing_eps)
if fast_info.get("trailingPE") is not None:
    trailing_pe = fast_info.get("trailingPE")

if fast_info.get("forwardPE") is not None:
    forward_pe = fast_info.get("forwardPE")
elif last_price and forward_eps:
    forward_pe = safe_div(last_price, forward_eps)

price_to_book = fast_info.get("priceToBook")
peg_ratio = fast_info.get("pegRatio")
ev = fast_info.get("enterpriseValue")
ev_ebitda = fast_info.get("enterpriseToEbitda")
if ev_ebitda is None and ev is not None:
    ebitda, _ = _latest_value_from(income_df, ["EBITDA"])
    if ebitda and ebitda != 0:
        ev_ebitda = ev / ebitda

val_cols = st.columns(4)
with val_cols[0]:
    st.metric("Trailing P/E", fmt_num(trailing_pe, 2) if trailing_pe is not None else "N/A")
    st.caption("ℹ️ Lower can mean cheaper vs. current earnings.")
with val_cols[1]:
    st.metric("Forward P/E", fmt_num(forward_pe, 2) if forward_pe is not None else "N/A")
    st.caption("ℹ️ Uses analysts’ expected next-year earnings.")
with val_cols[2]:
    st.metric("Price/Book", fmt_num(price_to_book, 2) if price_to_book is not None else "N/A")
    st.caption("ℹ️ Price vs. book (net asset) value.")
with val_cols[3]:
    st.metric("PEG", fmt_num(peg_ratio, 2) if peg_ratio is not None else "N/A")
    st.caption("ℹ️ P/E adjusted for growth (≈1 often considered fair).")

st.caption("ℹ️ EV/EBITDA: " + (fmt_num(ev_ebitda, 2) if ev_ebitda is not None else "N/A") + " — Lower can indicate better value.")

# DIVIDENDS
st.subheader("💸 Dividend", help="Yield, payout, growth history, and safety flags.")

# Dividend yield (prefer fast_info; otherwise from last 4 quarters)
dividend_yield = fast_info.get("dividendYield")
if dividend_yield is None and last_price and not dividends.empty:
    last_365 = dividends[dividends.index >= (pd.Timestamp.today(tz=TZ) - pd.DateOffset(days=365))]
    if not last_365.empty:
        annual_sum = float(last_365.sum())
        dividend_yield = annual_sum / last_price if last_price else None

payout_ratio = fast_info.get("payoutRatio")
years_div_growth = None
if not dividends.empty:
    # Compute annual sums and count consecutive increases ending today
    div_yearly = dividends.groupby(dividends.index.year).sum()
    # Count consecutive increase streak (approximate)
    years_div_growth = 0
    years_sorted = sorted(div_yearly.index.tolist())
    if len(years_sorted) >= 2:
        inc_streak = 0
        for i in range(1, len(years_sorted)):
            if div_yearly.iloc[i] > div_yearly.iloc[i - 1]:
                inc_streak += 1
            else:
                inc_streak = 0
        years_div_growth = inc_streak

div_cols = st.columns(3)
with div_cols[0]:
    st.metric("Dividend Yield", fmt_pct(dividend_yield) if dividend_yield is not None else "N/A")
    st.caption("ℹ️ Annual dividend / price.")
with div_cols[1]:
    st.metric("Payout Ratio", fmt_pct(payout_ratio) if payout_ratio is not None else "N/A")
    st.caption("ℹ️ % of earnings paid as dividends. Lower generally safer.")
with div_cols[2]:
    st.metric("Years of Growth", f"{int(years_div_growth)}" if years_div_growth is not None else "N/A")
    st.caption("ℹ️ Consecutive years with higher total dividends than the year before.")

if not dividends.empty:
    st.line_chart(dividends.rename("Dividends"), height=200)
    if payout_ratio is not None and payout_ratio > 0.8:
        st.warning("⚠️ Payout ratio > 80% — dividends may be less sustainable if earnings fall.", icon="⚠️")
else:
    st.info("No dividend history found.", icon="ℹ️")

# GROWTH
st.subheader("📈 Growth", help="3- and 5-year CAGR for revenue & EPS.")

rev_series = _series_from_row(income_df, ["Total Revenue"])
eps_series = _series_from_row(income_df, ["Diluted EPS", "Basic EPS"])
rev_cagr_3y = cagr(rev_series, 3)
rev_cagr_5y = cagr(rev_series, 5)
eps_cagr_3y = cagr(eps_series, 3)
eps_cagr_5y = cagr(eps_series, 5)

g_cols = st.columns(4)
with g_cols[0]:
    st.metric("Revenue CAGR (3y)", fmt_pct(rev_cagr_3y) if rev_cagr_3y is not None else "N/A")
with g_cols[1]:
    st.metric("Revenue CAGR (5y)", fmt_pct(rev_cagr_5y) if rev_cagr_5y is not None else "N/A")
with g_cols[2]:
    st.metric("EPS CAGR (3y)", fmt_pct(eps_cagr_3y) if eps_cagr_3y is not None else "N/A")
with g_cols[3]:
    st.metric("EPS CAGR (5y)", fmt_pct(eps_cagr_5y) if eps_cagr_5y is not None else "N/A")

if any([(rev_cagr_3y is not None and rev_cagr_3y < 0),
        (rev_cagr_5y is not None and rev_cagr_5y < 0),
        (eps_cagr_3y is not None and eps_cagr_3y < 0),
        (eps_cagr_5y is not None and eps_cagr_5y < 0)]):
    st.warning("⚠️ One or more growth metrics are negative.", icon="⚠️")

# TECHNICALS
st.subheader("🧭 Technicals", help="Moving averages and RSI to gauge trend and momentum (short term).")
ma_50 = ma_200 = rsi_14 = None
price_above_50 = price_above_200 = None
if not hist_1y.empty:
    close = hist_1y["Close"]
    ma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    ma_200 = float(hist_5y["Close"].rolling(200).mean().iloc[-1]) if not hist_5y.empty and len(hist_5y) >= 200 else None
    rsi_14 = rolling_rsi(close, 14)
    if last_price and ma_50:
        price_above_50 = last_price > ma_50
    if last_price and ma_200:
        price_above_200 = last_price > ma_200

t_cols = st.columns(4)
with t_cols[0]:
    st.metric("50D MA", fmt_money(ma_50), help="Average closing price over the last 50 trading days.")
with t_cols[1]:
    st.metric("200D MA", fmt_money(ma_200), help="Average closing price over the last 200 trading days.")
with t_cols[2]:
    st.metric("RSI(14)", fmt_num(rsi_14, 1) if rsi_14 is not None else "N/A",
              help="Momentum indicator: <30 oversold, >70 overbought.")
with t_cols[3]:
    trend_text = []
    if price_above_50 is not None:
        trend_text.append("Above 50D" if price_above_50 else "Below 50D")
    if price_above_200 is not None:
        trend_text.append("Above 200D" if price_above_200 else "Below 200D")
    st.metric("Trend vs MAs", " • ".join(trend_text) if trend_text else "N/A",
              help="Price relative to popular moving averages.")

# NEWS
st.subheader("🗞️ Latest News", help="Recent headlines can affect sentiment and risk.")
max_news = 5
if news_items:
    for item in news_items[:max_news]:
        title = item.get("title") or "Untitled"
        link = item.get("link") or "#"
        pub = item.get("publisher") or "Unknown"
        ts = item.get("time")
        when = ""
        if ts:
            try:
                dt = datetime.fromtimestamp(int(ts), tz=TZ)
                when = f" — {dt.strftime('%Y-%m-%d %H:%M UTC')}"
            except Exception:
                pass
        st.markdown(f"- **[{title}]({link})**  \n  _{pub}{when}_")
else:
    st.info("No news available.", icon="ℹ️")

# RISK FLAGS (and speculation check)
st.subheader("⚠️ Risk Flags", help="Things to watch out for.")
risk_flags = []

# High leverage
d_e = safe_div(total_debt, total_equity)
if d_e is not None and d_e > 2:
    risk_flags.append("High leverage: Debt/Equity > 2")

# Negative FCF
if free_cash_flow is not None and free_cash_flow < 0:
    risk_flags.append("Negative Free Cash Flow")

# Shrinking revenue
if rev_cagr_3y is not None and rev_cagr_3y < 0:
    risk_flags.append("Shrinking revenue (3y CAGR < 0)")

# Dilution > 10% over ~3 years
if not shares_hist.empty and shares_hist.shape[0] >= 2:
    three_years_ago = shares_hist.index.max() - pd.DateOffset(years=3, months=2)
    past = shares_hist[shares_hist.index <= three_years_ago]
    past_val = float(past.iloc[-1]) if not past.empty else float(shares_hist.iloc[0])
    last_val = float(shares_hist.iloc[-1])
    if past_val > 0:
        dilution = (last_val - past_val) / past_val
        if dilution > 0.10:
            risk_flags.append("Large dilution (>10% increase in shares over ~3y)")

# Volatility
beta = fast_info.get("beta")
if beta is not None and beta > 1.3:
    risk_flags.append("High volatility (beta > 1.3)")

# Speculation: today's volume vs 3-month avg
speculation_note = None
avg3m = fast_info.get("averageDailyVolume3Month")
today_vol = None
if not hist_1y.empty:
    today_vol = float(hist_1y["Volume"].iloc[-1])
if avg3m is None and not hist_1y.empty:
    avg3m = float(hist_1y["Volume"].tail(63).mean())  # ~3 months of trading days
if avg3m and today_vol:
    vol_ratio = today_vol / avg3m if avg3m else None
    if vol_ratio and vol_ratio > 2.0:
        risk_flags.append("Speculation alert: volume > 2× 3-month average today")
    speculation_note = f"Today's volume / 3m avg ≈ {fmt_num(vol_ratio, 2) if vol_ratio else 'N/A'}"

if risk_flags:
    for rf in risk_flags:
        st.markdown(f"- ❗ {rf}")
else:
    st.success("No major red flags detected by these simple checks.", icon="✅")

if speculation_note:
    st.caption(f"ℹ️ {speculation_note}")

# ------------------------------ #
# ------- SCORING ENGINE ------- #
# ------------------------------ #

st.subheader("🧮 Scoring & Decision", help="Rule-based 0–1 scores with user-adjustable weights.")

def scale_positive(value: Optional[float], breakpoints: List[Tuple[float, float]]) -> float:
    """
    Piecewise linear scaler.
    breakpoints: list of (x, score) sorted by x ascending.
    Missing values return 0.5 (neutral).
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return 0.5
    if value <= breakpoints[0][0]:
        return breakpoints[0][1]
    if value >= breakpoints[-1][0]:
        return breakpoints[-1][1]
    for i in range(1, len(breakpoints)):
        x0, y0 = breakpoints[i - 1]
        x1, y1 = breakpoints[i]
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + t * (y1 - y0)
    return 0.5

def scale_inverse(value: Optional[float], breakpoints: List[Tuple[float, float]]) -> float:
    """
    For 'lower is better' metrics (e.g., P/E).
    breakpoints define (x, score) but you should pass higher x -> lower score ordering.
    Missing -> 0.5.
    """
    return scale_positive(value, breakpoints)

# Component scores

# Growth: use rev/eps CAGRs
growth_subscores = []
for g in [rev_cagr_3y, rev_cagr_5y, eps_cagr_3y, eps_cagr_5y]:
    s = scale_positive(g, [(-0.10, 0.0), (0.0, 0.4), (0.10, 0.7), (0.20, 0.9), (0.30, 1.0)])
    growth_subscores.append(s)
growth_score = float(np.mean(growth_subscores)) if growth_subscores else 0.5

# Profitability: net margin, ROE, ROA
prof_sub = []
prof_sub.append(scale_positive(net_margin, [(0.0, 0.2), (0.10, 0.6), (0.20, 0.85), (0.30, 1.0)]))
prof_sub.append(scale_positive(roe, [(0.0, 0.2), (0.10, 0.6), (0.20, 0.85), (0.30, 1.0)]))
prof_sub.append(scale_positive(roa, [(0.0, 0.2), (0.05, 0.6), (0.10, 0.85), (0.15, 1.0)]))
profitability_score = float(np.mean(prof_sub))

# Valuation: lower better for P/E, EV/EBITDA, P/B; PEG ~1 is good
val_sub = []
val_sub.append(scale_inverse(trailing_pe, [(10, 1.0), (15, 0.9), (25, 0.7), (35, 0.5), (50, 0.3), (80, 0.1)]))
val_sub.append(scale_inverse(ev_ebitda, [(6, 1.0), (8, 0.9), (10, 0.8), (14, 0.6), (20, 0.4), (30, 0.2)]))
val_sub.append(scale_inverse(price_to_book, [(1, 1.0), (2, 0.9), (3, 0.75), (5, 0.5), (8, 0.3), (15, 0.1)]))
if peg_ratio is None or peg_ratio <= 0:
    peg_score = 0.5
else:
    peg_score = max(0.0, 1.0 - min(abs(peg_ratio - 1.0), 3.0)/3.0)
val_sub.append(peg_score)
valuation_score = float(np.mean(val_sub))

# Dividend Quality: yield, payout (<60% best), growth years
div_sub = []
div_sub.append(scale_positive(dividend_yield, [(0.0, 0.3), (0.01, 0.5), (0.02, 0.7), (0.04, 0.9), (0.06, 1.0)]))
if payout_ratio is None or payout_ratio <= 0:
    div_sub.append(0.6)  # neutral if unknown or zero payout
else:
    div_sub.append(scale_inverse(payout_ratio, [(0.3, 1.0), (0.5, 0.8), (0.6, 0.7), (0.8, 0.4), (1.2, 0.1)]))
div_sub.append(scale_positive(years_div_growth if years_div_growth is not None else None,
                              [(0, 0.3), (3, 0.6), (5, 0.75), (10, 0.9), (20, 1.0)]))
dividend_score = float(np.mean(div_sub))

# Risk: lower leverage, positive FCF, no dilution, stable beta
risk_sub = []
risk_sub.append(scale_inverse(d_e, [(0.3, 1.0), (0.5, 0.9), (1.0, 0.7), (2.0, 0.5), (3.0, 0.3), (5.0, 0.1)]))
risk_sub.append(1.0 if (free_cash_flow is not None and free_cash_flow > 0) else 0.4)
risk_sub.append(scale_inverse(beta, [(0.8, 1.0), (1.0, 0.9), (1.2, 0.7), (1.5, 0.5), (2.0, 0.3), (3.0, 0.1)]))
dilution_penalty = 0.0
if not shares_hist.empty and shares_hist.shape[0] >= 2:
    three_years_ago = shares_hist.index.max() - pd.DateOffset(years=3, months=2)
    past = shares_hist[shares_hist.index <= three_years_ago]
    past_val = float(past.iloc[-1]) if not past.empty else float(shares_hist.iloc[0])
    last_val = float(shares_hist.iloc[-1])
    if past_val > 0:
        dilution_rate = (last_val - past_val) / past_val
        dilution_penalty = min(max(dilution_rate, 0.0), 0.5)  # cap penalty
risk_score = max(0.0, float(np.mean(risk_sub)) - dilution_penalty)

# Weighted overall score
weights = st.session_state["weights"].copy()
w_sum = sum(weights.values()) or 1.0
norm_weights = {k: v / w_sum for k, v in weights.items()}

overall_score = (
    growth_score * norm_weights["Growth"]
    + profitability_score * norm_weights["Profitability"]
    + valuation_score * norm_weights["Valuation"]
    + dividend_score * norm_weights["Dividend Quality"]
    + risk_score * norm_weights["Risk"]
)

# Decision mapping
if overall_score >= 0.75:
    decision = "✅ BUY CANDIDATE (for further research)"
elif overall_score >= 0.55:
    decision = "🟨 WATCHLIST / NEUTRAL"
else:
    decision = "⛔ AVOID FOR NOW"

# Show the scoring breakdown
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.markdown(f"**Decision:** <span style='background-color:{decision_color(decision)}; color:white; padding:6px 10px; border-radius:6px;'>{decision}</span>", unsafe_allow_html=True)
    st.progress(min(max(overall_score, 0.0), 1.0), text=f"Overall Score: {overall_score:.2f} (0–1)")
with c2:
    st.metric("Growth", f"{growth_score:.2f}")
    st.metric("Profitability", f"{profitability_score:.2f}")
    st.metric("Valuation", f"{valuation_score:.2f}")
with c3:
    st.metric("Dividend Quality", f"{dividend_score:.2f}")
    st.metric("Risk", f"{risk_score:.2f}")
    st.caption("Weights: " + ", ".join([f"{k} {v:.0f}" for k, v in weights.items()]))

# Pros/Cons (top drivers)
pros = []
cons = []
# Pros
if rev_cagr_3y and rev_cagr_3y > 0.10: pros.append(f"Revenue growth (3y) {fmt_pct(rev_cagr_3y)}")
if eps_cagr_3y and eps_cagr_3y > 0.10: pros.append(f"EPS growth (3y) {fmt_pct(eps_cagr_3y)}")
if net_margin and net_margin > 0.15: pros.append(f"Healthy net margin {fmt_pct(net_margin)}")
if roe and roe > 0.20: pros.append(f"Strong ROE {fmt_pct(roe)}")
if trailing_pe and trailing_pe < 20: pros.append(f"Reasonable P/E {fmt_num(trailing_pe, 1)}")
if ev_ebitda and ev_ebitda < 10: pros.append(f"EV/EBITDA {fmt_num(ev_ebitda, 1)}")
if dividend_yield and dividend_yield >= 0.02: pros.append(f"Dividend yield {fmt_pct(dividend_yield)}")
if payout_ratio and payout_ratio < 0.6: pros.append("Conservative payout ratio")
if free_cash_flow and free_cash_flow > 0: pros.append(f"Positive FCF {fmt_money(free_cash_flow)}")
if 'price_above_200' in locals() and price_above_200: pros.append("Price above 200-day MA (trend)")

# Cons
if rev_cagr_3y is not None and rev_cagr_3y < 0: cons.append(f"Shrinking revenue (3y) {fmt_pct(rev_cagr_3y)}")
if eps_cagr_3y is not None and eps_cagr_3y < 0: cons.append(f"Shrinking EPS (3y) {fmt_pct(eps_cagr_3y)}")
if net_margin is not None and net_margin < 0.05: cons.append("Thin net margin")
if roe is not None and roe < 0.10: cons.append("Low ROE")
if trailing_pe and trailing_pe > 40: cons.append(f"High P/E {fmt_num(trailing_pe,1)}")
if ev_ebitda and ev_ebitda > 14: cons.append(f"High EV/EBITDA {fmt_num(ev_ebitda,1)}")
if payout_ratio and payout_ratio > 0.8: cons.append("Payout ratio > 80% (potentially unsustainable)")
if free_cash_flow is not None and free_cash_flow < 0: cons.append("Negative FCF")
if d_e and d_e > 2: cons.append("High leverage (D/E > 2)")
if 'price_above_200' in locals() and price_above_200 is not None and price_above_200 is False: cons.append("Price below 200-day MA (weak trend)")
if 'rsi_14' in locals() and rsi_14 and rsi_14 > 70: cons.append("RSI > 70 (overbought)")
if 'rsi_14' in locals() and rsi_14 and rsi_14 < 30: cons.append("RSI < 30 (oversold — can be risky)")

pc1, pc2 = st.columns(2)
with pc1:
    st.markdown("**Top Positives**")
    if pros:
        for p in pros[:6]:
            st.markdown(f"- ✅ {p}")
    else:
        st.write("—")
with pc2:
    st.markdown("**Watch-outs / Negatives**")
    if cons:
        for c in cons[:6]:
            st.markdown(f"- ⚠️ {c}")
    else:
        st.write("—")

# ------------------------------ #
# ------- PRE-BUY CHECKLIST ---- #
# ------------------------------ #

st.subheader("🧾 Pre-Buy Checklist", help="Automatic checks — read each item and confirm for yourself.")

def tri(flag: Optional[bool]) -> str:
    if flag is None:
        return "⚠️"
    return "✅" if flag else "❌"

check_items = []

# Understanding (manual reminder)
check_items.append(("Do you understand how the company makes money?", None))
# Consistent growth
cons_growth = ((rev_cagr_3y or 0) > 0 and (eps_cagr_3y or 0) > 0)
check_items.append(("Consistent revenue & EPS growth (3–5y)", cons_growth))
# Positive FCF
check_items.append(("Positive Free Cash Flow", (free_cash_flow or 0) > 0 if free_cash_flow is not None else None))
# Healthy margins
check_items.append(("Healthy margins (>10% net, >30% gross)", (net_margin or 0) > 0.10 and (gross_margin or 0) > 0.30 if (net_margin is not None and gross_margin is not None) else None))
# Reasonable debt
check_items.append(("Reasonable debt (D/E < 2)", (d_e or 0) < 2 if d_e is not None else None))
# Dividend safety (if applicable)
if dividend_yield and dividend_yield > 0:
    check_items.append(("Dividend payout < 60%", (payout_ratio or 1) < 0.60 if payout_ratio is not None else None))
else:
    check_items.append(("Dividend (if any) appears safe", None))
# Valuation sanity
pe_for_check = trailing_pe or forward_pe
check_items.append(("Valuation not extremely high (P/E < 40 unless growth is huge)", (pe_for_check or 0) < 40 if pe_for_check is not None else None))
# Trend
check_items.append(("Stock trend healthy (price above 200D MA)", price_above_200 if 'price_above_200' in locals() and price_above_200 is not None else None))

for txt, ok in check_items:
    st.markdown(f"- {tri(ok)} {txt}")

# ------------------------------ #
# ---- QUALITATIVE GUIDANCE ---- #
# ------------------------------ #

with st.expander("🧠 Research Guide (What to look for)", expanded=False):
    st.markdown("""
### 1) Company Fundamentals (Business Strength)
- **Revenue & Earnings Growth** → Are sales and profits increasing consistently over years?
- **Profit Margins** → Does the company make money efficiently compared to competitors?
- **Debt Levels** → Too much debt can be risky in downturns (check debt-to-equity ratio).
- **Cash Flow** → Strong free cash flow = ability to reinvest, pay dividends, buy back shares.

### 2) Industry & Competitive Advantage
- **Moat** → Brand, patents, tech, network effects, switching costs?
- **Market Position** → Leader or small competitor?
- **Trends** → Is the industry growing or shrinking long term?

### 3) Valuation (Price vs. Worth)
- **P/E Ratio** → Cheap or expensive vs. earnings?
- **PEG Ratio** → Does growth justify the price?
- **Price-to-Book / Price-to-Sales** → Other value lenses.
- Compare to **industry averages**.

### 4) Dividends (if dividend investor)
- **Dividend Yield** → % income per year.
- **Payout Ratio** → % of earnings paid (lower = safer).
- **Dividend Growth** → Years of increases (e.g., “Aristocrats”).

### 5) Management & Stability
- **Management Quality** → Track record.
- **Insider Ownership** → “Skin in the game.”
- **Culture / Scandals** → Long-term trust matters.

### 6) Risks
- **Volatility** → Does the stock swing wildly?
- **Regulation / Lawsuits** → Legal or political exposure?
- **Competition** → Could new tech disrupt them?
- **Speculation** → Compare today's volume vs. average volume.

Use the checkboxes below as your **own notes** (not used in the score).
    """)
    q1, q2 = st.columns(2)
    with q1:
        st.checkbox("I understand how this business makes money.")
        st.checkbox("Company appears to have a durable moat.")
        st.checkbox("Industry trend seems favorable.")
        st.checkbox("Management quality appears strong.")
    with q2:
        st.checkbox("Insider ownership looks healthy.")
        st.checkbox("No major legal/regulatory red flags found.")
        st.checkbox("Competition/disruption risk seems manageable.")
        st.checkbox("No unusual speculation beyond normal volume.")
    st.text_area("Personal notes", placeholder="Write what you learned about the business here...")

# ------------------------------ #
# --------- EXPORT REPORT ------ #
# ------------------------------ #

st.subheader("📤 Export", help="Download your analysis as a Markdown file.")

def markdown_report() -> str:
    lines = []
    lines.append(f"# {APP_NAME} — {ticker}")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"## Decision\n**{decision}**  \nOverall Score: **{overall_score:.2f}** (0–1)")
    lines.append("")
    lines.append("### Score Breakdown")
    lines.append(f"- Growth: **{growth_score:.2f}** (weight {weights['Growth']:.0f})")
    lines.append(f"- Profitability: **{profitability_score:.2f}** (weight {weights['Profitability']:.0f})")
    lines.append(f"- Valuation: **{valuation_score:.2f}** (weight {weights['Valuation']:.0f})")
    lines.append(f"- Dividend Quality: **{dividend_score:.2f}** (weight {weights['Dividend Quality']:.0f})")
    lines.append(f"- Risk: **{risk_score:.2f}** (weight {weights['Risk']:.0f})")
    lines.append("")
    lines.append("### Snapshot")
    lines.append(f"- Price: {fmt_money(last_price)} ({fmt_pct(daily_change) if daily_change is not None else 'N/A'})")
    if 'wk52_low' in locals() and wk52_low is not None and wk52_high is not None:
        lines.append(f"- 52-week range: {fmt_money(wk52_low)} – {fmt_money(wk52_high)}")
    lines.append(f"- Market Cap: {fmt_money(fast_info.get('market_cap'))}")
    lines.append(f"- Beta: {fmt_num(beta,2) if beta is not None else 'N/A'}")
    lines.append("")
    lines.append("### Fundamentals")
    lines.append(f"- Revenue: {fmt_money(rev_latest)}")
    lines.append(f"- Net Income: {fmt_money(net_income_latest)}")
    lines.append(f"- EPS (Diluted): {fmt_num(eps_diluted_latest,2) if eps_diluted_latest is not None else 'N/A'}")
    lines.append(f"- Gross/Operating/Net Margins: {fmt_pct(gross_margin) if gross_margin is not None else 'N/A'} / {fmt_pct(oper_margin) if oper_margin is not None else 'N/A'} / {fmt_pct(net_margin) if net_margin is not None else 'N/A'}")
    lines.append(f"- Debt/Equity: {fmt_num(d_e,2) if d_e is not None else 'N/A'}")
    lines.append(f"- Cash: {fmt_money(cash_latest)}")
    lines.append(f"- Free Cash Flow: {fmt_money(free_cash_flow)}")
    lines.append(f"- ROE / ROA: {fmt_pct(roe) if roe is not None else 'N/A'} / {fmt_pct(roa) if roa is not None else 'N/A'}")
    lines.append("")
    lines.append("### Valuation")
    lines.append(f"- Trailing P/E: {fmt_num(trailing_pe, 2) if trailing_pe is not None else 'N/A'}")
    lines.append(f"- Forward P/E: {fmt_num(forward_pe, 2) if forward_pe is not None else 'N/A'}")
    lines.append(f"- Price/Book: {fmt_num(price_to_book, 2) if price_to_book is not None else 'N/A'}")
    lines.append(f"- PEG: {fmt_num(peg_ratio, 2) if peg_ratio is not None else 'N/A'}")
    lines.append(f"- EV/EBITDA: {fmt_num(ev_ebitda, 2) if ev_ebitda is not None else 'N/A'}")
    lines.append("")
    lines.append("### Dividend")
    lines.append(f"- Yield: {fmt_pct(dividend_yield) if dividend_yield is not None else 'N/A'}")
    lines.append(f"- Payout Ratio: {fmt_pct(payout_ratio) if payout_ratio is not None else 'N/A'}")
    lines.append(f"- Years of Growth: {years_div_growth if years_div_growth is not None else 'N/A'}")
    lines.append("")
    lines.append("### Growth")
    lines.append(f"- Revenue CAGR (3y/5y): {fmt_pct(rev_cagr_3y) if rev_cagr_3y is not None else 'N/A'} / {fmt_pct(rev_cagr_5y) if rev_cagr_5y is not None else 'N/A'}")
    lines.append(f"- EPS CAGR (3y/5y): {fmt_pct(eps_cagr_3y) if eps_cagr_3y is not None else 'N/A'} / {fmt_pct(eps_cagr_5y) if eps_cagr_5y is not None else 'N/A'}")
    lines.append("")
    lines.append("### Technicals")
    lines.append(f"- 50D / 200D MA: {fmt_money(ma_50)} / {fmt_money(ma_200)}")
    lines.append(f"- RSI(14): {fmt_num(rsi_14,1) if rsi_14 is not None else 'N/A'}")
    lines.append("")
    lines.append("### Risks")
    if risk_flags:
        for rf in risk_flags:
            lines.append(f"- {rf}")
    else:
        lines.append("- None flagged by simple rules.")
    if speculation_note:
        lines.append(f"- {speculation_note}")
    lines.append("")
    lines.append("### Recent News")
    if news_items:
        for item in news_items[:5]:
            title = item.get("title") or "Untitled"
            link = item.get("link") or "#"
            pub = item.get("publisher") or "Unknown"
            ts = item.get("time")
            when = ""
            if ts:
                try:
                    dt = datetime.fromtimestamp(int(ts), tz=TZ)
                    when = f" ({dt.strftime('%Y-%m-%d')})"
                except Exception:
                    pass
            lines.append(f"- [{title}]({link}) — {pub}{when}")
    else:
        lines.append("- No news available.")
    lines.append("")
    lines.append("> **Disclaimer:** Educational only, not financial advice. Always do your own research.")
    return "\n".join(lines)

md = markdown_report()
st.download_button(
    "⬇️ Download Markdown Report",
    data=md.encode("utf-8"),
    file_name=f"{ticker}_analysis_report.md",
    mime="text/markdown",
    help="Saves the current analysis to a Markdown file you can keep."
)

# ------------------------------ #
# ---------- FOOTER ------------ #
# ------------------------------ #

st.markdown("---")
st.caption("**Educational only, not financial advice. Always do your own research.**")
