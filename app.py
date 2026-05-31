import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Favorita Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

# ── Load model and data ──────────────────────────────────────
@st.cache_resource
def load_model():
    with open("my_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("data_clean.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.reset_index(drop=False)
    df.rename(columns={"index": "Unnamed: 0"}, inplace=True)
    df["onpromotion"] = df["onpromotion"].apply(lambda x: 1 if x else 0)
    return df

model = load_model()
data  = load_data()

MODEL_FEATURES = [
    "Unnamed: 0", "store_nbr", "item_nbr", "id",
    "onpromotion", "year", "month", "day", "day_of_week",
    "unit_sales_7d_avg", "lag_1", "lag_7", "lag_30",
    "rolling_std_7", "is_weekend"
]

# ── Header ───────────────────────────────────────────────────
st.title("📈 Corporación Favorita — Sales Forecasting")
st.markdown(
    "Interactive sales forecast app built with **XGBoost** and **Streamlit**.  "
    "Select a store and product to explore predictions. Hover over the chart to see exact values."
)
st.divider()

# ── Sidebar selectors ────────────────────────────────────────
st.sidebar.header("🔧 Forecast Settings")

store_options = sorted(data["store_nbr"].unique())
store_nbr = st.sidebar.selectbox("Select Store ID", store_options)

data_store = data[data["store_nbr"] == store_nbr]
item_options = sorted(data_store["item_nbr"].unique())
item_nbr = st.sidebar.selectbox("Select Item ID", item_options)

data_filtered = data_store[data_store["item_nbr"] == item_nbr].copy()
max_days = len(data_filtered)

n_days = st.sidebar.slider(
    "Days to forecast",
    min_value=1,
    max_value=max_days,
    value=min(30, max_days)
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Store:** `{store_nbr}`")
st.sidebar.markdown(f"**Item:** `{item_nbr}`")
st.sidebar.markdown(f"**Available days:** `{max_days}`")

# ── What is this app? (info box) ─────────────────────────────
with st.expander("ℹ️ What does this app do? (Click to read)"):
    st.markdown("""
    This app predicts **daily unit sales** for products at Corporación Favorita stores.

    **How it works:** An XGBoost machine learning model was trained on historical
    sales data using time-series features:
    - **Lag features** — sales 1, 7 and 30 days ago
    - **Rolling statistics** — 7-day rolling standard deviation
    - **Calendar features** — month, day, weekend flag
    - **Promotion flag** — whether the product was on promotion

    **How to read the chart:** The teal line shows actual historical sales.
    The yellow dashed line shows what the model predicted. When the two lines
    are close together, the model is predicting accurately.
    """)

# ── KPI row ──────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records",    f"{len(data):,}")
col2.metric("Stores",           data["store_nbr"].nunique())
col3.metric("Products",         data["item_nbr"].nunique())
col4.metric("Date Range",       f"{data['date'].min().date()} → {data['date'].max().date()}")

st.divider()

# ── Forecast ─────────────────────────────────────────────────
data_to_predict = data_filtered.head(n_days).copy()
X = data_to_predict[MODEL_FEATURES]
predictions = model.predict(X)

result_df = pd.DataFrame({
    "Date":            data_to_predict["date"].values,
    "Actual Sales":    data_to_predict["unit_sales"].values,
    "Predicted Sales": predictions.round(2)
})
result_df["Error"] = (result_df["Actual Sales"] - result_df["Predicted Sales"]).round(2)

# ── Interactive Plotly Chart ─────────────────────────────────
st.subheader(f"Forecast — Store {store_nbr} · Item {item_nbr}")

fig = go.Figure()

# Actual sales line
fig.add_trace(go.Scatter(
    x=result_df["Date"],
    y=result_df["Actual Sales"],
    mode="lines",
    name="Actual Sales",
    line=dict(color="#4ECDC4", width=2.5),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>" +
                  "Actual Sales: <b>%{y:.1f}</b><br>" +
                  "<extra></extra>"
))

# Predicted sales line
fig.add_trace(go.Scatter(
    x=result_df["Date"],
    y=result_df["Predicted Sales"],
    mode="lines",
    name="Predicted Sales",
    line=dict(color="#E8C547", width=2.5, dash="dash"),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>" +
                  "Predicted Sales: <b>%{y:.1f}</b><br>" +
                  "<extra></extra>"
))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    hovermode="x unified",
    height=450,
    margin=dict(l=40, r=40, t=40, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1
    ),
    xaxis=dict(title="Date", gridcolor="#1A1A25"),
    yaxis=dict(title="Unit Sales", gridcolor="#1A1A25")
)

st.plotly_chart(fig, use_container_width=True)

# ── Metrics ──────────────────────────────────────────────────
mae  = np.mean(np.abs(result_df["Actual Sales"] - result_df["Predicted Sales"]))
rmse = np.sqrt(np.mean((result_df["Actual Sales"] - result_df["Predicted Sales"])**2))

m1, m2, m3 = st.columns(3)
m1.metric("MAE",  f"{mae:.2f}",  help="Mean Absolute Error — average difference between predicted and actual sales. Lower is better.")
m2.metric("RMSE", f"{rmse:.2f}", help="Root Mean Squared Error — like MAE but penalizes large errors more. Lower is better.")
m3.metric("Days Forecasted", n_days, help="Number of days shown in the forecast above.")

# ── Data table ───────────────────────────────────────────────
st.divider()
st.subheader("Prediction Table")
st.caption("Green/red Error column shows how far each prediction was from the actual value.")
st.dataframe(
    result_df.style.format({
        "Actual Sales":    "{:.1f}",
        "Predicted Sales": "{:.1f}",
        "Error":           "{:.1f}"
    }).background_gradient(subset=["Error"], cmap="RdYlGn"),
    use_container_width=True
)

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown(
    "Built by **Tigist Hayilemariyam** · "
    "XGBoost + Streamlit · "
    "[GitHub](https://github.com/tigist-hayilemariyam)",
    unsafe_allow_html=True
)
