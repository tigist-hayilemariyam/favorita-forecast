# 📈 Corporación Favorita — Sales Forecasting

An interactive sales forecasting web app built with **XGBoost** and **Streamlit**.
The app predicts daily unit sales for stores and products of Corporación Favorita
(a large Ecuadorian retailer) based on historical time-series data.

🔗 **Live App:** _deployed via Streamlit Community Cloud_

---

## What it does

Select a store and a product, choose how many days to forecast, and the app
shows actual vs predicted sales as an interactive chart — together with model
performance metrics (MAE, RMSE) and a detailed prediction table.

## How it works

The model is an **XGBoost Regressor** trained on time-series features:

- **Lag features** — sales 1 day, 7 days and 30 days ago
- **Rolling statistics** — 7-day rolling standard deviation
- **Calendar features** — year, month, day, day of week, weekend flag
- **Promotion flag** — whether the product was on promotion

These features let the model learn daily, weekly and monthly sales patterns.

## Model performance

| Metric | Value |
|--------|-------|
| MAE    | 2.20  |
| RMSE   | 3.99  |

Trained on 3,015 samples, tested on 801 samples (time-based split).

## Tech stack

- **Python** — pandas, numpy
- **XGBoost** — gradient boosting model
- **Streamlit** — interactive web interface
- **Matplotlib** — visualizations

## Project structure

```
favorita-forecast/
├── app.py            # Streamlit application
├── my_model.pkl      # Trained XGBoost model
├── data_clean.csv    # Cleaned time-series dataset
├── requirements.txt  # Python dependencies
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

---

Built by **Tigist Hayilemariyam** — Data Analyst with a background in
Accounting & Finance, specializing in turning data into clear, actionable insights.
