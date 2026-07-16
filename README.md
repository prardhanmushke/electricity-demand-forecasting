# Forecasting German Electricity Demand

This repository contains a reproducible time-series forecasting pipeline for modelling and forecasting German electricity demand. The project uses hourly electricity load data from Open Power System Data, aggregates it to weekly average load, and compares a range of forecasting models.

## Project Findings & Summary

### Exploratory Data Analysis
The German electricity demand series exhibited strong annual seasonality, with demand peaks consistently occurring during winter months and lower demand during summer periods. This behaviour reflects increased heating and lighting requirements during colder seasons. Aggregating the hourly data to daily and weekly resolutions reduced short-term noise and exposed the underlying seasonal structure more clearly. Stationarity testing using the Augmented Dickey-Fuller (ADF) tests indicated that the weekly demand series was sufficiently stationary for autoregressive modelling.

### Benchmark Models
Four benchmark forecasting models were implemented: Mean, Naive, Seasonal Naive, and Drift. The Seasonal Naive model achieved the best benchmark performance, demonstrating that German electricity demand follows highly repetitive annual cycles.

### Statistical & Feature-Based Modeling
* **SARIMA/SARIMAX:** SARIMA successfully captured annual seasonal behaviour but did not outperform the Seasonal Naive benchmark. Incorporating temperature variables (Heating Degree Days) into a SARIMAX model improved forecasting accuracy, confirming the importance of weather conditions.
* **Feature-Based Regression (GBR):** This model combined Fourier seasonal features, calendar variables, and temperature information. It achieved the best performance among the weekly forecasting models by successfully capturing nonlinear relationships while strictly avoiding data leakage.
* **LSTM:** The LSTM model achieved the lowest numerical RMSE and successfully captured short-term hourly demand dynamics, though it was evaluated on a rolling one-step-ahead basis rather than a fixed two-year horizon.

### Final Recommendation
The feature-based regression model provided the best balance between forecasting accuracy, interpretability, and computational efficiency. For operational weekly forecasting, the feature-based regression model is recommended.

---

## Repository Structure

* `DE_Electricity_Exploration.ipynb`: Original exploration, EDA, and deep learning models.
* `src/`: Core Python modules for data loading, feature engineering, and modeling.
* `scripts/`: Execution scripts to run the pipeline.
* `outputs/`: Automatically generated forecasts, evaluation metrics, and figures.
* `test_benchmarks.py` / `test_features.py`: Pytest suites for logic validation.

## How to Run the Pipeline
To reproduce the forecasts and generate the outputs from a fresh clone:

1. Install dependencies:
```bash
pip install -r requirements.txt

