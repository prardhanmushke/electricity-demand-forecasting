import pandas as pd
import numpy as np
import requests
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def load_and_preprocess_data():
    """Downloads hourly data and aggregates to weekly GW."""
    url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
    df = pd.read_csv(url, usecols=["utc_timestamp", "DE_load_actual_entsoe_transparency"], 
                     parse_dates=["utc_timestamp"], index_col="utc_timestamp").sort_index()
    df.index.name = "datetime"
    df.rename(columns={"DE_load_actual_entsoe_transparency": "actual_load_mw"}, inplace=True)
    
    # Filter 2015-2020 and aggregate
    load = df["actual_load_mw"].dropna().loc["2015-01-01":]
    weekly_load = (load.resample("W").mean() / 1000.0).interpolate("time")
    weekly_load.name = "load_gw"
    return pd.DataFrame(weekly_load)

def fetch_weather_features(target_index):
    """Fetches Berlin temperature and calculates heating/cooling degrees."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    res = requests.get(url, params={
        "latitude": 52.52, "longitude": 13.41,
        "start_date": str(target_index.min().date()), "end_date": str(target_index.max().date()),
        "daily": "temperature_2m_mean", "timezone": "Europe/Berlin"
    })
    data = res.json()["daily"]
    weather = pd.DataFrame({"temp_mean": data["temperature_2m_mean"]}, index=pd.to_datetime(data["time"]))
    
    weekly_weather = weather.resample("W").mean()
    weekly_weather.index = weekly_weather.index.tz_localize(None)
    weekly_weather["heat_deg"] = np.clip(15.5 - weekly_weather["temp_mean"], 0, None)
    weekly_weather["cool_deg"] = np.clip(weekly_weather["temp_mean"] - 22.0, 0, None)
    return weekly_weather

def calc_mase(y_true, y_pred, y_train, m=52):
    naive_err = np.abs(y_train.diff(m).dropna())
    return np.mean(np.abs(y_true - y_pred)) / naive_err.mean()

def score_model(name, y_true, y_pred, y_train):
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MASE": calc_mase(y_true, y_pred, y_train),
        "Bias": np.mean(y_pred - y_true)
    }

def get_benchmarks(train, test, horizon):
    preds = pd.DataFrame(index=test.index)
    preds["mean"] = train.mean()
    preds["naive"] = train.iloc[-1]
    
    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    preds["drift"] = train.iloc[-1] + slope * np.arange(1, horizon + 1)
    
    snaive = pd.concat([train.iloc[-52:]]*2).values[:horizon]
    preds["seasonal_naive"] = snaive
    return preds

def train_sarima(train, test, horizon):
    model = SARIMAX(train, order=(0, 0, 6), seasonal_order=(1, 1, 1, 52), trend='c')
    fit = model.fit(disp=False)
    return fit.get_forecast(steps=horizon).predicted_mean

def train_gbr(train_df, test_df, target_col="load_gw"):
    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)