import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.electricity_demand import pipeline

print("1. Loading and preprocessing data...")
df = pipeline.load_and_preprocess_data()
df.index = df.index.tz_localize(None)

print("2. Fetching weather features...")
weather_df = pipeline.fetch_weather_features(df.index)
master_df = df.join(weather_df).dropna()

print("3. Feature Engineering...")
master_df["week_num"] = master_df.index.isocalendar().week.astype(int)
master_df["fourier_sin_1"] = np.sin(2 * np.pi * 1 * master_df["week_num"] / 52)
master_df["fourier_cos_1"] = np.cos(2 * np.pi * 1 * master_df["week_num"] / 52)

horizon = 104
train_df, test_df = master_df.iloc[:-horizon], master_df.iloc[-horizon:]
train_target, test_target = train_df["load_gw"], test_df["load_gw"]

print("4. Running Models...")
forecasts = pd.DataFrame({"actual": test_target})
benchmarks = pipeline.get_benchmarks(train_target, test_target, horizon)
for col in benchmarks.columns:
    forecasts[col] = benchmarks[col]

print("   -> Training SARIMA...")
forecasts["sarima"] = pipeline.train_sarima(train_target, test_target, horizon).values

print("   -> Training Feature Model (GBR)...")
forecasts["feature_model"] = pipeline.train_gbr(train_df, test_df)

print("5. Saving Outputs...")
import os
from statsmodels.graphics.tsaplots import plot_acf

os.makedirs("outputs/forecasts", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# Save Forecasts & Metrics
forecasts.to_csv("outputs/forecasts/all_forecasts.csv")
metrics = [pipeline.score_model(m, forecasts["actual"], forecasts[m], train_target) for m in forecasts.columns.drop("actual")]
pd.DataFrame(metrics).round(3).to_csv("outputs/metrics/model_comparison.csv", index=False)

print("6. Generating Your 4 Specific Figures...")

# Figure 1: German Electricity Demand (Jan 2015 to Oct 2020)
plt.figure(figsize=(14, 5))
plt.plot(master_df.index, master_df["load_gw"], color="#1f77b4", linewidth=1)
plt.title("German Electricity Demand (Jan 2015 - Oct 2020)")
plt.ylabel("Load (GW)")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig("outputs/figures/german_electricity_demand.png")

# Figure 2: Forecast Overlay
plt.figure(figsize=(12, 6))
plt.plot(test_target.index, test_target, label="Actual Load", color="black", linewidth=2)
plt.plot(test_target.index, forecasts["sarima"], label="SARIMA Forecast", linestyle="--")
plt.plot(test_target.index, forecasts["feature_model"], label="GBR Forecast", linestyle="-.")
plt.legend()
plt.title("Test Set Forecast Comparison")
plt.tight_layout()
plt.savefig("outputs/figures/forecast_comparison.png")

# Figure 3: Distribution of SARIMA Residuals
sarima_residuals = test_target - forecasts["sarima"]
plt.figure(figsize=(8, 5))
plt.hist(sarima_residuals.dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of SARIMA Residuals")
plt.xlabel("Error (GW)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/figures/sarima_residual_distribution.png")

# Figure 4: Residual ACF Plot
fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(sarima_residuals.dropna(), ax=ax, lags=40, title="Autocorrelation of SARIMA Residuals")
plt.tight_layout()
plt.savefig("outputs/figures/residual_acf.png")

print("Pipeline complete! All 4 custom figures saved to outputs/figures/")