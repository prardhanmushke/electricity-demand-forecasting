# Forecasting German Electricity Demand



This repository contains an end-to-end machine learning pipeline to forecast weekly electricity demand in Germany using SARIMA and Gradient Boosting Regression (GBR), incorporating weather and calendar features.



## Repository Structure

* `DE\_Electricity\_Exploration.ipynb`: Original exploration, EDA, and deep learning models.

\* `src/`: Core Python modules for data loading, feature engineering, and modeling.

\* `scripts/`: Execution scripts to run the pipeline.

\* `outputs/`: Automatically generated forecasts, evaluation metrics, and figures.

\* `test\_benchmarks.py` / `test\_features.py`: Pytest suites for logic validation.



## How to Run the Pipeline

To reproduce the forecasts and generate the outputs from scratch:



1\. Install dependencies:

&#x20;  ```bash

&#x20;  pip install -r requirements.txt

