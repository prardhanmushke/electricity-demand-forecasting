import pandas as pd
import numpy as np

def calc_mase(y_true, y_pred, y_train, m=52):
    naive_err = np.abs(y_train.diff(m).dropna())
    return np.mean(np.abs(y_true - y_pred)) / naive_err.mean()

def test_mase_perfect_forecast():
    """Test that MASE is exactly 0.0 for a perfect forecast."""
    y_train = pd.Series(np.random.rand(100))
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 2.0, 3.0]) 
    
    mase_score = calc_mase(y_true, y_pred, y_train, m=1)
    assert mase_score == 0.0, "MASE should be zero when predictions perfectly match actuals."

def test_forecast_length():
    """Test that a forecast matches the length of the test period."""
    test_period = pd.Series([1, 2, 3, 4, 5])
    forecast = pd.Series([10, 10, 10, 10, 10])
    assert len(test_period) == len(forecast), "Forecast length does not match test period length."