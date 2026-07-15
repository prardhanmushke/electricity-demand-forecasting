import pandas as pd

def test_lag_no_leakage():
    """Ensure lag features do not use future values."""
    df = pd.DataFrame({'load': [10, 20, 30, 40, 50]})
    df['lag_1'] = df['load'].shift(1)
    
    assert pd.isna(df.loc[0, 'lag_1']), "Leakage: First row has future data."
    assert df.loc[1, 'lag_1'] == 10, "Leakage: Lag shifted the wrong direction."