"""
SmartPlant AI - Model Training and Feature Engineering Reference
----------------------------------------------------------------
Provides lag feature engineering and baseline model training routines.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def engineer_lag_features(df):
    """
    Constructs historical lag features without lookahead leakage.
    Since 15-minute intervals are used, 1 day corresponds to 96 intervals (24 * 4).
    """
    df = df.copy()

    # Chronological sort if needed
    try:
        dt_series = pd.to_datetime(df['date'], dayfirst=True, format='mixed')
        if not dt_series.is_monotonic_increasing:
            if 'NSM' in df.columns and (df['NSM'] != 0).any():
                date_part = pd.to_datetime(df['date'].astype(str).str.split(' ').str[0], dayfirst=True)
                scada_dt = date_part + pd.to_timedelta(np.where(df['NSM'] == 0, 86400, df['NSM']), unit='s')
                df = df.iloc[scada_dt.argsort()].reset_index(drop=True)
            else:
                df = df.iloc[dt_series.argsort()].reset_index(drop=True)
    except Exception:
        pass

    # Historical lags on Usage_kWh
    df['Usage_Lag_1'] = df['Usage_kWh'].shift(1)
    df['Usage_Lag_2'] = df['Usage_kWh'].shift(2)
    df['Usage_Lag_4'] = df['Usage_kWh'].shift(4)
    df['Usage_Lag_96'] = df['Usage_kWh'].shift(96)  # Exactly 24 hours ago

    # Rolling window statistics strictly on past intervals (shifted by 1)
    df['Usage_Roll_Mean_4'] = df['Usage_kWh'].shift(1).rolling(window=4).mean()
    df['Usage_Roll_Std_4'] = df['Usage_kWh'].shift(1).rolling(window=4).std()

    # Backfill warm-up period at start of series
    df['Usage_Lag_1'] = df['Usage_Lag_1'].bfill()
    df['Usage_Lag_2'] = df['Usage_Lag_2'].bfill()
    df['Usage_Lag_4'] = df['Usage_Lag_4'].bfill()
    df['Usage_Lag_96'] = df['Usage_Lag_96'].bfill()
    df['Usage_Roll_Mean_4'] = df['Usage_Roll_Mean_4'].bfill()
    df['Usage_Roll_Std_4'] = df['Usage_Roll_Std_4'].bfill().fillna(0.0)

    return df
