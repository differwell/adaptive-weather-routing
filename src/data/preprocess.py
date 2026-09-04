import pandas as pd
import numpy as np

def align_datasets(forecast_df, reference_df):
    merged = pd.merge(forecast_df, reference_df, on='time', how='inner')
    merged['error'] = merged['wind_speed'] - merged['wind_speed_ref']
    merged['abs_error'] = np.abs(merged['error'])
    return merged

def compute_error_metrics(df, group_by='lead_hours'):
    grouped = df.groupby(group_by).agg(
        MAE=('abs_error', 'mean'),
        RMSE=('error', lambda x: np.sqrt(np.mean(x**2))),
        Bias=('error', 'mean'),
        count=('error', 'count')
    ).reset_index()
    return grouped