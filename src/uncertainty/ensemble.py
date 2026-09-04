import numpy as np
import pandas as pd

def generate_ensemble(weather_df, n_ensemble=30, lead_time_col='lead_hours',
                      wind_col='wind_speed', wind_dir_col='wind_dir'):
    """
    Генерирует ансамбль прогнозов, добавляя шум к скорости и направлению ветра.
    Дисперсия шума растёт с lead time.
    Возвращает список DataFrame (каждый – один сценарий).
    """
    ensembles = []
    for i in range(n_ensemble):
        df = weather_df.copy()
        # Шум для скорости: дисперсия = 0.5 + 0.03 * lead_hours (м/с)^2
        noise_scale = np.sqrt(0.5 + 0.03 * df[lead_time_col].values)
        wind_noise = np.random.normal(0, noise_scale)
        df[wind_col] = df[wind_col] + wind_noise
        
        # Шум для направления: дисперсия = 5 + 0.5 * lead_hours (градусы)^2
        dir_noise_scale = np.sqrt(5 + 0.5 * df[lead_time_col].values)
        dir_noise = np.random.normal(0, dir_noise_scale)
        df[wind_dir_col] = (df[wind_dir_col] + dir_noise) % 360
        
        ensembles.append(df)
    return ensembles