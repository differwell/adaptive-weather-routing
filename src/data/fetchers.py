import requests
import pandas as pd
import xarray as xr
import cdsapi
import os
import numpy as np
import tempfile
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class OpenMeteoFetcher:
    """Загрузка исторических прогнозов из Open-Meteo Archive API"""
    
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    def fetch_forecast(self, lat, lon, start_date, end_date, hourly_vars=None):
        if hourly_vars is None:
            hourly_vars = ["wind_speed_10m", "wind_direction_10m"]
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(hourly_vars),
            "timezone": "GMT",
            "models": "ecmwf_ifs"
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Open-Meteo API error: {response.status_code}, {response.text}")
        data = response.json()
        hourly = data['hourly']
        df = pd.DataFrame({
            'time': pd.to_datetime(hourly['time']),
            'wind_speed': hourly.get('wind_speed_10m', [None]*len(hourly['time'])),
            'wind_dir': hourly.get('wind_direction_10m', [None]*len(hourly['time']))
        })
        df['issue_time'] = df['time'].dt.floor('D')
        df['lead_hours'] = (df['time'] - df['issue_time']).dt.total_seconds() / 3600.0
        return df

    def fetch_actual(self, lat, lon, start_date, end_date):
        """Загружает фактические почасовые данные (референс) из Open-Meteo"""
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "wind_speed_10m,wind_direction_10m",
            "timezone": "GMT"
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Open-Meteo actual API error: {response.status_code}")
        data = response.json()
        hourly = data['hourly']
        df = pd.DataFrame({
            'time': pd.to_datetime(hourly['time']),
            'wind_speed_ref': hourly['wind_speed_10m'],
            'wind_dir_ref': hourly['wind_direction_10m']
        })
        return df


class Era5Fetcher:
    """Загрузка реанализа ERA5 через CDS API (требуется .cdsapirc)"""
    
    def __init__(self):
        self.client = cdsapi.Client()
    
    def fetch_reanalysis(self, lat, lon, start_date, end_date, area=None, grid=0.25):
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        days = [d.strftime('%d') for d in pd.date_range(start, end, freq='D')]
        times = [f"{h:02d}:00" for h in range(0, 24, 6)]
        if area is None:
            area = [lat+1, lon-1, lat-1, lon+1]
        request = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": start.strftime('%Y'),
            "month": start.strftime('%m'),
            "day": days,
            "time": times,
            "area": area,
            "grid": [grid, grid],
            "dataset": "reanalysis-era5-single-levels"
        }
        # Создаём временный файл
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            temp_filename = tmp.name
        # Скачиваем
        self.client.retrieve("reanalysis-era5-single-levels", request, temp_filename)
        if not os.path.exists(temp_filename):
            raise Exception(f"File {temp_filename} was not created")
        # Читаем
        ds = xr.open_dataset(temp_filename)
        try:
            point = ds.sel(latitude=lat, longitude=lon, method='nearest')
            speed = np.sqrt(point['u10']**2 + point['v10']**2)
            speed = speed.squeeze()
            times_arr = point['valid_time'].values
            values = speed.values
            df = pd.DataFrame({
                'time': pd.to_datetime(times_arr),
                'wind_speed_ref': values
            })
        finally:
            ds.close()
        # Удаляем временный файл
        os.remove(temp_filename)
        return df