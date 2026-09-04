import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data.fetchers import Era5Fetcher
import xarray as xr
import tempfile

def diagnose():
    fetcher = Era5Fetcher()
    lat, lon = 55.0, 15.0
    start_date = "2022-01-01"
    end_date = "2022-01-31"
    
    # Формируем запрос как в fetch_reanalysis
    from datetime import datetime
    import pandas as pd
    import numpy as np
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    days = [d.strftime('%d') for d in pd.date_range(start, end, freq='D')]
    times = [f"{h:02d}:00" for h in range(0, 24, 6)]
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
        "grid": [0.25, 0.25],
        "dataset": "reanalysis-era5-single-levels"
    }
    
    # Скачиваем во временный файл
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        temp_filename = tmp.name
    print("Downloading ERA5 file...")
    fetcher.client.retrieve("reanalysis-era5-single-levels", request, temp_filename)
    print("File downloaded:", temp_filename)
    
    # Открываем и исследуем
    ds = xr.open_dataset(temp_filename)
    print("\n=== FULL STRUCTURE ===")
    print("Dimensions:", ds.dims)
    print("Data variables:", list(ds.data_vars))
    print("Coordinates:", list(ds.coords))
    print("\nDetailed coordinates:")
    for coord in ds.coords:
        print(f"  {coord}: {ds[coord].values[:5]} ... (type: {ds[coord].dtype})")
    print("\nAttributes:", ds.attrs)
    
    # Попробуем найти переменную времени
    time_candidates = [c for c in ds.coords if 'time' in c.lower() or 'valid' in c.lower() or 'forecast' in c.lower()]
    print("\nTime candidates:", time_candidates)
    
    # Проверим, есть ли 'time' как измерение
    if 'time' in ds.dims:
        print("Dimension 'time' exists with size:", ds.dims['time'])
    else:
        print("No dimension named 'time'")
    
    os.remove(temp_filename)
    print("Temp file removed.")

if __name__ == "__main__":
    diagnose()