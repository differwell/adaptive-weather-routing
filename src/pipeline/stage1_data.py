import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetchers import OpenMeteoFetcher, Era5Fetcher
from data.preprocess import align_datasets, compute_error_metrics
import matplotlib.pyplot as plt
import pandas as pd

def run_stage1():
    LAT, LON = 55.0, 15.0
    START = "2022-01-01"
    END = "2022-01-31"
    
    print("=== Stage 1: Data Feasibility ===")
    
    print("Fetching historical forecast from Open-Meteo...")
    om = OpenMeteoFetcher()
    forecast = om.fetch_forecast(LAT, LON, START, END)
    forecast.to_csv("data/raw/forecast_jan2022.csv", index=False)
    print(f"Forecast saved: {len(forecast)} rows")
    
    print("Fetching ERA5 reanalysis...")
    try:
        era = Era5Fetcher()
        reference = era.fetch_reanalysis(LAT, LON, START, END)
        reference.to_csv("data/raw/reference_jan2022.csv", index=False)
        print(f"Reference saved: {len(reference)} rows")
    except Exception as e:
        print(f"ERA5 failed: {e}. Using fallback (forecast as reference).")
        # fallback – копируем прогноз (для демонстрации пайплайна)
        reference = forecast.copy()
        reference = reference.rename(columns={'wind_speed': 'wind_speed_ref'})
        reference = reference[['time', 'wind_speed_ref']]
        reference.to_csv("data/raw/reference_jan2022_fallback.csv", index=False)
    
    print("Merging datasets...")
    merged = align_datasets(forecast, reference)
    merged.to_csv("data/processed/merged_jan2022.csv", index=False)
    print(f"Merged data: {len(merged)} rows")
    
    metrics = compute_error_metrics(merged)
    metrics.to_csv("reports/tables/error_metrics_jan2022.csv", index=False)
    print("Error metrics saved.")
    
    plt.figure(figsize=(10,6))
    plt.plot(metrics['lead_hours'], metrics['MAE'], 'o-', label='MAE (м/с)', color='blue')
    plt.plot(metrics['lead_hours'], metrics['RMSE'], 's-', label='RMSE (м/с)', color='red')
    plt.xlabel('Lead Time (часы)')
    plt.ylabel('Ошибка скорости ветра (м/с)')
    plt.title(f'Ошибка прогноза ветра (ECMWF) vs Lead Time\n{START} – {END}, точка {LAT}N {LON}E')
    plt.grid(True)
    plt.legend()
    plt.savefig("reports/figures/forecast_error_vs_leadtime_jan2022.png", dpi=300)
    plt.show()
    print("=== Stage 1 completed ===")

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports/tables", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    run_stage1()