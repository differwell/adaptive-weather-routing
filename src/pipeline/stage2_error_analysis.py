import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data.fetchers import OpenMeteoFetcher, Era5Fetcher
from data.preprocess import align_datasets, compute_error_metrics

def run_stage2():
    LAT, LON = 55.0, 15.0
    months = [f"2022-{m:02d}-01" for m in range(1,13)]
    all_metrics = []
    
    for start_date in months:
        end_date = pd.to_datetime(start_date) + pd.offsets.MonthEnd(1)
        end_date = end_date.strftime("%Y-%m-%d")
        print(f"Processing {start_date} to {end_date}")
        
        om = OpenMeteoFetcher()
        forecast = om.fetch_forecast(LAT, LON, start_date, end_date)
        
        # Пытаемся использовать ERA5
        use_era5 = True
        try:
            era = Era5Fetcher()
            reference = era.fetch_reanalysis(LAT, LON, start_date, end_date)
            print("  ERA5 loaded successfully")
        except Exception as e:
            print(f"  ERA5 failed: {e}")
            use_era5 = False
        
        if not use_era5:
            print("  Using Open-Meteo actual weather as reference")
            reference = om.fetch_actual(LAT, LON, start_date, end_date)
            reference = reference.rename(columns={'wind_speed': 'wind_speed_ref'})
            reference = reference[['time', 'wind_speed_ref']]
        
        merged = align_datasets(forecast, reference)
        metrics = compute_error_metrics(merged)
        metrics['month'] = start_date[:7]
        all_metrics.append(metrics)
        # Сохраняем промежуточный результат
        metrics.to_csv(f"reports/tables/error_metrics_{start_date[:7]}.csv", index=False)
    
    # Объединяем все месяцы
    df_all = pd.concat(all_metrics, ignore_index=True)
    df_all.to_csv("reports/tables/error_metrics_2022.csv", index=False)
    
    # График сезонной изменчивости
    plt.figure(figsize=(14,8))
    for month in df_all['month'].unique():
        sub = df_all[df_all['month'] == month]
        plt.plot(sub['lead_hours'], sub['MAE'], label=month, alpha=0.7)
    plt.xlabel('Lead Time (часы)')
    plt.ylabel('MAE (м/с)')
    plt.title('Сезонная изменчивость ошибки прогноза ветра (2022)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/figures/seasonal_mae_2022.png", dpi=300)
    plt.show()
    
    # Boxplot ошибок для разных lead time
    plt.figure(figsize=(10,6))
    df_all.boxplot(column='MAE', by='lead_hours', grid=False)
    plt.title('Распределение MAE по lead time')
    plt.suptitle('')
    plt.xlabel('Lead Time (часы)')
    plt.ylabel('MAE (м/с)')
    plt.savefig("reports/figures/boxplot_mae_by_lead.png", dpi=300)
    plt.show()
    
    print("Stage 2 completed. Results saved in reports/")

if __name__ == "__main__":
    os.makedirs("reports/tables", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    run_stage2()