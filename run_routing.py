import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.routing.grid import RouteGrid
from src.routing.algorithms import a_star
from src.forecast_error.vessel_model import VesselModel
from src.uncertainty.ensemble import generate_ensemble
from src.utils.edge_weight import get_edge_weight

def main():
    # Загружаем данные погоды
    weather_df = pd.read_csv("data/processed/merged_jan2022.csv")
    print(f"Загружено {len(weather_df)} записей погоды")
    
    # Сетка
    grid = RouteGrid(lat_min=53, lat_max=57, lon_min=13, lon_max=17, step_deg=0.25)
    print(f"Сетка: {grid.n_lat} x {grid.n_lon} = {grid.n_lat*grid.n_lon} точек, после фильтрации суши: {len(grid.points)} точек")
    
    vessel = VesselModel()
    
    start_lat, start_lon = 54.5, 18.5
    goal_lat, goal_lon = 59.3, 18.1
    start_idx = grid.get_index(start_lat, start_lon)
    goal_idx = grid.get_index(goal_lat, goal_lon)
    print(f"Старт: {start_lat}, {start_lon} (индекс {start_idx})")
    print(f"Финиш: {goal_lat}, {goal_lon} (индекс {goal_idx})")
    
    # Базовый маршрут
    def edge_weight(idx_from, idx_to):
        return get_edge_weight(grid, idx_from, idx_to, weather_df, vessel)
    
    path_base, time_base_sec = a_star(grid, start_idx, goal_idx, edge_weight, max_speed_ms=10)
    if path_base is None:
        print("Базовый маршрут не найден")
        return
    time_base_hours = time_base_sec / 3600
    print(f"\n=== Базовый маршрут ===")
    print(f"Количество точек: {len(path_base)}, время: {time_base_hours:.2f} ч")
    coords = [grid.get_coord(idx) for idx in path_base]
    print("Координаты маршрута (lat, lon):")
    for lat, lon in coords:
        print(f"  {lat:.2f}, {lon:.2f}")
    
    # Ансамбль
    print("\nГенерируем ансамбль прогнозов...")
    ensemble = generate_ensemble(weather_df, n_ensemble=30)
    times_hours = []
    successful = 0
    for i, scenario in enumerate(ensemble):
        print(f"  Сценарий {i+1}/{len(ensemble)}", end='\r')
        def edge_weight_scenario(idx_from, idx_to):
            return get_edge_weight(grid, idx_from, idx_to, scenario, vessel)
        path, t_sec = a_star(grid, start_idx, goal_idx, edge_weight_scenario, max_speed_ms=10)
        if path is not None:
            times_hours.append(t_sec / 3600)
            successful += 1
    print(f"\nУспешно построено маршрутов: {successful}/{len(ensemble)}")
    
    if times_hours:
        times = np.array(times_hours)
        print("\n=== Статистика ансамбля ===")
        print(f"Среднее время: {times.mean():.2f} ч")
        print(f"Стандартное отклонение: {times.std():.2f} ч")
        print(f"Минимум: {times.min():.2f} ч, Максимум: {times.max():.2f} ч")
        threshold = 7.0
        risk = (times > threshold).mean() * 100
        print(f"Вероятность превышения {threshold} ч: {risk:.1f}%")
        print(f"Базовое время: {time_base_hours:.2f} ч")
        print(f"Среднее отклонение от базового: {times.mean() - time_base_hours:.2f} ч")

if __name__ == "__main__":
    main()