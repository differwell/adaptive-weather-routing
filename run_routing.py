import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.routing.grid import RouteGrid
from src.routing.algorithms import a_star, haversine
from src.forecast_error.vessel_model import VesselModel

def get_edge_weight(grid, idx_from, idx_to, weather_df, vessel_model):
    """
    Вычисляет время перехода (в секундах) между двумя соседними точками
    на основе данных погоды (усреднённых по всему временному ряду).
    """
    lat1, lon1 = grid.get_coord(idx_from)
    lat2, lon2 = grid.get_coord(idx_to)
    
    # Расстояние между точками (м)
    dist_m = haversine(lat1, lon1, lat2, lon2) * 1000
    
    # Усреднённые погодные условия (берём средние по всем временам)
    # Предполагаем, что в weather_df есть колонки 'wind_speed' и 'wind_dir'
    wind_speed = weather_df['wind_speed'].mean()
    wind_dir = weather_df['wind_dir'].mean()
    
    # Направление от точки from к точке to (курс в градусах)
    heading = np.degrees(np.arctan2(lon2 - lon1, lat2 - lat1)) % 360
    
    # Скорость яхты (м/с)
    speed = vessel_model.get_speed(
        true_wind_speed=wind_speed,
        true_wind_direction=wind_dir,
        heading=heading,
        current_speed=0,
        current_direction=0,
        wave_height=0
    )
    
    # Время = расстояние / скорость
    if speed < 0.1:
        return float('inf')  # нельзя двигаться
    return dist_m / speed

def main():
    # Загружаем данные погоды (январь 2022)
    weather_df = pd.read_csv("data/processed/merged_jan2022.csv")
    print(f"Загружено {len(weather_df)} записей погоды")
    
    # Создаём сетку для Балтийского моря (53-57N, 13-17E, шаг 0.25°)
    grid = RouteGrid(lat_min=53, lat_max=57, lon_min=13, lon_max=17, step_deg=0.25)
    print(f"Сетка: {grid.n_lat} x {grid.n_lon} = {len(grid.points)} точек")
    
    # Модель яхты
    vessel = VesselModel()
    
    # Задаём старт и финиш (Гданьск -> Стокгольм)
    start_lat, start_lon = 54.5, 18.5   # Гданьск
    goal_lat, goal_lon = 59.3, 18.1     # Стокгольм
    start_idx = grid.get_index(start_lat, start_lon)
    goal_idx = grid.get_index(goal_lat, goal_lon)
    print(f"Старт: {start_lat}, {start_lon} (индекс {start_idx})")
    print(f"Финиш: {goal_lat}, {goal_lon} (индекс {goal_idx})")
    
    # Функция веса (замыкаем weather_df и vessel)
    def edge_weight(idx_from, idx_to):
        return get_edge_weight(grid, idx_from, idx_to, weather_df, vessel)
    
    # Запускаем A*
    path, total_time = a_star(grid, start_idx, goal_idx, edge_weight, max_speed_ms=10)
    
    if path is None:
        print("Маршрут не найден")
    else:
        print(f"Найден маршрут из {len(path)} точек, время: {total_time/3600:.2f} часов")
        coords = [grid.get_coord(idx) for idx in path]
        print("Координаты маршрута (lat, lon):")
        for lat, lon in coords:
            print(f"  {lat:.2f}, {lon:.2f}")

if __name__ == "__main__":
    main()