import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import folium
from src.routing.grid import RouteGrid
from src.routing.algorithms import a_star
from src.forecast_error.vessel_model import VesselModel
from src.utils.edge_weight import get_edge_weight

def main():
    # Загружаем погодные данные (январь 2022)
    weather_df = pd.read_csv("data/processed/merged_jan2022.csv")
    print(f"Загружено {len(weather_df)} записей погоды")
    
    # Создаём сетку и модель
    grid = RouteGrid(lat_min=53, lat_max=57, lon_min=13, lon_max=17, step_deg=0.25)
    vessel = VesselModel()
    
    # Старт и финиш
    start_lat, start_lon = 54.5, 18.5  # Гданьск
    goal_lat, goal_lon = 59.3, 18.1    # Стокгольм
    start_idx = grid.get_index(start_lat, start_lon)
    goal_idx = grid.get_index(goal_lat, goal_lon)
    
    # Строим базовый маршрут
    def edge_weight(idx_from, idx_to):
        return get_edge_weight(grid, idx_from, idx_to, weather_df, vessel)
    path, total_time_sec = a_star(grid, start_idx, goal_idx, edge_weight, max_speed_ms=10)
    if path is None:
        print("Маршрут не найден")
        return
    coords = [grid.get_coord(idx) for idx in path]
    total_time_hours = total_time_sec / 3600
    
    # Создаём карту
    m = folium.Map(location=[55.5, 16.0], zoom_start=6, tiles='OpenStreetMap')
    # Добавляем маршрут
    folium.PolyLine(
        locations=coords,
        color='blue',
        weight=4,
        opacity=0.8,
        popup=f"Время: {total_time_hours:.2f} ч"
    ).add_to(m)
    # Старт и финиш
    folium.Marker(
        location=[start_lat, start_lon],
        popup='🚤 Старт (Гданьск)',
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    folium.Marker(
        location=[goal_lat, goal_lon],
        popup='🏁 Финиш (Стокгольм)',
        icon=folium.Icon(color='red', icon='flag', prefix='fa')
    ).add_to(m)
    # Сохраняем
    m.save("route_map.html")
    print(f"✅ Карта сохранена как route_map.html. Откройте её в браузере.")
    print(f"📊 Время маршрута: {total_time_hours:.2f} ч, точек: {len(coords)}")

if __name__ == "__main__":
    main()