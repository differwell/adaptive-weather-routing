import numpy as np
from src.routing.algorithms import haversine

def get_edge_weight(grid, idx_from, idx_to, weather_df, vessel_model):
    """
    Вычисляет время перехода (в секундах) между двумя соседними точками
    на основе данных погоды (усреднённых по всему временному ряду).
    """
    lat1, lon1 = grid.get_coord(idx_from)
    lat2, lon2 = grid.get_coord(idx_to)
    
    # Расстояние между точками (м)
    dist_m = haversine(lat1, lon1, lat2, lon2) * 1000
    
    # Усреднённые погодные условия
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
    
    if speed < 0.1:
        return float('inf')
    return dist_m / speed