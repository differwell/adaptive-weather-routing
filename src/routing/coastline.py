import geopandas as gpd
import numpy as np

def get_land_mask(lat_min, lat_max, lon_min, lon_max, resolution='10m'):
    """
    Загружает береговую линию и возвращает маску (True – суша, False – море)
    для заданной области.
    """
    # Загружаем упрощённую береговую линию из Natural Earth
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Вырезаем область
    bbox = (lon_min, lat_min, lon_max, lat_max)
    land = world.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    return land

def is_on_land(lat, lon, land_gdf):
    """
    Проверяет, находится ли точка (lat, lon) на суше.
    """
    point = gpd.points_from_xy([lon], [lat])
    # Проверяем, попадает ли точка в полигон суши
    for polygon in land_gdf.geometry:
        if polygon.contains(point[0]):
            return True
    return False