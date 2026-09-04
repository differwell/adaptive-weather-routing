import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
import requests

class RouteGrid:
    """
    Сетка для маршрутизации с учётом береговой линии.
    Точки, попадающие на сушу, исключаются.
    """
    def __init__(self, lat_min, lat_max, lon_min, lon_max, step_deg=0.25):
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.step = step_deg
        
        self.lats = np.arange(lat_min, lat_max + step_deg, step_deg)
        self.lons = np.arange(lon_min, lon_max + step_deg, step_deg)
        self.n_lat = len(self.lats)
        self.n_lon = len(self.lons)
        
        # Загружаем береговую линию
        self.land_gdf = self._load_land_mask()
        
        # Формируем список только морских точек
        self.points = []
        for i in range(self.n_lat):
            for j in range(self.n_lon):
                lat = self.lats[i]
                lon = self.lons[j]
                if not self._is_on_land(lat, lon):
                    self.points.append((i, j))
        
        self.coords = [(self.lats[i], self.lons[j]) for i, j in self.points]
        self.idx_to_coord = {idx: (self.lats[i], self.lons[j]) for idx, (i, j) in enumerate(self.points)}
        self.coord_to_idx = {(self.lats[i], self.lons[j]): idx for idx, (i, j) in enumerate(self.points)}
        
        print(f"Сетка: {self.n_lat}x{self.n_lon} = {self.n_lat*self.n_lon} точек, после фильтрации суши: {len(self.points)} точек")
    
    def _load_land_mask(self):
        """
        Загружает береговую линию. Если локального файла нет – скачивает.
        """
        # Путь к локальному шейп-файлу
        shp_path = os.path.join("data", "external", "ne_110m_land.shp")
        os.makedirs(os.path.dirname(shp_path), exist_ok=True)
        
        # Если файла нет – скачиваем
        if not os.path.exists(shp_path):
            print("Скачиваю береговую линию (Natural Earth 110m)...")
            url = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
            zip_path = shp_path.replace(".shp", ".zip")
            try:
                r = requests.get(url, stream=True)
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                # Распаковываем
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(shp_path))
                os.remove(zip_path)
                print("Береговая линия загружена.")
            except Exception as e:
                print(f"Не удалось скачать береговую линию: {e}")
                return gpd.GeoDataFrame()
        
        # Загружаем
        try:
            land = gpd.read_file(shp_path)
            # Обрезаем по области
            bbox = (self.lon_min, self.lat_min, self.lon_max, self.lat_max)
            land = land.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            return land
        except Exception as e:
            print(f"Ошибка загрузки береговой линии: {e}")
            return gpd.GeoDataFrame()
    
    def _is_on_land(self, lat, lon):
        if self.land_gdf.empty:
            return False
        point = Point(lon, lat)
        for polygon in self.land_gdf.geometry:
            if polygon.contains(point):
                return True
        return False
    
    def get_neighbors(self, idx, connectivity=8):
        i, j = self.points[idx]
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                if connectivity == 4 and abs(di) + abs(dj) > 1:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.n_lat and 0 <= nj < self.n_lon:
                    coord = (self.lats[ni], self.lons[nj])
                    if coord in self.coord_to_idx:
                        neighbors.append(self.coord_to_idx[coord])
        return neighbors
    
    def get_coord(self, idx):
        return self.idx_to_coord[idx]
    
    def get_index(self, lat, lon):
        i = np.argmin(np.abs(self.lats - lat))
        j = np.argmin(np.abs(self.lons - lon))
        coord = (self.lats[i], self.lons[j])
        if coord in self.coord_to_idx:
            return self.coord_to_idx[coord]
        else:
            # ищем ближайшую морскую
            min_dist = float('inf')
            best_idx = None
            for idx, (lat0, lon0) in self.idx_to_coord.items():
                dist = (lat - lat0)**2 + (lon - lon0)**2
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            return best_idx