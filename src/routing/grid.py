import numpy as np
import pandas as pd

class RouteGrid:
    """
    Сетка для маршрутизации на основе географических координат.
    """
    def __init__(self, lat_min, lat_max, lon_min, lon_max, step_deg=0.25):
        """
        lat_min, lat_max, lon_min, lon_max – границы области (в градусах)
        step_deg – шаг сетки (в градусах)
        """
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.step = step_deg
        
        # Создаём массив координат
        self.lats = np.arange(lat_min, lat_max + step_deg, step_deg)
        self.lons = np.arange(lon_min, lon_max + step_deg, step_deg)
        self.n_lat = len(self.lats)
        self.n_lon = len(self.lons)
        
        # Список всех точек (индексы)
        self.points = [(i, j) for i in range(self.n_lat) for j in range(self.n_lon)]
        self.coords = [(self.lats[i], self.lons[j]) for i, j in self.points]
        
        # Словарь индекс -> координаты
        self.idx_to_coord = {idx: (self.lats[i], self.lons[j]) for idx, (i, j) in enumerate(self.points)}
        self.coord_to_idx = {coord: idx for idx, coord in self.idx_to_coord.items()}
    
    def get_neighbors(self, idx, connectivity=8):
        """
        Возвращает список индексов соседних точек (4 или 8 соседей).
        """
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
                    neighbors.append(self.coord_to_idx[(self.lats[ni], self.lons[nj])])
        return neighbors
    
    def get_coord(self, idx):
        return self.idx_to_coord[idx]
    
    def get_index(self, lat, lon):
        # Находит ближайший индекс к заданным координатам
        i = np.argmin(np.abs(self.lats - lat))
        j = np.argmin(np.abs(self.lons - lon))
        return self.coord_to_idx[(self.lats[i], self.lons[j])]