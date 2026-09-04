import heapq
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """
    Расстояние между двумя точками на сфере (км).
    """
    R = 6371  # радиус Земли (км)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def a_star(grid, start_idx, goal_idx, get_edge_weight, max_speed_ms=10):
    """
    Алгоритм A* для поиска маршрута с минимальным временем.
    
    Параметры:
        grid – объект RouteGrid
        start_idx, goal_idx – индексы начальной и конечной точек
        get_edge_weight – функция, принимающая (idx_from, idx_to, time) и возвращающая время перехода (в секундах)
        max_speed_ms – максимальная скорость (для эвристики, м/с)
    
    Возвращает:
        (path, total_time) – список индексов и полное время (сек)
    """
    open_set = []
    heapq.heappush(open_set, (0, start_idx))
    
    g_score = {start_idx: 0}
    f_score = {start_idx: heuristic(grid, start_idx, goal_idx, max_speed_ms)}
    came_from = {}
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current == goal_idx:
            # Восстанавливаем путь
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_idx)
            path.reverse()
            return path, g_score[goal_idx]
        
        for neighbor in grid.get_neighbors(current):
            # Вес ребра – время перехода (сек)
            weight = get_edge_weight(current, neighbor)
            tentative_g = g_score[current] + weight
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(grid, neighbor, goal_idx, max_speed_ms)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None, float('inf')

def heuristic(grid, idx, goal_idx, max_speed_ms):
    """
    Эвристика – расстояние по прямой / максимальная скорость (в секундах).
    """
    lat1, lon1 = grid.get_coord(idx)
    lat2, lon2 = grid.get_coord(goal_idx)
    dist_km = haversine(lat1, lon1, lat2, lon2)
    dist_m = dist_km * 1000
    return dist_m / max_speed_ms  # время в секундах