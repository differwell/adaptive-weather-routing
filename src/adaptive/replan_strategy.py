import numpy as np
from src.routing.algorithms import a_star
from src.utils.edge_weight import get_edge_weight

class ReplanStrategy:
    """
    Управляет маршрутом: хранит текущий путь, проверяет необходимость реплана,
    и при необходимости перестраивает маршрут от текущей позиции.
    """
    def __init__(self, grid, vessel_model, trigger, max_speed_ms=10):
        self.grid = grid
        self.vessel = vessel_model
        self.trigger = trigger
        self.max_speed_ms = max_speed_ms
        self.current_path = None
        self.current_pos_idx = 0
        self.current_time = 0.0
        self.total_time = 0.0
        self.replan_count = 0
    
    def set_initial_route(self, start_idx, goal_idx, weather_df):
        """
        Задаёт начальный маршрут (статический) от старта до финиша.
        """
        def edge_weight(idx_from, idx_to):
            return get_edge_weight(self.grid, idx_from, idx_to, weather_df, self.vessel)
        path, total_time_sec = a_star(self.grid, start_idx, goal_idx, edge_weight, self.max_speed_ms)
        if path is None:
            raise ValueError("Initial route not found")
        self.current_path = path
        self.current_pos_idx = 0
        self.current_time = 0.0
        self.total_time = total_time_sec / 3600.0
        self.replan_count = 0
        return path, self.total_time
    
    def advance(self, time_step_hours, weather_df, goal_idx, month):
        """
        Продвигает судно на time_step_hours по текущему маршруту.
        Если триггер срабатывает – перестраивает маршрут от текущей позиции.
        Возвращает обновлённый путь и текущее время (часы).
        """
        if self.current_path is None:
            return None, 0.0
        
        # Продвигаемся на один шаг (упрощённо – на одну точку)
        # В реальной симуляции нужно вычислять пройденное расстояние, но для демонстрации так.
        if self.current_pos_idx + 1 < len(self.current_path):
            self.current_pos_idx += 1
            self.current_time += time_step_hours
        else:
            # Достигли финиша
            self.total_time = self.current_time
            return self.current_path, self.total_time
        
        # Текущая позиция – индекс точки, в которой мы находимся
        current_idx = self.current_path[self.current_pos_idx]
        
        # Оцениваем lead time (оставшееся время / расстояние) – для простоты возьмём 12 ч
        lead_hours = 12  # можно улучшить, вычислив оставшееся расстояние / среднюю скорость
        
        # Проверяем триггер
        if self.trigger.should_replan(lead_hours, month):
            # Перестраиваем маршрут от текущей позиции до финиша
            def edge_weight(idx_from, idx_to):
                return get_edge_weight(self.grid, idx_from, idx_to, weather_df, self.vessel)
            new_path, total_sec = a_star(self.grid, current_idx, goal_idx, edge_weight, self.max_speed_ms)
            if new_path is not None and len(new_path) > 1:
                self.current_path = new_path
                self.current_pos_idx = 0
                self.replan_count += 1
                print(f"  Реплан #{self.replan_count} в момент {self.current_time:.2f} ч, новый путь из {len(new_path)} точек")
        
        return self.current_path, self.current_time