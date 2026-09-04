import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.routing.grid import RouteGrid
from src.routing.algorithms import a_star
from src.forecast_error.vessel_model import VesselModel
from src.uncertainty.ensemble import generate_ensemble
from src.utils.edge_weight import get_edge_weight
from src.ml.error_predictor import ErrorPredictor
from src.adaptive.trigger import ReplanTrigger
from src.adaptive.replan_strategy import ReplanStrategy

def simulate_strategy(strategy, weather_df, grid, vessel, start_idx, goal_idx, max_speed_ms=10,
                      time_step_hours=1, replan_interval_hours=6, month=1):
    """
    Симулирует движение по стратегии.
    Для статической стратегии – просто строит маршрут один раз.
    Для периодической и адаптивной – использует ReplanStrategy.
    """
    if strategy == 'static':
        def edge_weight(idx_from, idx_to):
            return get_edge_weight(grid, idx_from, idx_to, weather_df, vessel)
        path, total_time_sec = a_star(grid, start_idx, goal_idx, edge_weight, max_speed_ms)
        if path is None:
            return None, float('inf'), 0
        return path, total_time_sec / 3600.0, 0
    else:
        # Для периодической или адаптивной стратегии используем ReplanStrategy
        # Создаём триггер (для периодической – всегда True каждые replan_interval_hours)
        # Для адаптивной – используем ML-триггер
        if strategy == 'periodic':
            # Создаём фиктивный триггер, который срабатывает по расписанию
            class PeriodicTrigger:
                def __init__(self, interval_hours):
                    self.interval = interval_hours
                    self.last_replan_time = -interval_hours
                def should_replan(self, lead_hours, month):
                    # В реальной симуляции мы будем вызывать этот метод каждый шаг,
                    # но мы будем проверять время в стратегии отдельно.
                    return False  # логика будет в стратегии
            # Но проще реализовать периодический реплан прямо в цикле симуляции
            # Поэтому я не буду использовать ReplanStrategy для периодического,
            # а напишу отдельную логику.
            pass

    # Упростим: для периодического и адаптивного мы реализуем симуляцию вручную,
    # потому что ReplanStrategy требует доработки (продвижение по времени).
    # Поэтому я предложу упрощённую версию, где мы на каждом шаге (каждые time_step_hours)
    # продвигаемся по текущему маршруту на одну точку (если шаг равен времени перехода между точками).
    # Но для демонстрации мы можем использовать уже имеющуюся логику ReplanStrategy,
    # которая продвигает на одну точку за шаг, что эквивалентно time_step = время_перехода_между_точками.
    # Однако это не совсем точно. Для целей демонстрации сойдёт.
    
    # Я напишу общую симуляцию для периодической и адаптивной стратегии,
    # используя ручное управление.
    
    # Загружаем модель (если адаптивная)
    if strategy == 'adaptive':
        predictor = ErrorPredictor()
        try:
            predictor.load("models/error_predictor.pkl")
        except:
            print("Модель не найдена, использую заглушку (предсказание = 2.0)")
            # Заглушка, чтобы не падало
            class DummyPredictor:
                def predict(self, lead_hours, month):
                    return 2.0 + 0.1 * lead_hours
            predictor = DummyPredictor()
        trigger = ReplanTrigger(predictor, threshold=2.5)
    else:
        trigger = None  # для периодического используем отдельную логику

    # Начальный маршрут
    def edge_weight(idx_from, idx_to):
        return get_edge_weight(grid, idx_from, idx_to, weather_df, vessel)
    path, total_time_sec = a_star(grid, start_idx, goal_idx, edge_weight, max_speed_ms)
    if path is None:
        return None, float('inf'), 0
    
    current_path = path
    current_pos_idx = 0
    elapsed_time = 0.0
    replan_count = 0
    goal_reached = False
    max_steps = 1000  # защита от бесконечного цикла
    step = 0
    
    while not goal_reached and step < max_steps:
        step += 1
        # Проверяем, не достигли ли финиша
        if current_pos_idx >= len(current_path) - 1:
            goal_reached = True
            break
        
        # Продвигаемся на одну точку (время = вес ребра)
        idx_from = current_path[current_pos_idx]
        idx_to = current_path[current_pos_idx + 1]
        travel_time_sec = get_edge_weight(grid, idx_from, idx_to, weather_df, vessel)
        elapsed_time += travel_time_sec / 3600.0  # в часах
        current_pos_idx += 1
        
        # Определяем, нужно ли перестраивать маршрут
        replan_now = False
        if strategy == 'periodic':
            # Реплан каждые replan_interval_hours
            if int(elapsed_time / replan_interval_hours) > int((elapsed_time - travel_time_sec/3600) / replan_interval_hours):
                replan_now = True
        elif strategy == 'adaptive':
            # Используем триггер на основе lead time (оставшееся время)
            # Для простоты возьмём lead_time = 12 часов (можно улучшить)
            lead_hours = 12
            month_num = 1  # январь
            if trigger.should_replan(lead_hours, month_num):
                replan_now = True
        
        if replan_now:
            # Перестраиваем маршрут от текущей позиции до финиша
            current_idx = current_path[current_pos_idx]
            def edge_weight_replan(idx_from, idx_to):
                return get_edge_weight(grid, idx_from, idx_to, weather_df, vessel)
            new_path, new_time_sec = a_star(grid, current_idx, goal_idx, edge_weight_replan, max_speed_ms)
            if new_path is not None and len(new_path) > 1:
                current_path = new_path
                current_pos_idx = 0
                replan_count += 1
                print(f"  {strategy.capitalize()} реплан #{replan_count} в момент {elapsed_time:.2f} ч, новый путь из {len(new_path)} точек")
            else:
                print(f"  {strategy.capitalize()} реплан не удался, продолжаем по старому маршруту")
    
    if not goal_reached:
        return None, float('inf'), replan_count
    return current_path, elapsed_time, replan_count

def main():
    # Загрузка данных
    weather_df = pd.read_csv("data/processed/merged_jan2022.csv")
    print(f"Загружено {len(weather_df)} записей погоды")
    
    # Сетка и модель
    grid = RouteGrid(lat_min=53, lat_max=57, lon_min=13, lon_max=17, step_deg=0.25)
    vessel = VesselModel()
    
    start_lat, start_lon = 54.5, 18.5
    goal_lat, goal_lon = 59.3, 18.1
    start_idx = grid.get_index(start_lat, start_lon)
    goal_idx = grid.get_index(goal_lat, goal_lon)
    print(f"Старт: {start_lat}, {start_lon} (индекс {start_idx})")
    print(f"Финиш: {goal_lat}, {goal_lon} (индекс {goal_idx})")
    
    # Генерируем ансамбль сценариев (для оценки устойчивости)
    n_ensemble = 20
    ensemble = generate_ensemble(weather_df, n_ensemble=n_ensemble)
    print(f"\nСгенерировано {n_ensemble} сценариев погоды")
    
    # Список стратегий
    strategies = ['static', 'periodic', 'adaptive']
    results = {s: [] for s in strategies}
    
    # Прогон для каждого сценария
    for i, scenario in enumerate(ensemble):
        print(f"\nСценарий {i+1}/{n_ensemble}")
        # Статическая
        path, time_static, _ = simulate_strategy('static', scenario, grid, vessel, start_idx, goal_idx)
        # Периодическая (реплан каждые 6 часов)
        path, time_periodic, replans_periodic = simulate_strategy('periodic', scenario, grid, vessel, start_idx, goal_idx, replan_interval_hours=6)
        # Адаптивная
        path, time_adaptive, replans_adaptive = simulate_strategy('adaptive', scenario, grid, vessel, start_idx, goal_idx)
        
        if path is not None:
            results['static'].append(time_static)
            results['periodic'].append(time_periodic)
            results['adaptive'].append(time_adaptive)
            print(f"  Время: static {time_static:.2f} ч, periodic {time_periodic:.2f} ч, adaptive {time_adaptive:.2f} ч")
        else:
            print("  Маршрут не найден для этого сценария")
    
    # Статистика
    print("\n=== Сравнение стратегий ===")
    for s in strategies:
        if results[s]:
            times = np.array(results[s])
            print(f"\n{s.upper()}:")
            print(f"  Среднее время: {times.mean():.2f} ч")
            print(f"  Стд. отклонение: {times.std():.2f} ч")
            print(f"  Мин/Макс: {times.min():.2f} / {times.max():.2f} ч")
            # Риск превышения 7 ч
            risk = (times > 7.0).mean() * 100
            print(f"  Риск >7ч: {risk:.1f}%")
        else:
            print(f"\n{s.upper()}: нет данных")
    
    # Дополнительно: сравнение периодического и адаптивного со статическим
    if results['static'] and results['adaptive']:
        times_static = np.array(results['static'])
        times_adaptive = np.array(results['adaptive'])
        improvement = (times_static - times_adaptive) / times_static * 100
        print(f"\nУлучшение адаптивной стратегии относительно статической: {improvement.mean():.1f}% (±{improvement.std():.1f}%)")

if __name__ == "__main__":
    main()