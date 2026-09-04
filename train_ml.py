import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from src.ml.error_predictor import ErrorPredictor

def main():
    # Загружаем метрики за весь год
    df = pd.read_csv("reports/tables/error_metrics_2022.csv")
    # Проверим, что колонки есть
    if 'month' not in df.columns or 'lead_hours' not in df.columns or 'MAE' not in df.columns:
        print("Ошибка: в файле error_metrics_2022.csv должны быть колонки month, lead_hours, MAE")
        return
    predictor = ErrorPredictor()
    predictor.train(df)
    predictor.save()
    print("Модель обучена и сохранена в models/error_predictor.pkl")

if __name__ == "__main__":
    main()