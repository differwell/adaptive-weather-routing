import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from src.ml.error_predictor import ErrorPredictor

def main():
    # Загружаем метрики за весь год (у вас есть error_metrics_2022.csv)
    df = pd.read_csv("reports/tables/error_metrics_2022.csv")
    # Преобразуем month в формат "2022-01" -> "01" (если нужно)
    df['month'] = df['month'].str[-2:]  # извлекаем номер месяца
    predictor = ErrorPredictor()
    predictor.train(df)
    predictor.save()
    print("Модель обучена и сохранена.")

if __name__ == "__main__":
    main()