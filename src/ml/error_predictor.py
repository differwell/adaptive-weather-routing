import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import os

class ErrorPredictor:
    def __init__(self):
        self.model = None
    
    def train(self, df_metrics):
        """
        Обучает XGBoost на исторических метриках ошибок.
        df_metrics должен содержать колонки: month, lead_hours, MAE.
        """
        df = df_metrics.copy()
        # Извлекаем номер месяца из строки (предполагаем формат '2022-01')
        df['month_num'] = df['month'].str[-2:].astype(int)
        # Создаём циклические признаки
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
        
        X = df[['lead_hours', 'month_sin', 'month_cos']]
        y = df['MAE']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.model.fit(X_train, y_train)
        
        print(f"Train R²: {self.model.score(X_train, y_train):.3f}")
        print(f"Test R²: {self.model.score(X_test, y_test):.3f}")
        return self.model
    
    def predict(self, lead_hours, month):
        """Предсказывает MAE для заданных lead_hours и месяца (1-12)."""
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        X = np.array([[lead_hours, month_sin, month_cos]])
        return self.model.predict(X)[0]
    
    def save(self, path="models/error_predictor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
    
    def load(self, path="models/error_predictor.pkl"):
        self.model = joblib.load(path)