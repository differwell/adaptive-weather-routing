class ReplanTrigger:
    """
    Принимает решение о необходимости реплана на основе прогноза ошибки.
    """
    def __init__(self, error_predictor, threshold=2.0):
        """
        error_predictor – экземпляр класса ErrorPredictor (ML-модель)
        threshold – порог ошибки (м/с), при превышении которого запускается реплан
        """
        self.error_predictor = error_predictor
        self.threshold = threshold
    
    def should_replan(self, lead_hours, month):
        """
        Возвращает True, если прогнозируемая ошибка превышает порог.
        """
        predicted_error = self.error_predictor.predict(lead_hours, month)
        return predicted_error > self.threshold