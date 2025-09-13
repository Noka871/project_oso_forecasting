import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from src.config import Config


def evaluate_model(model, X_test, y_test, y_scaler):
    """
    Оценка модели на тестовых данных и сохранение результатов
    """
    # Предсказание
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Расчет метрик
    mse = mean_squared_error(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)

    # Сохранение метрик
    metrics = {
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    }

    # Сохранение в файл
    os.makedirs("../results", exist_ok=True)
    with open("../results/metrics.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    return metrics, y_test_orig, y_pred