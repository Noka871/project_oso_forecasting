"""
Вспомогательные функции и утилиты
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def create_directories():
    """
    Создание необходимых директорий
    """
    directories = ['data', 'results', 'models']

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Создана директория: {directory}")


def print_dataset_info(X_train, X_test, y_train, y_test):
    """
    Вывод информации о наборе данных
    """
    print("\n" + "=" * 50)
    print("ИНФОРМАЦИЯ О НАБОРЕ ДАННЫХ")
    print("=" * 50)
    print(f"Обучающая выборка: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Тестовая выборка: {X_test.shape[0]} samples")
    print(f"Период обучения: {1960}-{1960 + X_train.shape[0] - 1}")
    print(f"Период тестирования: {1960 + X_train.shape[0]}-{1960 + X_train.shape[0] + X_test.shape[0] - 1}")
    print("=" * 50)


def calculate_metrics(y_true, y_pred):
    """
    Расчет различных метрик качества
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }