import numpy as np
from sklearn.preprocessing import StandardScaler
from src.config import Config


def prepare_time_series(data):
    """
    Подготовка временных рядов для обучения LSTM
    Возвращает X (признаки) и y (целевая переменная)
    """
    # Проверка данных на пропуски
    if data.isnull().any().any():
        raise ValueError("Данные содержат пропущенные значения")

    # Извлечение признаков и целевой переменной
    X = data[Config.FEATURES].values
    y = data[Config.TARGET_COLUMN].values

    # Нормализация данных (хотя данные уже стандартизированы, на всякий случай)
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Создание временных окон для LSTM
    X_lstm, y_lstm = [], []
    for i in range(len(X_scaled) - Config.LOOKBACK):
        X_lstm.append(X_scaled[i:(i + Config.LOOKBACK)])
        y_lstm.append(y_scaled[i + Config.LOOKBACK])

    return np.array(X_lstm), np.array(y_lstm), x_scaler, y_scaler