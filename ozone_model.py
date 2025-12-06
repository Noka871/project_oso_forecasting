"""
ozone_model.py
Реализация гибридной нейросетевой модели CNN-LSTM для прогнозирования ОСО
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from datetime import datetime
import os


class OzoneModel:
    """
    Гибридная модель CNN-LSTM для прогнозирования общего содержания озона (ОСО)
    """

    def __init__(self, input_shape=(12, 1), model_path=None):
        """
        Инициализация модели

        Args:
            input_shape: форма входных данных (последовательность, признаки)
            model_path: путь к сохраненной модели (если есть)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None

        # Если указан путь к модели, загружаем ее
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()

    def build_model(self):
        """Построение архитектуры гибридной модели CNN-LSTM"""
        model = keras.Sequential([
            # Входной слой
            layers.Input(shape=self.input_shape),

            # Conv1D слой для выявления локальных паттернов
            layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                          padding='same', name='conv1d_layer'),

            # MaxPooling для уменьшения размерности
            layers.MaxPooling1D(pool_size=2, name='maxpool_layer'),

            # LSTM слой для учета временных зависимостей
            layers.LSTM(128, return_sequences=False, name='lstm_layer'),

            # Полносвязные слои
            layers.Dense(64, activation='relu', name='dense_64'),
            layers.Dropout(0.3, name='dropout_30'),
            layers.Dense(32, activation='relu', name='dense_32'),

            # Выходной слой (1 значение прогноза)
            layers.Dense(1, name='output_layer')
        ])

        # Компиляция модели
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Mean Squared Error
            metrics=['mae', 'mse']  # Mean Absolute Error, Mean Squared Error
        )

        self.model = model
        print("[OzoneModel] Модель успешно построена")
        print(f"[OzoneModel] Архитектура: CNN-LSTM")
        print(f"[OzoneModel] Параметры: {self.model.count_params():,} параметров")

        return model

    def summary(self):
        """Вывод информации о модели"""
        if self.model:
            return self.model.summary()
        else:
            print("[OzoneModel] Модель не построена")

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=50, batch_size=32, validation_split=0.2):
        """
        Обучение модели

        Args:
            X_train: тренировочные данные
            y_train: целевые значения тренировочных данных
            X_val: валидационные данные (опционально)
            y_val: целевые значения валидационных данных (опционально)
            epochs: количество эпох обучения
            batch_size: размер батча
            validation_split: доля данных для валидации (если X_val не указан)

        Returns:
            history: история обучения
        """
        if not self.model:
            self.build_model()

        # Подготовка данных для валидации
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None

        print(f"[OzoneModel] Начало обучения...")
        print(f"[OzoneModel] Эпохи: {epochs}, Батч: {batch_size}")

        # Обучение модели
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            verbose=1
        )

        print(f"[OzoneModel] Обучение завершено")
        print(f"[OzoneModel] Финальный loss: {self.history.history['loss'][-1]:.4f}")
        if 'val_loss' in self.history.history:
            print(f"[OzoneModel] Финальный val_loss: {self.history.history['val_loss'][-1]:.4f}")

        return self.history

    def predict(self, X):
        """
        Прогнозирование

        Args:
            X: входные данные для прогноза

        Returns:
            predictions: массив прогнозов
        """
        if not self.model:
            raise ValueError("[OzoneModel] Модель не обучена. Сначала обучите модель.")

        # Преобразуем входные данные, если нужно
        if len(X.shape) == 1:
            X = X.reshape(1, -1, 1)
        elif len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def evaluate(self, X_test, y_test):
        """
        Оценка модели на тестовых данных

        Args:
            X_test: тестовые данные
            y_test: целевые значения тестовых данных

        Returns:
            metrics: словарь с метриками
        """
        if not self.model:
            raise ValueError("[OzoneModel] Модель не обучена.")

        # Преобразуем данные, если нужно
        if len(X_test.shape) == 2:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Оценка модели
        loss, mae, mse = self.model.evaluate(X_test, y_test, verbose=0)

        metrics = {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse)
        }

        print(f"[OzoneModel] Оценка модели:")
        print(f"  Loss: {loss:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {np.sqrt(mse):.4f}")

        return metrics

    def save_model(self, filepath="trained_models/ozone_model.h5"):
        """
        Сохранение модели

        Args:
            filepath: путь для сохранения модели
        """
        if not self.model:
            raise ValueError("[OzoneModel] Нет модели для сохранения.")

        # Создаем папку, если она не существует
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Сохраняем модель
        self.model.save(filepath)
        print(f"[OzoneModel] Модель сохранена: {filepath}")

        # Также сохраняем историю обучения, если есть
        if self.history:
            history_file = filepath.replace('.h5', '_history.json')
            history_df = pd.DataFrame(self.history.history)
            history_df.to_json(history_file, indent=2)
            print(f"[OzoneModel] История обучения сохранена: {history_file}")

    def load_model(self, filepath):
        """
        Загрузка модели

        Args:
            filepath: путь к файлу модели
        """
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            print(f"[OzoneModel] Модель загружена: {filepath}")
            print(f"[OzoneModel] Архитектура: {self.model.name}")
            print(f"[OzoneModel] Параметры: {self.model.count_params():,}")
        else:
            print(f"[OzoneModel] Файл модели не найден: {filepath}")
            print("[OzoneModel] Будет построена новая модель.")
            self.build_model()


class DataPreprocessor:
    """
    Класс для предобработки данных ОСО
    """

    @staticmethod
    def prepare_sequences(data, sequence_length=12):
        """
        Подготовка последовательностей для обучения

        Args:
            data: массив данных временного ряда
            sequence_length: длина последовательности

        Returns:
            X, y: массивы признаков и целевых значений
        """
        X, y = [], []

        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])

        return np.array(X), np.array(y)

    @staticmethod
    def normalize_data(data):
        """
        Нормализация данных

        Args:
            data: массив данных

        Returns:
            normalized_data: нормализованные данные
            scaler_params: параметры нормализации для обратного преобразования
        """
        data_min = np.min(data)
        data_max = np.max(data)

        # Избегаем деления на ноль
        if data_max == data_min:
            normalized_data = np.zeros_like(data)
        else:
            normalized_data = (data - data_min) / (data_max - data_min)

        scaler_params = {
            'min': data_min,
            'max': data_max
        }

        return normalized_data, scaler_params

    @staticmethod
    def denormalize_data(normalized_data, scaler_params):
        """
        Обратное преобразование нормализованных данных

        Args:
            normalized_data: нормализованные данные
            scaler_params: параметры нормализации

        Returns:
            original_data: данные в исходном масштабе
        """
        data_min = scaler_params['min']
        data_max = scaler_params['max']

        return normalized_data * (data_max - data_min) + data_min


# Функция для создания демонстрационных данных
def create_demo_data(num_samples=1000):
    """
    Создание демонстрационных данных для тестирования модели

    Args:
        num_samples: количество образцов данных

    Returns:
        X, y: демонстрационные данные
    """
    # Генерация временного ряда с трендом и сезонностью
    time = np.arange(num_samples)
    trend = 0.01 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 12)  # Годовая сезонность
    noise = np.random.normal(0, 1, num_samples)

    # Исходные данные ОСО (условные единицы)
    ozone_data = 300 + trend + seasonality + noise

    # Подготовка последовательностей
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_sequences(ozone_data, sequence_length=12)

    # Нормализация
    X_normalized, scaler_params = preprocessor.normalize_data(X)
    y_normalized, _ = preprocessor.normalize_data(y)

    # Reshape для CNN-LSTM (samples, timesteps, features)
    X_reshaped = X_normalized.reshape(X_normalized.shape[0], X_normalized.shape[1], 1)

    return X_reshaped, y_normalized, scaler_params, ozone_data


# Пример использования
if __name__ == "__main__":
    print("=" * 60)
    print("Тестирование модели OzoneModel")
    print("=" * 60)

    # Создание демонстрационных данных
    print("\n1. Создание демонстрационных данных...")
    X, y, scaler_params, original_data = create_demo_data(1000)

    print(f"   Размер X: {X.shape}")
    print(f"   Размер y: {y.shape}")

    # Разделение на train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")

    # Создание и обучение модели
    print("\n2. Создание модели...")
    model = OzoneModel(input_shape=(12, 1))
    model.summary()

    print("\n3. Обучение модели (быстрая демо - 5 эпох)...")
    history = model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=5, batch_size=32
    )

    print("\n4. Оценка модели...")
    metrics = model.evaluate(X_test, y_test)

    print("\n5. Прогнозирование...")
    # Прогноз для последней последовательности
    last_sequence = X_test[-1:].reshape(1, 12, 1)
    prediction_normalized = model.predict(last_sequence)

    # Обратное преобразование
    prediction = DataPreprocessor.denormalize_data(
        prediction_normalized, scaler_params
    )

    print(f"   Прогноз: {prediction[0]:.2f} е.Д.")

    print("\n" + "=" * 60)
    print("Тестирование завершено успешно!")
    print("=" * 60)