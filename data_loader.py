"""
Модуль для загрузки и предобработки данных временных рядов ОСО
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config


class DataLoader:
    def __init__(self):
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.data = None

    def load_data(self):
        """
        Загрузка данных из файла
        """
        try:
            # Чтение данных (предполагаем, что файл имеет разделитель табуляции)
            self.data = pd.read_csv(config.DATA_PATH, sep='\t', encoding='utf-8')
            print("Данные успешно загружены")
            print(f"Размерность данных: {self.data.shape}")
            print(f"Колонки: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return False

    def clean_data(self):
        """
        Очистка данных от некорректных значений
        """
        if self.data is None:
            return False

        # Преобразуем все данные к числовому формату, ошибки заменяем на NaN
        for col in config.FEATURE_COLUMNS + [config.TARGET_COLUMN]:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Удаляем строки с пропущенными значениями
        initial_size = len(self.data)
        self.data = self.data.dropna()
        print(f"Удалено строк с ошибками: {initial_size - len(self.data)}")

        return True

    def prepare_features_target(self):
        """
        Подготовка признаков и целевой переменной
        """
        X = self.data[config.FEATURE_COLUMNS].values
        y = self.data[config.TARGET_COLUMN].values.reshape(-1, 1)

        # Масштабирование данных (хотя они уже стандартизированы, но на всякий случай)
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        return X_scaled, y_scaled

    def train_test_split_temporal(self, X, y, test_size=0.2):
        """
        Разделение данных на обучающую и тестовую выборки с учетом временного порядка
        """
        split_index = int(len(X) * (1 - test_size))

        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        print(f"Обучающая выборка: {X_train.shape[0]} samples")
        print(f"Тестовая выборка: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test

    def get_feature_names(self):
        """
        Возвращает названия признаков
        """
        return config.FEATURE_COLUMNS