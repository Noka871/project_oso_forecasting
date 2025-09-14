# src/data_loader.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Класс для загрузки и подготовки данных ОСО"""

    def __init__(self, data_path=None):
        if data_path is None:
            # Автоматически определяем путь к файлу данных
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, "..")
            self.data_path = os.path.join(base_dir, "data", "raw", "ОСО_индекс_12.dat")
        else:
            self.data_path = data_path

        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler_x, self.scaler_y = StandardScaler(), StandardScaler()

    def load_data(self):
        """Загрузка данных из файла"""
        try:
            logger.info(f"Загрузка данных из {self.data_path}")
            logger.info(f"Абсолютный путь: {os.path.abspath(self.data_path)}")

            if not os.path.exists(self.data_path):
                logger.error(f"Файл не существует: {self.data_path}")
                return False

            # Чтение текстового файла с разделителями-пробелами
            self.data = pd.read_csv(self.data_path, sep='\s+', header=None)

            logger.info(f"Данные загружены. Форма: {self.data.shape}")
            logger.info(f"Первые 3 строки данных:")
            print(self.data.head(3))
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")
            return False

    def prepare_data(self, test_size=0.2, random_state=42):
        """Подготовка и разделение данных"""
        if self.data is None:
            logger.error("Данные не загружены!")
            return False

        try:
            # Извлекаем признаки (месяцы) и целевую переменную
            # Столбцы 0-11: месячные данные (12 месяцев)
            # Столбец 12: среднегодовое значение (целевая переменная)
            X = self.data.iloc[:, 0:12].values
            y = self.data.iloc[:, 12].values.reshape(-1, 1)

            logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

            # Нормализация данных (хотя они уже стандартизированы, но на всякий случай)
            X_scaled = self.scaler_x.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y)

            # Разделение на train/test
            # Для временных рядов shuffle=False чтобы сохранить временной порядок
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y_scaled,
                test_size=test_size,
                random_state=random_state,
                shuffle=False
            )

            logger.info("Данные подготовлены:")
            logger.info(f"  X_train: {self.X_train.shape}")
            logger.info(f"  X_test: {self.X_test.shape}")
            logger.info(f"  y_train: {self.y_train.shape}")
            logger.info(f"  y_test: {self.y_test.shape}")

            return True

        except Exception as e:
            logger.error(f"Ошибка подготовки данных: {e}")
            return False

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def inverse_transform_y(self, y_scaled):
        """Обратное преобразование предсказаний в исходный масштаб"""
        return self.scaler_y.inverse_transform(y_scaled)

    def get_original_data(self):
        return self.data