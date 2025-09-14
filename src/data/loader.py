"""
Загрузка и подготовка данных ОСО
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from ..utils.logger import logger
from ..utils.config import Config


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.data = None
        self.scaler = None

    def load_data(self, file_path: str) -> bool:
        """Загрузка данных из файла"""
        try:
            logger.info(f"Загрузка данных из {file_path}")

            if not Path(file_path).exists():
                logger.error(f"Файл {file_path} не найден")
                return False

            self.data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            logger.info(f"Данные загружены: {self.data.shape}")

            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return False

    def preprocess_data(self) -> bool:
        """Предобработка данных"""
        try:
            logger.info("Начало предобработки данных")

            # Проверка данных
            if self.data is None:
                logger.error("Данные не загружены")
                return False

            # Очистка и преобразование
            numeric_data = self.data.copy()
            for col in self.config.data.features + [self.config.data.target]:
                numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')

            # Удаление пропусков
            initial_size = len(numeric_data)
            numeric_data = numeric_data.dropna()
            removed = initial_size - len(numeric_data)

            if removed > 0:
                logger.warning(f"Удалено строк с ошибками: {removed}")

            self.data = numeric_data
            logger.info("Предобработка данных завершена")
            return True

        except Exception as e:
            logger.error(f"Ошибка предобработки: {e}")
            return False

    def create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Создание последовательностей для LSTM"""
        try:
            X, y = [], []

            for i in range(len(self.data) - self.config.data.sequence_length):
                X.append(self.data[self.config.data.features].iloc[i:i + self.config.data.sequence_length].values)
                y.append(self.data[self.config.data.target].iloc[i + self.config.data.sequence_length])

            return np.array(X), np.array(y)

        except Exception as e:
            logger.error(f"Ошибка создания последовательностей: {e}")
            raise