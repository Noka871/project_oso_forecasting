"""
Создание архитектур нейронных сетей
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import List, Dict
from ..utils.logger import logger


class NeuralNetworkFactory:
    @staticmethod
    def create_model(model_config, input_shape: tuple) -> Sequential:
        """Создание модели по конфигурации"""
        try:
            logger.info(f"Создание модели {model_config.name}")

            model = Sequential()

            # Добавление слоев
            for i, layer_config in enumerate(model_config.layers):
                layer_type = layer_config['type']

                if layer_type == "LSTM":
                    model.add(LSTM(
                        units=layer_config['units'],
                        return_sequences=layer_config.get('return_sequences', False),
                        input_shape=input_shape if i == 0 else None,
                        name=f"lstm_{i}"
                    ))

                elif layer_type == "Dense":
                    model.add(Dense(
                        units=layer_config['units'],
                        activation=layer_config.get('activation', 'linear'),
                        name=f"dense_{i}"
                    ))

                elif layer_type == "Dropout":
                    model.add(Dropout(
                        rate=layer_config['rate'],
                        name=f"dropout_{i}"
                    ))

            # Компиляция
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=model_config.compile['loss'],
                metrics=model_config.compile['metrics']
            )

            logger.info("Модель успешно создана")
            return model

        except Exception as e:
            logger.error(f"Ошибка создания модели: {e}")
            raise