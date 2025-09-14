# src/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import logging

logger = logging.getLogger(__name__)


def create_model(input_dim=12):
    """
    Создает нейронную сеть для прогнозирования временных рядов
    Архитектура: 12 входов -> 24 -> 12 -> 6 -> 1 выход
    """
    logger.info("Создание модели нейронной сети...")

    model = Sequential()

    # Входной слой
    model.add(Dense(24, input_dim=input_dim, activation='relu',
                    kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Скрытые слои
    model.add(Dense(12, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(6, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Выходной слой (регрессия)
    model.add(Dense(1, activation=None))

    # Компиляция модели
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )

    logger.info("Модель создана и скомпилирована")
    model.summary(print_fn=logger.info)

    return model