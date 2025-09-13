"""
Модуль для создания архитектуры нейронной сети
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import config


def create_model(input_dim):
    """
    Создание модели нейронной сети для прогнозирования временных рядов

    Args:
        input_dim (int): Количество входных признаков

    Returns:
        model: Скомпилированная модель Keras
    """
    model = Sequential()

    # Входной слой
    model.add(Dense(config.HIDDEN_LAYERS[0],
                    activation=config.ACTIVATION,
                    input_shape=(input_dim,),
                    name='input_layer'))

    # Добавление скрытых слоев
    for i, neurons in enumerate(config.HIDDEN_LAYERS[1:], 1):
        model.add(Dense(neurons,
                        activation=config.ACTIVATION,
                        name=f'hidden_layer_{i}'))
        # Добавляем Dropout для регуляризации
        model.add(Dropout(0.2, name=f'dropout_{i}'))

    # Выходной слой (1 нейрон для регрессии)
    model.add(Dense(1, activation=config.OUTPUT_ACTIVATION, name='output_layer'))

    # Компиляция модели
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=config.LOSS,
        metrics=config.METRICS
    )

    # Вывод информации о модели
    model.summary()

    return model


def plot_model_architecture(model, filename='model_architecture.png'):
    """
    Визуализация архитектуры модели
    """
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=filename,
            show_shapes=True,
            show_layer_names=True,
            dpi=96
        )
        print(f"Архитектура модели сохранена в {filename}")
    except ImportError:
        print("Для визуализации модели установите graphviz и pydot")