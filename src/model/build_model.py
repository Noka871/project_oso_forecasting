from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from src.config import Config


def build_lstm_model(input_shape):
    """
    Создание модели LSTM для прогнозирования временных рядов
    """
    model = Sequential()

    # Первый LSTM слой с Dropout
    model.add(LSTM(
        units=Config.HIDDEN_LAYERS[0],
        input_shape=input_shape,
        return_sequences=True,
        activation=Config.ACTIVATION
    ))
    model.add(Dropout(Config.DROPOUT_RATE))

    # Второй LSTM слой
    model.add(LSTM(
        units=Config.HIDDEN_LAYERS[1],
        activation=Config.ACTIVATION
    ))
    model.add(Dropout(Config.DROPOUT_RATE))

    # Выходной слой
    model.add(Dense(1))

    # Компиляция модели
    model.compile(
        loss=Config.LOSS,
        optimizer=Config.OPTIMIZER
    )

    return model