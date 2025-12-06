import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from datetime import datetime
import os


class OzoneModel:

    def __init__(self, input_shape=(12, 1), model_path=None):
        self.input_shape = input_shape
        self.model = None
        self.history = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()

    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),
            layers.LSTM(128, return_sequences=False),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model = model

        return model

    def summary(self):
        if self.model:
            return self.model.summary()

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, validation_split=0.2):
        if not self.model:
            self.build_model()

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            verbose=1
        )

        return self.history

    def predict(self, X):
        if not self.model:
            raise ValueError("Модель не обучена")

        if len(X.shape) == 1:
            X = X.reshape(1, -1, 1)
        elif len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def evaluate(self, X_test, y_test):
        if not self.model:
            raise ValueError("Модель не обучена")

        if len(X_test.shape) == 2:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        loss, mae, mse = self.model.evaluate(X_test, y_test, verbose=0)

        metrics = {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse)
        }

        return metrics

    def save_model(self, filepath="trained_models/ozone_model.h5"):
        if not self.model:
            raise ValueError("Нет модели для сохранения")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)

        if self.history:
            history_file = filepath.replace('.h5', '_history.json')
            history_df = pd.DataFrame(self.history.history)
            history_df.to_json(history_file, indent=2)

    def load_model(self, filepath):
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
        else:
            self.build_model()


class DataPreprocessor:

    @staticmethod
    def prepare_sequences(data, sequence_length=12):
        X, y = [], []

        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])

        return np.array(X), np.array(y)

    @staticmethod
    def normalize_data(data):
        data_min = np.min(data)
        data_max = np.max(data)

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
        data_min = scaler_params['min']
        data_max = scaler_params['max']

        return normalized_data * (data_max - data_min) + data_min


def create_demo_data(num_samples=1000):
    time = np.arange(num_samples)
    trend = 0.01 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 12)
    noise = np.random.normal(0, 1, num_samples)

    ozone_data = 300 + trend + seasonality + noise

    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_sequences(ozone_data, sequence_length=12)

    X_normalized, scaler_params = preprocessor.normalize_data(X)
    y_normalized, _ = preprocessor.normalize_data(y)

    X_reshaped = X_normalized.reshape(X_normalized.shape[0], X_normalized.shape[1], 1)

    return X_reshaped, y_normalized, scaler_params, ozone_data