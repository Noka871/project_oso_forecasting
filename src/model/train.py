from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.config import Config
import os


def train_model(model, X_train, y_train):
    """
    Обучение модели с использованием ранней остановки и сохранения лучших весов
    """
    # Создание директории для моделей, если ее нет
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint(
            Config.MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_split=Config.VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    return history