# src/trainer.py
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import logging

logger = logging.getLogger(__name__)


def train_model(model, X_train, y_train, X_val, y_val,
                model_save_path="../models/best_model.h5",
                epochs=200, batch_size=8, patience=25):
    """
    Обучение модели с callback'ами
    """
    logger.info("Начало обучения модели...")

    # Создаем директорию для моделей если не существует
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Callback'и
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    ]

    # Обучение
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
        shuffle=False  # Для временных рядов не перемешиваем
    )

    logger.info("Обучение завершено!")
    return history


def evaluate_model(model, X_test, y_test):
    """Оценка модели на тестовых данных"""
    logger.info("Оценка модели на тестовых данных...")

    results = model.evaluate(X_test, y_test, verbose=0)
    metrics = dict(zip(model.metrics_names, results))

    logger.info("Результаты оценки:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    return metrics