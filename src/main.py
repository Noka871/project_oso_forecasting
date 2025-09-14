"""
Главный файл для прогнозирования ОСО
"""

import os
import sys
import numpy as np

# Добавляем папку modules в путь импорта
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from src.utils.config import Config
from data_loader import DataLoader
from model import create_model, get_callbacks
from trainer import ModelTrainer
from predictor import Predictor


def main():
    """Основная функция"""
    print("=" * 70)
    print("🌍 ПРОГНОЗИРОВАНИЕ ОБЩЕГО СОДЕРЖАНИЯ ОЗОНА (ОСО)")
    print("=" * 70)

    # Создаем необходимые директории
    os.makedirs('results/predictions', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    # Инициализация конфигурации
    config = Config()

    # 1. Загрузка и подготовка данных
    print("\n1. 📂 ЗАГРУЗКА ДАННЫХ")
    print("-" * 40)

    data_loader = DataLoader(config)

    if not data_loader.load_data():
        print("❌ Не удалось загрузить данные")
        return

    if not data_loader.clean_data():
        print("❌ Не удалось очистить данные")
        return

    # 2. Подготовка признаков
    print("\n2. 🔧 ПОДГОТОВКА ПРИЗНАКОВ")
    print("-" * 40)

    X, y = data_loader.prepare_features()
    X_train, X_test, y_train, y_test, years_test = data_loader.split_data(X, y)

    # Анализ важности признаков
    feature_importance = data_loader.get_feature_importance()
    print("\n📊 Важность признаков:")
    for feature, importance in feature_importance.items():
        print(f"   {feature}: {importance:.4f}")

    # 3. Создание модели
    print("\n3. 🧠 СОЗДАНИЕ МОДЕЛИ")
    print("-" * 40)

    model = create_model(X_train.shape[1], config)
    model.summary()

    # 4. Обучение модели
    print("\n4. 🎓 ОБУЧЕНИЕ МОДЕЛИ")
    print("-" * 40)

    trainer = ModelTrainer(model, config)
    callbacks = get_callbacks()

    history = trainer.train(X_train, y_train, callbacks)

    # 5. Оценка модели
    print("\n5. 📊 ОЦЕНКА МОДЕЛИ")
    print("-" * 40)

    metrics, predictions = trainer.evaluate(X_test, y_test)

    # 6. Визуализация результатов
    print("\n6. 📈 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("-" * 40)

    # Создаем Predictor
    predictor = Predictor(data_loader, data_loader.scaler_y)

    # Создаем DataFrame с результатами
    results_df = predictor.create_results_dataframe(
        y_test, predictions, years_test, X_test
    )

    # Сохраняем результаты
    predictor.save_predictions(results_df, config.OUTPUT_PATH)

    # Визуализируем результаты
    trainer.plot_training_history('results/plots/training_history.png')
    predictor.plot_predictions(
        results_df,
        'results/plots/predictions.png',
        config
    )
    predictor.plot_feature_importance(
        feature_importance,
        'results/plots/feature_importance.png',
        config
    )

    print("\n" + "=" * 70)
    print("✅ ПРОГРАММА УСПЕШНО ЗАВЕРШЕНА!")
    print("=" * 70)
    print(f"📁 Результаты сохранены в папке 'results/'")
    print(f"📊 Файл с прогнозами: {config.OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()