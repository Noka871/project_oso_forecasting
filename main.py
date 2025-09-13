"""
Главный исполняемый файл для прогнозирования временных рядов ОСО
Использование: python main.py
"""

import warnings

warnings.filterwarnings('ignore')

from data_loader import DataLoader
from model import create_model, plot_model_architecture
from trainer import ModelTrainer
from predictor import Predictor
from utils import create_directories, print_dataset_info
import config


def main():
    """
    Основная функция программы
    """
    print("=" * 60)
    print("ПРОГРАММА ПРОГНОЗИРОВАНИЯ ВРЕМЕННЫХ РЯДОВ ОСО")
    print("=" * 60)

    # Создание необходимых директорий
    create_directories()

    # 1. Загрузка и подготовка данных
    print("\n1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("-" * 40)

    data_loader = DataLoader()

    if not data_loader.load_data():
        print("Ошибка: Не удалось загрузить данные")
        return

    if not data_loader.clean_data():
        print("Ошибка: Не удалось