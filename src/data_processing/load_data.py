import pandas as pd
import os
from pathlib import Path
from src.config import Config


def load_data(file_path=Config.DATA_PATH):
    """
    Усовершенствованная функция загрузки данных с проверками
    """
    try:
        # 1. Проверка существования файла
        if not os.path.exists(file_path):
            available_files = os.listdir(os.path.dirname(file_path))
            raise FileNotFoundError(
                f"Файл {file_path} не найден.\n"
                f"Доступные файлы в директории: {available_files}"
            )

        # 2. Чтение данных с обработкой возможных ошибок
        try:
            data = pd.read_csv(file_path, sep='\s+', engine='python')
        except pd.errors.ParserError:
            # Попробуем другой вариант, если не сработает
            data = pd.read_csv(file_path, sep='\s+', engine='python', encoding='utf-8')

        # 3. Проверка необходимых колонок
        required_columns = [Config.TIME_COLUMN] + Config.FEATURES + [Config.TARGET_COLUMN]
        missing_cols = [col for col in required_columns if col not in data.columns]

        if missing_cols:
            available_cols = data.columns.tolist()
            raise ValueError(
                f"Отсутствуют необходимые колонки: {missing_cols}\n"
                f"Доступные колонки: {available_cols}"
            )

        return data

    except Exception as e:
        # Детализированное сообщение об ошибке
        error_msg = (
            f"Ошибка при загрузке данных из {file_path}:\n"
            f"Тип ошибки: {type(e).__name__}\n"
            f"Сообщение: {str(e)}\n"
            f"Текущая рабочая директория: {os.getcwd()}"
        )
        print(error_msg)
        raise