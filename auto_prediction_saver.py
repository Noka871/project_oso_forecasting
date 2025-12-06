"""
auto_prediction_saver.py
Модуль для автоматического сохранения прогнозов с последовательной нумерацией
"""

import os
import json
import pandas as pd
from datetime import datetime
import glob


class AutoPredictionSaver:
    """
    Автоматически сохраняет каждый новый прогноз с последовательными именами:
    1-й прогноз: ОСО_predict.csv
    2-й прогноз: ОСО_predict1.csv
    3-й прогноз: ОСО_predict2.csv
    и т.д.
    """

    def __init__(self, save_dir="data/predictions"):
        """
        Инициализация савера прогнозов

        Args:
            save_dir: Папка для сохранения прогнозов (по умолчанию: "data/predictions")
        """
        self.save_dir = save_dir
        self.base_name = "ОСО_predict"
        os.makedirs(save_dir, exist_ok=True)

        # Определяем текущий номер для следующего сохранения
        self.current_number = self._get_current_number()

        print(f"[AutoPredictionSaver] Инициализирован. Следующий номер: {self.current_number}")

    def _get_current_number(self):
        """Определяет текущий номер для следующего сохранения"""
        # Ищем все существующие файлы с расширением .csv
        pattern = os.path.join(self.save_dir, f"{self.base_name}*.csv")
        existing_files = glob.glob(pattern)

        if not existing_files:
            print(f"[AutoPredictionSaver] Файлов не найдено. Начинаем с базового файла.")
            return 0  # Нет файлов - начнем с базового (без номера)

        # Проверяем, есть ли файл без номера
        file_no_num = os.path.join(self.save_dir, f"{self.base_name}.csv")
        has_no_number = os.path.exists(file_no_num)

        if not has_no_number:
            print(f"[AutoPredictionSaver] Файла без номера нет. Следующий будет базовым.")
            return 0  # Если нет файла без номера, то следующий будет без номера

        # Если есть файл без номера, ищем максимальный номер
        max_num = 0
        for file in existing_files:
            # Извлекаем номер из имени файла
            filename = os.path.basename(file)

            # Убираем базовое имя и расширение
            num_part = filename[len(self.base_name):-4]  # -4 для ".csv"

            if num_part == "":
                continue  # Пропускаем файл без номера

            if num_part.isdigit():
                num = int(num_part)
                if num > max_num:
                    max_num = num

        next_num = max_num + 1
        print(f"[AutoPredictionSaver] Найден максимальный номер: {max_num}. Следующий: {next_num}")
        return next_num

    def save_prediction(self, predictions, input_data=None, model_info=None, metadata=None):
        """
        Сохраняет прогноз с автоматическим именем

        Args:
            predictions: массив прогнозов (numpy array или список)
            input_data: исходные данные для прогноза (опционально)
            model_info: информация о модели (опционально)
            metadata: дополнительные метаданные (опционально)

        Returns:
            str: путь к сохраненному CSV файлу
        """
        # Определяем имя файла
        if self.current_number == 0:
            filename = f"{self.base_name}.csv"
            display_num = "базовый"
        else:
            filename = f"{self.base_name}{self.current_number}.csv"
            display_num = self.current_number

        filepath = os.path.join(self.save_dir, filename)

        # Подготавливаем данные для сохранения
        save_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Преобразуем predictions в список, если это numpy array
        if hasattr(predictions, 'tolist'):
            predictions_list = predictions.tolist()
        else:
            predictions_list = list(predictions)

        # Создаем DataFrame для сохранения в CSV
        data_to_save = {
            'save_time': [save_time],
            'prediction_count': [len(predictions_list)],
            'predictions': [json.dumps(predictions_list)],
            'prediction_number': [self.current_number if self.current_number > 0 else "base"]
        }

        # Добавляем опциональные данные
        if model_info:
            data_to_save['model_info'] = [json.dumps(model_info)]

        if metadata:
            data_to_save['metadata'] = [json.dumps(metadata)]

        # Сохраняем в CSV
        df = pd.DataFrame(data_to_save)
        df.to_csv(filepath, index=False, encoding='utf-8')

        # Также сохраняем JSON с полными данными
        json_path = filepath.replace('.csv', '.json')
        full_data = {
            'filename': filename,
            'save_time': save_time,
            'prediction_number': self.current_number if self.current_number > 0 else "base",
            'predictions': predictions_list,
            'input_data': input_data.tolist() if input_data is not None and hasattr(input_data,
                                                                                    'tolist') else input_data,
            'model_info': model_info or {},
            'metadata': metadata or {}
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2)

        # Логируем сохранение
        print(f"[AutoPredictionSaver] Прогноз №{display_num} сохранен:")
        print(f"  CSV: {filename}")
        print(f"  JSON: {os.path.basename(json_path)}")
        print(f"  Количество прогнозов: {len(predictions_list)}")

        # Увеличиваем счетчик для следующего сохранения
        self.current_number += 1

        return filepath

    def get_saved_predictions_list(self):
        """Возвращает список всех сохраненных прогнозов"""
        if not os.path.exists(self.save_dir):
            return []

        predictions_list = []

        # Проверяем файл без номера
        base_file = os.path.join(self.save_dir, f"{self.base_name}.csv")
        if os.path.exists(base_file):
            predictions_list.append({
                'filename': f"{self.base_name}.csv",
                'json_file': f"{self.base_name}.json",
                'number': 'base',
                'path': base_file
            })

        # Ищем файлы с номерами
        i = 1
        while True:
            csv_file = os.path.join(self.save_dir, f"{self.base_name}{i}.csv")
            if os.path.exists(csv_file):
                predictions_list.append({
                    'filename': f"{self.base_name}{i}.csv",
                    'json_file': f"{self.base_name}{i}.json",
                    'number': i,
                    'path': csv_file
                })
                i += 1
            else:
                break

        return predictions_list

    def get_next_prediction_info(self):
        """Возвращает информацию о следующем прогнозе"""
        if self.current_number == 0:
            return f"{self.base_name}.csv", "базовый"
        else:
            return f"{self.base_name}{self.current_number}.csv", self.current_number