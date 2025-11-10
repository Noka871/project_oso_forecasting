import pandas as pd
import numpy as np
import os
from datetime import datetime


class OzoneDataLoader:
    def __init__(self):
        self.data_path = "data/ram/"

    def create_demo_oso_data(self):
        """Создание демонстрационных данных ОСО"""
        # Генерируем реалистичные данные за 1960-2024 гг
        years = range(1960, 2025)
        months = range(1, 13)

        data = []
        for year in years:
            for month in months:
                # Базовое значение ОСО
                base_oso = 300

                # Сезонная компонента (максимум весной, минимум осенью)
                seasonal = 20 * np.sin((month - 3) * 2 * np.pi / 12)

                # Долгосрочный тренд (небольшое снижение)
                trend = -0.1 * (year - 1960)

                # Случайная компонента
                noise = np.random.normal(0, 5)

                # Аномалия в последние годы (согласно исследованиям)
                if year >= 2020 and month in [9, 10, 11, 12]:  # осенне-зимний период
                    anomaly = -8
                else:
                    anomaly = 0

                oso_value = base_oso + seasonal + trend + noise + anomaly

                data.append({
                    'year': year,
                    'month': month,
                    'oso': max(250, min(350, oso_value)),  # Ограничиваем разумные значения
                    'latitude': 56.5,
                    'longitude': 84.95,
                    'temperature': 15 + 20 * np.sin((month - 1) * 2 * np.pi / 12) + np.random.normal(0, 3),
                    'pressure': 1013 + np.random.normal(0, 8)
                })

        return pd.DataFrame(data)

    def create_demo_index_data(self):
        """Создание демонстрационных индексных данных"""
        years = range(1960, 2025)

        data = []
        for year in years:
            # Индексные данные (например, индексы циркуляции)
            for month in range(1, 13):
                data.append({
                    'year': year,
                    'month': month,
                    'oso_index': (month + year % 3) % 12 + 1,
                    'seasonal_factor': 0.8 + 0.3 * np.sin((month - 1) * 2 * np.pi / 12),
                    'trend_component': (year - 1960) * 0.02,
                    'anomaly_flag': 1 if (year >= 2020 and month in [9, 10, 11, 12]) else 0
                })

        return pd.DataFrame(data)

    def load_oso_predict(self, file_path=None):
        """Загрузка файла ОСО_predict.dat"""
        if file_path is None:
            file_path = os.path.join(self.data_path, "ОСО_predict.dat")

        print(f"Загрузка данных из: {file_path}")

        # Если файл существует - загружаем, иначе создаем демо-данные
        if os.path.exists(file_path):
            try:
                # Пробуем разные форматы
                for delimiter in ['\s+', '\t', ',', ';']:
                    try:
                        data = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
                        if len(data.columns) > 1:
                            print(f"Успешно загружено с разделителем: {delimiter}")
                            return data
                    except:
                        continue

                # Если не получилось, пробуем фиксированную ширину
                data = pd.read_fwf(file_path, encoding='utf-8')
                return data

            except Exception as e:
                print(f"Ошибка загрузки файла: {e}")
                return self.create_demo_oso_data()
        else:
            print("Файл не найден, создаются демонстрационные данные")
            return self.create_demo_oso_data()

    def load_oso_index_12(self, file_path=None):
        """Загрузка файла ОСО_индекс_12.dat"""
        if file_path is None:
            file_path = os.path.join(self.data_path, "ОСО_индекс_12.dat")

        print(f"Загрузка индексных данных из: {file_path}")

        if os.path.exists(file_path):
            try:
                for delimiter in ['\s+', '\t', ',', ';']:
                    try:
                        data = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
                        if len(data.columns) > 1:
                            return data
                    except:
                        continue

                data = pd.read_fwf(file_path, encoding='utf-8')
                return data

            except Exception as e:
                print(f"Ошибка загрузки индексного файла: {e}")
                return self.create_demo_index_data()
        else:
            print("Индексный файл не найден, создаются демонстрационные данные")
            return self.create_demo_index_data()

    def analyze_data(self, data):
        """Анализ данных"""
        analysis = {
            'total_records': len(data),
            'columns': list(data.columns),
            'date_range': None,
            'oso_stats': None
        }

        if 'year' in data.columns and 'month' in data.columns:
            years = data['year'].unique()
            analysis['date_range'] = f"{min(years)}-{max(years)}"

        if 'oso' in data.columns:
            analysis['oso_stats'] = {
                'mean': data['oso'].mean(),
                'min': data['oso'].min(),
                'max': data['oso'].max(),
                'std': data['oso'].std()
            }

        return analysis