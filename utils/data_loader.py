import pandas as pd
import numpy as np
from datetime import datetime
import os


class DataLoader:

    def load_demo_data(self):
        years = np.arange(1960, 2025)

        base_value = 300
        trend = 0.5 * (years - 1960)
        seasonality = 10 * np.sin(2 * np.pi * (years - 1960) / 11)
        noise = np.random.randn(len(years)) * 5

        oso_values = base_value + trend + seasonality + noise

        data = pd.DataFrame({
            'year': years,
            'month': np.tile(np.arange(1, 13), len(years))[:len(years)],
            'oso': oso_values,
            'region': 'Томская область',
            'latitude': 56.92,
            'longitude': 84.95
        })

        return data

    def analyze_data(self, data):
        if data is None or data.empty:
            return {}

        analysis = {
            'Количество записей': len(data),
            'Период': f"{data['year'].min()}-{data['year'].max()}",
            'Среднее ОСО': f"{data['oso'].mean():.2f} е.Д.",
            'Мин. ОСО': f"{data['oso'].min():.2f} е.Д.",
            'Макс. ОСО': f"{data['oso'].max():.2f} е.Д.",
            'Стандартное отклонение': f"{data['oso'].std():.2f} е.Д.",
            'Тренд': 'Положительный' if data['oso'].iloc[-1] > data['oso'].iloc[0] else 'Отрицательный'
        }

        return analysis

    def save_data(self, data, filename):
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        data.to_csv(filepath, index=False)
        return filepath

    def load_from_file(self, filepath):
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        return None