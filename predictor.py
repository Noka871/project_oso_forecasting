"""
Модуль для прогнозирования и визуализации результатов
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import config


class Predictor:
    def __init__(self, data_loader, scaler_y):
        self.data_loader = data_loader
        self.scaler_y = scaler_y
        self.results = None

    def create_prediction_dataframe(self, y_true, y_pred, years):
        """
        Создание DataFrame с результатами прогноза
        """
        # Преобразуем масштабированные значения обратно в исходные
        y_true_original = self.scaler_y.inverse_transform(y_true.reshape(-1, 1))
        y_pred_original = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1))

        results_df = pd.DataFrame({
            'Год': years,
            'Реальное_значение': y_true_original.flatten(),
            'Прогноз': y_pred_original.flatten(),
            'Ошибка': np.abs(y_true_original.flatten() - y_pred_original.flatten())
        })

        return results_df

    def save_predictions(self, results_df, filename=config.OUTPUT_PATH):
        """
        Сохранение результатов прогноза в файл
        """
        results_df.to_csv(filename, sep='\t', index=False, encoding='utf-8')
        print(f"Результаты прогноза сохранены в {filename}")

        return results_df

    def plot_predictions(self, results_df, filename='results/prediction_plot.png'):
        """
        Визуализация результатов прогноза
        """
        plt.style.use(config.PLOT_STYLE)
        plt.figure(figsize=config.FIGURE_SIZE)

        plt.plot(results_df['Год'], results_df['Реальное_значение'],
                 'b-', label='Реальные значения', linewidth=2, marker='o')
        plt.plot(results_df['Год'], results_df['Прогноз'],
                 'r--', label='Прогноз', linewidth=2, marker='s')

        plt.fill_between(results_df['Год'],
                         results_df['Реальное_значение'] - results_df['Ошибка'],
                         results_df['Реальное_значение'] + results_df['Ошибка'],
                         alpha=0.2, color='gray', label='Ошибка прогноза')

        plt.title('Прогнозирование общего содержания озона (ОСО)\nРеальные значения vs Прогноз', fontsize=14)
        plt.xlabel('Год', fontsize=12)
        plt.ylabel('Стандартизированный индекс ОСО', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Добавляем аннотацию с метриками
        mae = results_df['Ошибка'].mean()
        plt.annotate(f'Средняя ошибка: {mae:.3f}',
                     xy=(0.02, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"График прогноза сохранен в {filename}")