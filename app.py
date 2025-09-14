# app.py
from flask import Flask, render_template, jsonify, request
import os
import sys
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
import base64

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataLoader
from src.model import create_model
from src.trainer import train_model, evaluate_model
from src.predictor import make_predictions, load_trained_model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'oso_forecasting_secret'

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OSOForecastingApp:
    def __init__(self):
        self.data_loader = None
        self.model = None
        self.results = None
        self.history = None

    def load_and_prepare_data(self):
        """Загрузка и подготовка данных"""
        try:
            self.data_loader = DataLoader()
            if self.data_loader.prepare_data():
                return True, "Данные успешно загружены и подготовлены"
            else:
                return False, "Ошибка при подготовке данных"
        except Exception as e:
            return False, f"Ошибка: {str(e)}"

    def train_model(self):
        """Обучение модели"""
        try:
            if self.data_loader is None:
                success, message = self.load_and_prepare_data()
                if not success:
                    return False, message

            self.model = create_model()
            X_train, y_train = self.data_loader.get_train_data()
            X_test, y_test = self.data_loader.get_test_data()

            self.history = train_model(self.model, X_train, y_train, X_test, y_test)

            # Оценка модели
            metrics = evaluate_model(self.model, X_test, y_test)
            return True, f"Модель обучена. MSE: {metrics.get('mse', 0):.4f}, MAE: {metrics.get('mae', 0):.4f}"

        except Exception as e:
            return False, f"Ошибка при обучении: {str(e)}"

    def make_predictions(self):
        """Выполнение прогнозов"""
        try:
            if self.model is None:
                # Пытаемся загрузить существующую модель
                self.model = load_trained_model()
                if self.model is None:
                    return False, "Модель не обучена. Сначала обучите модель."

            if self.data_loader is None:
                success, message = self.load_and_prepare_data()
                if not success:
                    return False, message

            self.results = make_predictions(self.model, self.data_loader)
            if self.results is not None:
                return True, "Прогнозы успешно выполнены"
            else:
                return False, "Ошибка при выполнении прогнозов"

        except Exception as e:
            return False, f"Ошибка при прогнозировании: {str(e)}"

    def create_plots(self):
        """Создание графиков для визуализации"""
        if self.results is None or self.history is None:
            return None

        plots = {}

        # График обучения
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Обучающая выборка')
        plt.plot(self.history.history['val_loss'], label='Валидационная выборка')
        plt.title('История обучения модели')
        plt.ylabel('Потери (MSE)')
        plt.xlabel('Эпоха')
        plt.legend()
        plt.grid(True)

        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', dpi=100)
        img_buf.seek(0)
        plots['training'] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        plt.close()

        # График прогнозов
        plt.figure(figsize=(12, 6))
        plt.plot(self.results['Actual'], 'bo-', label='Реальные значения', alpha=0.7)
        plt.plot(self.results['Predicted'], 'ro-', label='Предсказания', alpha=0.7)
        plt.title('Сравнение реальных и предсказанных значений')
        plt.ylabel('Значение ОСО')
        plt.xlabel('Временные точки')
        plt.legend()
        plt.grid(True)

        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', dpi=100)
        img_buf.seek(0)
        plots['predictions'] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        plt.close()

        return plots


# Глобальный экземпляр приложения
oso_app = OSOForecastingApp()


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/api/load_data', methods=['POST'])
def api_load_data():
    """API для загрузки данных"""
    success, message = oso_app.load_and_prepare_data()
    return jsonify({'success': success, 'message': message})


@app.route('/api/train_model', methods=['POST'])
def api_train_model():
    """API для обучения модели"""
    success, message = oso_app.train_model()
    return jsonify({'success': success, 'message': message})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API для выполнения прогнозов"""
    success, message = oso_app.make_predictions()

    plots = None
    if success:
        plots = oso_app.create_plots()

    return jsonify({
        'success': success,
        'message': message,
        'plots': plots
    })


@app.route('/api/status')
def api_status():
    """API для получения статуса"""
    status = {
        'data_loaded': oso_app.data_loader is not None,
        'model_trained': oso_app.model is not None,
        'predictions_made': oso_app.results is not None,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)


if __name__ == '__main__':
    # Создаем необходимые папки
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    app.run(debug=True, host='0.0.0.0', port=5000)