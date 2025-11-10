# main.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from model import ForecastingModel
import threading
import os


class SimpleForecastingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Простое прогнозирование")
        self.root.geometry("1000x700")
        self.root.configure(bg='white')

        # Центрирование окна
        self.center_window()

        # Инициализация модели
        self.model = ForecastingModel()
        self.data = None
        self.forecast_result = None

        # Создание интерфейса
        self.create_interface()

    def center_window(self):
        """Центрирование окна на экране"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def create_interface(self):
        """Создание простого интерфейса"""
        # Главный контейнер
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Шаги работы (простая навигация)
        self.create_steps_section(main_frame)

        # Область контента
        self.create_content_area(main_frame)

        # Статус
        self.create_status_bar(main_frame)

    def create_steps_section(self, parent):
        """Секция с шагами работы"""
        steps_frame = ttk.LabelFrame(parent, text="Шаги работы", padding="10")
        steps_frame.pack(fill=tk.X, pady=(0, 10))

        # Создаем фрейм для кнопок шагов
        buttons_frame = ttk.Frame(steps_frame)
        buttons_frame.pack(fill=tk.X)

        # Кнопки шагов
        self.step1_btn = ttk.Button(buttons_frame, text="1. Загрузить данные",
                                    command=self.load_data, width=20)
        self.step1_btn.pack(side=tk.LEFT, padx=5)

        self.step2_btn = ttk.Button(buttons_frame, text="2. Настроить модель",
                                    command=self.show_model_settings, width=20, state=tk.DISABLED)
        self.step2_btn.pack(side=tk.LEFT, padx=5)

        self.step3_btn = ttk.Button(buttons_frame, text="3. Запустить прогноз",
                                    command=self.run_forecast, width=20, state=tk.DISABLED)
        self.step3_btn.pack(side=tk.LEFT, padx=5)

        self.step4_btn = ttk.Button(buttons_frame, text="4. Сохранить результаты",
                                    command=self.export_results, width=20, state=tk.DISABLED)
        self.step4_btn.pack(side=tk.LEFT, padx=5)

    def create_content_area(self, parent):
        """Основная область контента"""
        # Разделяем на левую и правую части
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Левая панель - настройки и информация
        self.create_left_panel(content_frame)

        # Правая панель - график
        self.create_right_panel(content_frame)

    def create_left_panel(self, parent):
        """Левая панель с настройками"""
        left_frame = ttk.Frame(parent, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        # Информация о данных
        self.data_info_frame = ttk.LabelFrame(left_frame, text="Информация о данных", padding="10")
        self.data_info_frame.pack(fill=tk.X, pady=(0, 10))

        self.data_info_label = ttk.Label(self.data_info_frame, text="Данные не загружены",
                                         background="#f8f9fa", padding="5")
        self.data_info_label.pack(fill=tk.X)

        # Настройки модели
        self.model_frame = ttk.LabelFrame(left_frame, text="Настройки прогноза", padding="10")
        self.model_frame.pack(fill=tk.X, pady=(0, 10))

        # Выбор колонки
        ttk.Label(self.model_frame, text="Что прогнозируем:").pack(anchor=tk.W)
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(self.model_frame, textvariable=self.column_var, state="readonly")
        self.column_combo.pack(fill=tk.X, pady=5)

        # Период прогноза
        ttk.Label(self.model_frame, text="Дней прогноза:").pack(anchor=tk.W)
        self.period_var = tk.StringVar(value="30")
        period_entry = ttk.Entry(self.model_frame, textvariable=self.period_var)
        period_entry.pack(fill=tk.X, pady=5)

        # Метод прогноза
        ttk.Label(self.model_frame, text="Метод:").pack(anchor=tk.W)
        self.method_var = tk.StringVar(value="Авто")
        method_combo = ttk.Combobox(self.model_frame, textvariable=self.method_var,
                                    values=["Авто", "Линейная регрессия", "Случайный лес", "Нейросеть"])
        method_combo.pack(fill=tk.X, pady=5)

        # Кнопка обучения
        self.train_btn = ttk.Button(self.model_frame, text="Обучить модель",
                                    command=self.train_model, state=tk.DISABLED)
        self.train_btn.pack(fill=tk.X, pady=10)

        # Результаты
        self.results_frame = ttk.LabelFrame(left_frame, text="Результаты", padding="10")
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(self.results_frame, height=10, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_text.insert(tk.END, "Здесь появятся результаты прогноза...")
        self.results_text.config(state=tk.DISABLED)

    def create_right_panel(self, parent):
        """Правая панель с графиком"""
        right_frame = ttk.LabelFrame(parent, text="График прогноза")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Создание графика
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Простые кнопки управления графиком
        graph_controls = ttk.Frame(right_frame)
        graph_controls.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(graph_controls, text="Обновить график",
                   command=self.update_plot).pack(side=tk.LEFT, padx=2)
        ttk.Button(graph_controls, text="Сохранить график",
                   command=self.save_plot).pack(side=tk.LEFT, padx=2)
        ttk.Button(graph_controls, text="Показать данные",
                   command=self.show_data_preview).pack(side=tk.LEFT, padx=2)

    def create_status_bar(self, parent):
        """Статус бар"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_var = tk.StringVar(value="Готов к работе")
        status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                 relief=tk.SUNKEN, anchor=tk.W, padding="5")
        status_label.pack(fill=tk.X)

    def load_data(self):
        """Простая загрузка данных"""
        try:
            file_path = filedialog.askopenfilename(
                title="Выберите CSV файл с данными",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
            )

            if file_path:
                self.status_var.set("Загружаем данные...")

                # Определяем тип файла и загружаем
                if file_path.endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    self.data = pd.read_excel(file_path)
                else:
                    messagebox.showerror("Ошибка", "Поддерживаются только CSV и Excel файлы")
                    return

                # Обновляем информацию
                filename = os.path.basename(file_path)
                info_text = f"Файл: {filename}\nСтрок: {len(self.data)}\nКолонок: {len(self.data.columns)}"
                self.data_info_label.config(text=info_text)

                # Заполняем выбор колонки
                self.column_combo['values'] = list(self.data.columns)
                if len(self.data.columns) > 0:
                    self.column_combo.set(self.data.columns[0])

                # Активируем следующий шаг
                self.step2_btn.config(state=tk.NORMAL)
                self.train_btn.config(state=tk.NORMAL)
                self.status_var.set(f"Данные загружены: {filename}")

                # Показываем превью данных
                self.show_data_preview()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")
            self.status_var.set("Ошибка загрузки")

    def show_data_preview(self):
        """Показ превью данных"""
        if self.data is None:
            return

        # Создаем простое окно с превью
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Просмотр данных")
        preview_window.geometry("600x400")

        # Текст с информацией о данных
        info_text = f"Первые 10 строк из {len(self.data)}:\n\n"
        info_text += self.data.head(10).to_string()

        text_widget = tk.Text(preview_window, wrap=tk.NONE)
        scroll_x = ttk.Scrollbar(preview_window, orient=tk.HORIZONTAL, command=text_widget.xview)
        scroll_y = ttk.Scrollbar(preview_window, orient=tk.VERTICAL, command=text_widget.yview)

        text_widget.configure(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)

        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

    def show_model_settings(self):
        """Показ настроек модели (просто активируем соответствующую область)"""
        self.model_frame.tkraise()
        messagebox.showinfo("Настройка модели",
                            "Выберите колонку для прогноза и настройте параметры в левой панели")

    def train_model(self):
        """Обучение модели"""
        if self.data is None or not self.column_var.get():
            messagebox.showwarning("Внимание", "Сначала загрузите данные и выберите колонку для прогноза")
            return

        try:
            self.status_var.set("Обучаем модель...")
            self.train_btn.config(state=tk.DISABLED)

            # Запуск в отдельном потоке
            thread = threading.Thread(target=self._train_thread)
            thread.daemon = True
            thread.start()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обучении: {str(e)}")
            self.train_btn.config(state=tk.NORMAL)

    def _train_thread(self):
        """Поток для обучения модели"""
        try:
            target_column = self.column_var.get()
            method = self.method_var.get()

            # Обучаем модель
            self.model.train(self.data, target_column, method)

            # Обновляем интерфейс в основном потоке
            self.root.after(0, self._on_training_complete)

        except Exception as e:
            self.root.after(0, lambda: self._on_training_error(str(e)))

    def _on_training_complete(self):
        """После успешного обучения"""
        self.train_btn.config(state=tk.NORMAL)
        self.step3_btn.config(state=tk.NORMAL)
        self.status_var.set("Модель обучена!")

        # Показываем метрики
        self.show_training_results()
        messagebox.showinfo("Готово", "Модель успешно обучена!")

    def _on_training_error(self, error_msg):
        """При ошибке обучения"""
        self.train_btn.config(state=tk.NORMAL)
        self.status_var.set("Ошибка обучения")
        messagebox.showerror("Ошибка", f"Не удалось обучить модель:\n{error_msg}")

    def show_training_results(self):
        """Показ результатов обучения"""
        if hasattr(self.model, 'metrics'):
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)

            results = "=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===\n\n"
            results += f"Метод: {self.method_var.get()}\n"
            results += f"Прогнозируем: {self.column_var.get()}\n\n"
            results += "Метрики качества:\n"

            for metric, value in self.model.metrics.items():
                results += f"  {metric}: {value:.4f}\n"

            self.results_text.insert(tk.END, results)
            self.results_text.config(state=tk.DISABLED)

    def run_forecast(self):
        """Запуск прогнозирования"""
        if not self.model.is_trained:
            messagebox.showwarning("Внимание", "Сначала обучите модель!")
            return

        try:
            self.status_var.set("Выполняем прогноз...")
            periods = int(self.period_var.get())

            # Прогнозирование
            self.forecast_result = self.model.forecast(periods)

            # Обновляем график и результаты
            self.update_plot()
            self.show_forecast_results()

            # Активируем следующий шаг
            self.step4_btn.config(state=tk.NORMAL)
            self.status_var.set("Прогноз готов!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка прогнозирования:\n{str(e)}")
            self.status_var.set("Ошибка прогноза")

    def show_forecast_results(self):
        """Показ результатов прогноза"""
        if self.forecast_result is not None:
            self.results_text.config(state=tk.NORMAL)
            self.results_text.insert(tk.END, f"\n\n=== ПРОГНОЗ НА {len(self.forecast_result)} ДНЕЙ ===\n\n")

            # Показываем первые 10 значений прогноза
            for i, value in enumerate(self.forecast_result[:10], 1):
                self.results_text.insert(tk.END, f"День {i}: {value:.2f}\n")

            if len(self.forecast_result) > 10:
                self.results_text.insert(tk.END, f"... и ещё {len(self.forecast_result) - 10} значений\n")

            self.results_text.config(state=tk.DISABLED)

    def update_plot(self):
        """Обновление графика"""
        self.figure.clear()

        if self.data is not None and self.model.is_trained:
            ax = self.figure.add_subplot(111)
            target_column = self.column_var.get()

            # Исторические данные
            history = self.data[target_column].values
            ax.plot(history, 'b-', label='Исторические данные', linewidth=2)

            # Прогноз
            if self.forecast_result is not None:
                forecast_start = len(history)
                forecast_x = range(forecast_start, forecast_start + len(self.forecast_result))
                ax.plot(forecast_x, self.forecast_result, 'r--', label='Прогноз', linewidth=2)

                # Область уверенности (простая)
                confidence = self.forecast_result * 0.1  # 10% уверенность
                ax.fill_between(forecast_x,
                                self.forecast_result - confidence,
                                self.forecast_result + confidence,
                                alpha=0.2, color='red')

            ax.set_title(f'Прогноз: {target_column}')
            ax.set_xlabel('Время')
            ax.set_ylabel(target_column)
            ax.legend()
            ax.grid(True, alpha=0.3)

        else:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Загрузите данные и обучите модель\nдля отображения графика',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('График прогноза')

        self.figure.tight_layout()
        self.canvas.draw()

    def save_plot(self):
        """Сохранение графика"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
            )

            if file_path:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Успех", f"График сохранен:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить график:\n{str(e)}")

    def export_results(self):
        """Экспорт результатов"""
        if self.forecast_result is None:
            messagebox.showwarning("Внимание", "Сначала выполните прогноз!")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
            )

            if file_path:
                # Создаем DataFrame с прогнозом
                forecast_df = pd.DataFrame({
                    'День': range(1, len(self.forecast_result) + 1),
                    'Прогноз': self.forecast_result
                })

                # Сохраняем в нужном формате
                if file_path.endswith('.csv'):
                    forecast_df.to_csv(file_path, index=False, encoding='utf-8')
                else:
                    forecast_df.to_excel(file_path, index=False)

                messagebox.showinfo("Успех", f"Результаты сохранены:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить результаты:\n{str(e)}")


def main():
    # Создаем главное окно
    root = tk.Tk()

    # Устанавливаем иконку (если есть)
    try:
        root.iconbitmap('icon.ico')  # Можно добавить простую иконку
    except:
        pass

    # Создаем приложение
    app = SimpleForecastingApp(root)

    # Запускаем главный цикл
    root.mainloop()


if __name__ == "__main__":
    main()