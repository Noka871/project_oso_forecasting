import os
import json
import pandas as pd
from datetime import datetime
import glob

class AutoPredictionSaver:

    def __init__(self, save_dir="data/predictions"):
        self.save_dir = save_dir
        self.base_name = "ОСО_predict"
        os.makedirs(save_dir, exist_ok=True)

        self.current_number = self._get_current_number()

    def _get_current_number(self):
        pattern = os.path.join(self.save_dir, f"{self.base_name}*.csv")
        existing_files = glob.glob(pattern)

        if not existing_files:
            return 0

        file_no_num = os.path.join(self.save_dir, f"{self.base_name}.csv")
        has_no_number = os.path.exists(file_no_num)

        if not has_no_number:
            return 0

        max_num = 0
        for file in existing_files:
            filename = os.path.basename(file)
            num_part = filename[len(self.base_name):-4]

            if num_part == "":
                continue

            if num_part.isdigit():
                num = int(num_part)
                if num > max_num:
                    max_num = num

        return max_num + 1

    def save_prediction(self, predictions, input_data=None, model_info=None):
        if self.current_number == 0:
            filename = f"{self.base_name}.csv"
        else:
            filename = f"{self.base_name}{self.current_number}.csv"

        filepath = os.path.join(self.save_dir, filename)

        save_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        data_to_save = {
            'save_time': [save_time],
            'prediction_count': [len(predictions)],
            'predictions': [json.dumps(predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions))],
        }

        if model_info:
            data_to_save['model_info'] = [json.dumps(model_info)]

        df = pd.DataFrame(data_to_save)
        df.to_csv(filepath, index=False, encoding='utf-8')

        json_path = filepath.replace('.csv', '.json')
        full_data = {
            'filename': filename,
            'save_time': save_time,
            'prediction_number': self.current_number if self.current_number > 0 else "base",
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            'model_info': model_info or {},
            'input_data': input_data.tolist() if input_data is not None and hasattr(input_data, 'tolist') else input_data
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Прогноз №{self.current_number if self.current_number > 0 else 'базовый'} сохранен:")
        print(f"   CSV: {filename}")
        print(f"   JSON: {os.path.basename(json_path)}")

        self.current_number += 1

        return filepath

    def get_saved_predictions_list(self):
        if not os.path.exists(self.save_dir):
            return []

        predictions_list = []

        base_file = os.path.join(self.save_dir, f"{self.base_name}.csv")
        if os.path.exists(base_file):
            predictions_list.append({
                'filename': f"{self.base_name}.csv",
                'json_file': f"{self.base_name}.json",
                'number': 'base',
                'path': base_file
            })

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
        if self.current_number == 0:
            return f"{self.base_name}.csv", "базовый"
        else:
            return f"{self.base_name}{self.current_number}.csv", self.current_number