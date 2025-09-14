"""
Конфигурация проекта с загрузкой из YAML
"""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ModelConfig:
    name: str
    type: str
    layers: List[Dict[str, Any]]
    compile: Dict[str, Any]


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    validation_split: float
    early_stopping: Dict[str, Any]


@dataclass
class DataConfig:
    sequence_length: int
    test_size: float
    features: List[str]
    target: str


class Config:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config_path = Path(config_path)
        self.load_config()

    def load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        self.model = ModelConfig(**config_data['model'])
        self.training = TrainingConfig(**config_data['training'])
        self.data = DataConfig(**config_data['data'])

    def save_config(self, path: str):
        config_data = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__
        }
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True)