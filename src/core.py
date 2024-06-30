from enum import Enum
from typing import Dict

import yaml
from pydantic import BaseModel


class ResnetBackboneEnum(str, Enum):
    resnet18 = "resnet18"
    resnet50 = "resnet50"
    wide_resnet50_2 = "wide_resnet50_2"
    wide_resnet101_2 = "wide_resnet101_2"


class DataLoaderConfig(BaseModel):
    train_batch_size: int
    test_batch_size: int
    val_batch_size: int


class OptimizerConfig(BaseModel):
    lr: int


class SchedulerConfig(BaseModel):
    mode: str
    factor: float
    patience: int
    threshold: float
    threshold_mode: str
    cooldown: int
    min_lr: float
    eps: float
    verbose: bool


class ModelConfig(BaseModel):
    age_labels_to_bins: Dict[int, str]
    resnet_backbone: ResnetBackboneEnum
    pretrained: bool
    freeze_starting_layers: bool

    class Config:
        use_enum_values = True


class ModelTrainingConfig(BaseModel):
    data_loader_config: DataLoaderConfig
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig


class Config(BaseModel):
    model_config: ModelConfig
    model_training_config: ModelTrainingConfig


def load_and_validate_config(
    path_to_model_config: str,
    path_to_model_training_config: str,
) -> Config:

    with open(path_to_model_config, "r") as file:
        model_config = yaml.safe_load(file)

    with open(path_to_model_training_config, "r") as file:
        model_training_config = yaml.safe_load(file)

    config = Config(
        model_config=model_config,
        model_training_config=model_training_config,
    )

    return config
