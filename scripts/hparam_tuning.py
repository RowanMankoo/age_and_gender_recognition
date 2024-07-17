import argparse

import numpy as np
from sklearn.model_selection import ParameterSampler

from src.training import train_model

parser = argparse.ArgumentParser()
parser.add_argument("--n_iter_per_resnet", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--early_stopping_patience", type=int, default=30)
parser.add_argument("--scheduler_patience", type=int, default=10)
parser.add_argument("--max_epochs", type=int, default=300)

args = parser.parse_args()

param_grid = {
    "learning_rate": np.logspace(-6, -3, num=1000),
    "age_hidden_head_dim": range(1, 100),
    "gender_hidden_head_dim": range(2, 100),
}

resnet_models = [
    # "resnet18",
    "resnet34",
    "resnet50",
    # "resnext50_32x4d",
    "wide_resnet50_2",
]

for resnet_model in resnet_models:
    print(f"Training with {resnet_model}")
    param_sampler = ParameterSampler(param_grid, n_iter=args.n_iter_per_resnet)
    for i, hparams in enumerate(param_sampler):
        print(
            "-" * 20
            + f"{resnet_model} Iteration {i+1}/{args.n_iter_per_resnet}"
            + "-" * 20
        )
        print(hparams)
        hparams["resnet_model"] = resnet_model
        hparams["scheduler_patience"] = args.scheduler_patience
        train_model(
            max_epochs=args.max_epochs,
            production_model=False,
            batch_size=args.batch_size,
            early_stopping_patience=args.early_stopping_patience,
            **hparams,
        )
