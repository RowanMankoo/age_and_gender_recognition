import argparse

import numpy as np
from sklearn.model_selection import ParameterSampler

from src.training import train_model

parser = argparse.ArgumentParser()
parser.add_argument("--n_iter_per_resnet", type=int, default=15)
args = parser.parse_args()

param_grid = {
    "learning_rate": np.logspace(-5, -1, num=1000),
    "age_hidden_head_dim": range(1, 100),
    "gender_hidden_head_dim": range(2, 100),
}

resnet_models = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext50_32x4d",
    "wide_resnet50_2",
]

for idx, resnet_model in enumerate(resnet_models):
    print(f"Training with {resnet_model}")
    param_sampler = ParameterSampler(param_grid, n_iter=args.n_iter_per_resnet, random_state=idx)
    for i, hparams in enumerate(param_sampler):
        print("-" * 20 + f"{resnet_model} Iteration {i+1}/{args.n_iter_per_resnet}" + "-" * 20)
        print(hparams)
        hparams["resnet_model"] = resnet_model
        train_model(max_epochs=4, production_model=False, batch_size=64, **hparams)
