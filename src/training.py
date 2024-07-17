import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm
from typing import Tuple
from src.datasets import UTKFaceDataset
from src.modelling import MultiTaskNet


def split_dataset(df, train_size=0.6, val_size=0.2):
    # Create a new column for stratification
    df["strata"] = df["age"].astype(str) + "_" + df["gender"].astype(str)

    # Group less frequent classes into a single class
    counts = df["strata"].value_counts()
    df.loc[df["strata"].isin(counts[counts == 1].index), "strata"] = "rare_class"

    # Calculate test size
    test_size = 1 - train_size - val_size

    # Split into train+val and test
    df_train_val, df_test = train_test_split(
        df, test_size=test_size, stratify=df["strata"], random_state=42
    )

    # Calculate the ratio of val_size with respect to train_size + val_size for the second split
    val_size_ratio = val_size / (train_size + val_size)

    # Split train+val into train and val
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_size_ratio,
        stratify=df_train_val["strata"],
        random_state=42,
    )

    # Drop the strata column
    df_train = df_train.drop(columns="strata").reset_index(drop=True)
    df_val = df_val.drop(columns="strata").reset_index(drop=True)
    df_test = df_test.drop(columns="strata").reset_index(drop=True)

    return df_train, df_val, df_test


# def get_preprocessing_transforms(
#     *,
#     resize: bool = True,
#     random_horizontal_flip: bool = False,
#     normalize: bool = True,
# ) -> transforms.Compose:
#     # TODO: add bounding box, check if this runs on GPU first?
#     transform_conditions = [
#         (resize, transforms.Resize(256)),
#         (resize, transforms.CenterCrop(224)),
#         (random_horizontal_flip, transforms.RandomHorizontalFlip()),
#         (True, transforms.ToTensor()),
#         (
#             normalize,
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#             ),  # these values from from mean and std of the imagenet dataset
#         ),
#     ]

#     transform_list = [transform for condition, transform in transform_conditions if condition]
#     return transforms.Compose(transform_list)


def calculate_mean_std(dataset, batch_size=100):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    mean = 0.0
    std = 0.0
    nb_samples = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for data, _, _ in tqdm(loader, desc="Calculating mean and std"):
        data = data.to(device)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean.cpu(), std.cpu()


def train_model(
    max_epochs: int,
    production_model: bool,
    batch_size: int,
    early_stopping_patience: int,
    **hparams,
):
    logger = TensorBoardLogger(
        "tb_logs", name=hparams["resnet_model"], default_hp_metric=False, log_graph=True
    )

    df_metadata = pd.read_csv("Data/metadata.csv")
    if production_model:
        df_train = df_metadata
        df_val = df_metadata
        df_test = df_metadata
    else:
        df_train, df_val, df_test = split_dataset(
            df_metadata, train_size=0.6, val_size=0.2
        )

    train_dataset = UTKFaceDataset(df_train, transforms=ToTensor())
    DATA_MEANS, DATA_STD = calculate_mean_std(train_dataset)
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(DATA_MEANS, DATA_STD)]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(DATA_MEANS, DATA_STD),
        ]
    )

    train_dataset = UTKFaceDataset(df=df_train, transforms=train_transform)
    val_dataset = UTKFaceDataset(df=df_val, transforms=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        UTKFaceDataset(df=df_test, transforms=test_transform),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True, mode="min", monitor="val/loss_combined"
    )

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            EarlyStopping(
                monitor="val/loss_combined",
                min_delta=0,
                patience=early_stopping_patience,
                mode="min",
            ),
        ],
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
    )

    model = MultiTaskNet(**hparams)

    trainer.fit(model, train_loader, val_loader)

    if not production_model:
        trainer.validate(model, val_loader)
        best_model = MultiTaskNet.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        trainer.test(best_model, test_loader)


class CustomTrainer:
    def __init__(
        self,
        max_epochs,
        production_model,
        batch_size,
        early_stopping_patience,
    ):
        self.max_epochs = max_epochs
        self.production_model = production_model
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience

        self.df_metadata = pd.read_csv("Data/metadata.csv")

    def train_hparam_set(self, hparams: dict) -> None:

        logger = self._instantiate_logger(hparams["resnet_model"])
        df_train, df_val, df_test = split_dataset(
            self.df_metadata, train_size=0.6, val_size=0.2
        )

    def train_production_model(self) -> None:

        logger = self._instantiate_logger("production_model")
        df_train = self.df_metadata

    def _instantiate_logger(self, name: str) -> TensorBoardLogger:
        return TensorBoardLogger(
            "tb_logs",
            name=name,
            default_hp_metric=False,
            log_graph=True,
        )

    def _calculate_mean_std(
        self, df: pd.DataFrame, batch_size=100
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        dataset = UTKFaceDataset(df, transforms=ToTensor())
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        mean = 0.0
        std = 0.0
        nb_samples = 0.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for data, _, _ in tqdm(loader, desc="Calculating mean and std"):
            data = data.to(device)
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        return mean.cpu(), std.cpu()
