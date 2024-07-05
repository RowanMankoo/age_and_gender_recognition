from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config.core import Config
from src.training import AgeTransformer, get_preprocessing_transforms
from torchvision.transforms.functional import to_tensor


def get_dataloaders(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    config: Config,
):
    dataframes = {
        "train": {
            "df": df_train,
            "batch_size": config.model_training_config.data_loader_config.train_batch_size,
        },
        "test": {
            "df": df_test,
            "batch_size": config.model_training_config.data_loader_config.test_batch_size,
        },
        "val": {
            "df": df_test,
            "batch_size": config.model_training_config.data_loader_config.val_batch_size,
        },
    }
    for _, dataloader_config in dataframes.items():
        yield _get_dataloader(
            dataloader_config["df"],
            dataloader_config["batch_size"],
            config.model_config.age_labels_to_bins,
        )


def _get_dataloader(df: pd.DataFrame, batch_size: int, age_labels_to_bins: dict):

    dataset = IMDBDataset(df)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=MyCollate(batch_size, age_labels_to_bins),
    )
    return dataloader


class IMDBDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transforms: transforms.Compose = get_preprocessing_transforms(
            resize=True, random_horizontal_flip=True, normalize=True
        ),
    ):
        self.folder_directory = Path("Data/crop_part1")
        self.df = df.reset_index(drop=True)
        self.transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.df["filename"][index]
        age = self.df["age"][index]
        gender = self.df["gender"][index]

        # Load image into Tensor
        filename = self.folder_directory / filename
        image = Image.open(filename)
        # image = to_tensor(image)
        # image = self.transform(image)

        label = torch.tensor([int(age), int(gender)], dtype=torch.float)

        return image, label


# TODO: implement this
from torchvision.datasets.vision import VisionDataset


class UTKFaceDataset(VisionDataset):
    def __init__(self, df, root=Path("Data/crop_part1"), transforms=None):
        super(UTKFaceDataset, self).__init__(root=root, transforms=transforms)
        self.df = df

    def __getitem__(self, index):

        filename = self.df["filename"][index]
        age = self.df["age"][index]
        gender = self.df["gender"][index]

        filename = self.root / filename
        image = Image.open(filename)
        if self.transforms is not None:
            image = self.transforms(image)

        return (
            image,
            torch.tensor(float(age), dtype=torch.long),
            torch.tensor(float(gender), dtype=torch.long),
        )

    def __len__(self):
        return len(self.df)