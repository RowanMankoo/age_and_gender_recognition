from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.core import Config
from src.training import AgeTransformer, get_preprocessing_transforms


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
            resize=True, random_horizontal_flip=False, normalize=False
        ),
    ):
        self.folder_directory = Path("Data/imdb_crop")
        self.df = df.reset_index(drop=True)
        self.transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.df["image_path"][index]
        age = self.df["age"][index]
        gender = self.df["gender"][index]

        # Load image into Tensor
        image_path = self.folder_directory / image_path
        image = Image.open(image_path)
        image = self.transform(image)

        label = torch.tensor([int(age), int(gender)], dtype=torch.float)

        return image, label


class MyCollate:
    def __init__(self, batch_size: int, age_labels_to_bins: dict):
        self.batch_size = batch_size
        self.age_labels_to_bins = age_labels_to_bins

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        images, labels = zip(*batch)

        # Convert images to tensor and reorder dimensions
        images = torch.stack(images, dim=0)

        # Convert labels to tensor
        labels = torch.stack(labels, dim=0)

        # Transform ages to categorial labels
        ages = labels[:, 0].long()
        # TODO: don't like use of this class here
        age_transformer = AgeTransformer(self.age_labels_to_bins)
        ages = age_transformer.ages_to_labels(ages.numpy())

        gender = labels[:, 1]

        labels = (ages, gender)

        return images, labels
