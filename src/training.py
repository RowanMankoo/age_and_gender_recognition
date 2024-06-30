import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms


# TODO: figure out how to implement this nicely
class AgeTransformer:
    def __init__(self, age_labels_to_bins: dict):
        self.age_labels_to_bins = age_labels_to_bins
        self.bins = [0] + [
            int(label.split(",")[1].strip(" )"))
            for label in age_labels_to_bins.values()
        ]
        self.labels = list(age_labels_to_bins.keys())

    def ages_to_labels(self, ages: np.array):
        ordinal_labels = pd.cut(
            ages, bins=self.bins, labels=self.labels, include_lowest=True
        )
        return torch.Tensor(ordinal_labels.to_list())

    def labels_to_ages(self, labels: np.array):
        ages = [self.bins[label] for label in labels]
        return torch.Tensor(ages)


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
    df_train = df_train.drop(columns="strata")
    df_val = df_val.drop(columns="strata")
    df_test = df_test.drop(columns="strata")

    return df_train, df_val, df_test


def get_age_class_weights(df: pd.DataFrame, age_labels_to_bins: dict):

    age_transformer = AgeTransformer(age_labels_to_bins)
    age_labels = age_transformer.ages_to_labels(df["age"])
    age_counts = torch.bincount(age_labels.to(torch.int64))
    age_class_weights = len(age_labels) / (len(age_counts) * age_counts)

    return age_class_weights


def get_preprocessing_transforms(
    *,
    resize: bool = True,
    random_horizontal_flip: bool = False,
    normalize: bool = True,
) -> transforms.Compose:
    # TODO: add bounding box, check if this runs on GPU first?
    transform_conditions = [
        (resize, transforms.Resize(256)),
        (resize, transforms.CenterCrop(224)),
        (random_horizontal_flip, transforms.RandomHorizontalFlip()),
        (True, transforms.ToTensor()),
        (
            normalize,
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # these values from from mean and std of the imagenet dataset
        ),
    ]

    transform_list = [
        transform for condition, transform in transform_conditions if condition
    ]
    return transforms.Compose(transform_list)
