import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms


# TODO: figure out how to implement this nicely
class AgeTransformer:
    def __init__(self, age_labels_to_bins: dict):
        self.age_labels_to_bins = age_labels_to_bins
        self.bins = [0] + [int(label.split(",")[1].strip(" )")) for label in age_labels_to_bins.values()]
        self.labels = list(age_labels_to_bins.keys())

    def ages_to_labels(self, ages: np.array):
        ordinal_labels = pd.cut(ages, bins=self.bins, labels=self.labels, include_lowest=True)
        return torch.Tensor(ordinal_labels.to_list())

    def labels_to_ages(self, labels: np.array):
        ages = [self.bins[label] for label in labels]
        return torch.Tensor(ages)


def train_test_split_celeb_ids(df, test_size=0.2, random_state=None):
    """
    Split a Pandas DataFrame into train and test sets while ensuring no shared celeb_ids in the test set.
    """
    unique_celeb_ids = df["celeb_id"].unique()
    train_ids, test_ids = train_test_split(unique_celeb_ids, test_size=test_size, random_state=random_state)

    df_train = df[df["celeb_id"].isin(train_ids)]
    df_test = df[df["celeb_id"].isin(test_ids)]

    return df_train, df_test


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

    transform_list = [transform for condition, transform in transform_conditions if condition]
    return transforms.Compose(transform_list)
