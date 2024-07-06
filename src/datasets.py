from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class UTKFaceDataset(VisionDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root: Path = Path("Data/crop_part1"),
        transforms: Optional[Callable] = None,
    ):
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
            torch.tensor(float(age), dtype=torch.float),
            torch.tensor(float(gender), dtype=torch.long),
        )

    def __len__(self):
        return len(self.df)
