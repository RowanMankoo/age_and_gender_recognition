import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

from src.core import Config
from src.training import get_preprocessing_transforms


# TODO: fix this
def freeze_layers_resnet18(model):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Unfreeze the first few layers
    for param in model.conv1.parameters():
        param.requires_grad = True
    for param in model.bn1.parameters():
        param.requires_grad = True
    for param in model.layer1.parameters():
        param.requires_grad = True

    return model


class HeadBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HeadBlock, self).__init__()

        self.fc1 = nn.Linear(in_channels, 500)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(500, out_channels)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class Identity(nn.Module):
    def forward(self, x):
        return x


class MultiTaskNet(pl.LightningModule):
    def __init__(self, config: Config, age_class_weights: torch.Tensor):
        super(MultiTaskNet, self).__init__()
        self.config = config

        self.age_loss_fn = nn.CrossEntropyLoss(weight=age_class_weights)
        self.gender_loss_fn = nn.BCELoss()

        self._init_model_architecture()

    def _init_model_architecture(self):
        if self.config.model_config.pretrained:
            weights = models.ResNet18_Weights
        else:
            weights = None

        self.resnet = models.resnet18(
            weights=weights,
        )
        self.resnet.fc = Identity()

        if self.config.model_config.freeze_starting_layers:
            self.resnet = freeze_layers_resnet18(self.resnet)

        resnet_output_shape = 512
        self.sig = nn.Sigmoid()

        # Ordinal Classification
        self.age_head = HeadBlock(
            in_channels=resnet_output_shape,
            out_channels=len(self.config.model_config.age_labels_to_bins),
        )
        self.softmax = nn.Softmax(dim=1)
        self.gender_head = HeadBlock(in_channels=resnet_output_shape, out_channels=1)

    def forward(self, x: torch.Tensor, training: bool):
        if not training:
            preprocessing_transforms = get_preprocessing_transforms(
                resize=True, random_horizontal_flip=False, normalize=True
            )
            x = preprocessing_transforms(x)

        out = self.resnet(x)

        age = self.age_head(out)
        gender = self.gender_head(out)
        gender = self.sig(gender).reshape(-1)

        return age, gender

    def transform_scores_to_predictions(self, age, gender):
        with torch.no_grad():
            age = self.softmax(age)
            age_pred = torch.argmax(age, dim=1)

            gender_pred = gender > 0.5

        return age_pred, gender_pred

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    # TODO: figure out what this does?
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch):
        # TODO: figure this one out
        x, (y_age, y_gender) = batch
        age, gender = self.forward(x, training=False)
        age_pred, gender_pred = self.transform_scores_to_predictions(age, gender)

        return age_pred, gender_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            **dict(self.config.model_training_config.optimizer_config),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **dict(self.config.model_training_config.scheduler_config)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss_total",
                "frequency": 1,
            },
        }

    def _shared_step(self, batch, name: str):
        x, (y_age, y_gender) = batch
        age, gender = self.forward(x, training=True)

        self.log_dict(
            {
                name + "_loss_age": (loss_age := self.age_loss_fn(age, y_age)),
                name
                + "_loss_gender": (
                    loss_gender := self.gender_loss_fn(gender, y_gender)
                ),
                name + "_loss_total": (total_loss := loss_age + loss_gender),
            },
            prog_bar=True,
        )
        return total_loss
