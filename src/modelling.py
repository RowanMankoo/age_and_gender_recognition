import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

from src.core import Config
from src.training import get_preprocessing_transforms


def freeze_layers(model, num_layers=6):
    ct = 0
    for child in model.children():
        ct += 1
        if ct < num_layers:
            for param in child.parameters():
                param.requires_grad = False

    return model


class HeadBlock(nn.Module):
    def __init__(self, out_channels, in_channels):
        super(HeadBlock, self).__init__()

        self.fc1 = nn.Linear(in_channels, 500)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(500, out_channels)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class MultiTaskNet(pl.LightningModule):
    def __init__(self, config: Config, age_class_weights: torch.Tensor):
        super(MultiTaskNet, self).__init__()
        self.config = config

        self.age_loss_fn = nn.CrossEntropyLoss(weight=age_class_weights)
        self.gender_loss_fn = nn.BCELoss()

        self.resnet = models.resnet18(
            pretrained=self.config.model_config.pretrained,
        )
        if self.config.model_config.freeze_starting_layers:
            self.resnet = freeze_layers(self.resnet)

        resnet_output_shape = self.resnet.fc.in_features
        self.sig = nn.Sigmoid()

        # Ordinal Classification
        self.age_head = HeadBlock(8, in_channels=resnet_output_shape)
        self.softmax = nn.Softmax(dim=1)
        # Binary Classification Task so output of dimension 2
        self.gender_head = HeadBlock(1, in_channels=resnet_output_shape)

    def forward(self, x: torch.Tensor, training=False):
        if training:
            preprocessing_transforms = get_preprocessing_transforms(
                resize=False, random_horizontal_flip=True, normalize=True
            )
        else:
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

    def predict(self, x, training=False):
        age, gender = self.forward(x, training=training)
        age_pred, gender_pred = self.transform_scores_to_predictions(age, gender)

        return age_pred, gender_pred

    def training_step(self, batch, batch_idx):
        x, y_age, y_gender = batch
        age, gender = self.forward(x, training=True)

        # TODO: remove this
        loss_age = F.cross_entropy(age, y_age)
        loss_gender = F.binary_cross_entropy(gender, y_gender)
        loss = loss_age + loss_gender
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_age, y_gender = batch
        age, gender = self.forward(x)
        loss_age = F.cross_entropy(age, y_age)
        loss_gender = F.binary_cross_entropy(gender, y_gender)
        loss = loss_age + loss_gender
        self.log("val_loss", loss)

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
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
