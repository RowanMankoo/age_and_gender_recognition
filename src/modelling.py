from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import lightning as L
from torchvision.models import resnet18
import torch.optim as optim


from config.core import Config
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


class MultiTaskNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        # TODO: add option to swap resnet out?
        self.base_model, num_ftrs = (
            self.create_base_model_and_get_num_features()
        )  # Get the base model and the number of input features
        self.gender_head = nn.Linear(num_ftrs, 2)  # Classification head for gender
        self.age_head = nn.Linear(num_ftrs, 1)  # Regression head for age

        self.loss_module_gender = nn.CrossEntropyLoss()
        self.loss_module_age = nn.MSELoss()

    def create_base_model_and_get_num_features(self):
        base_model = resnet18(pretrained=True)
        num_ftrs = (
            base_model.fc.in_features
        )  # Get the number of input features of the final layer
        base_model.fc = nn.Identity()  # Remove the final layer
        return base_model, num_ftrs

    def forward(self, X):
        features = self.base_model(X)
        gender_out = self.gender_head(features)
        # age_out = self.age_head(features)
        return gender_out

    def _parse_output_to_preds(self, out):
        preds = torch.argmax(out, dim=1)
        return preds

    def configure_optimizers(self):

        optimizer = optim.AdamW(self.parameters(), lr=0.1, weight_decay=0.0001)

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        X, y_age, y_gender = batch
        out = self.forward(X)
        loss = self.loss_module_gender(out, y_gender)

        preds = self._parse_output_to_preds(out)
        acc = (y_gender == preds).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # TODO: reduce code duplication
    def validation_step(self, batch, batch_idx):
        X, y_age, y_gender = batch
        out = self.forward(X)
        preds = self._parse_output_to_preds(out)

        acc = (y_gender == preds).float().mean()
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        X, y_age, y_gender = batch
        out = self.forward(X)
        preds = self._parse_output_to_preds(out)

        acc = (y_gender == preds).float().mean()
        self.log("test_acc", acc)
