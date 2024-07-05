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
        # if not training:
        #     preprocessing_transforms = get_preprocessing_transforms(
        #         resize=True, random_horizontal_flip=False, normalize=True
        #     )
        #     x = preprocessing_transforms(x)

        out = self.resnet(x)

        # age = self.age_head(out)
        gender = self.gender_head(out)
        gender = self.sig(gender).reshape(-1)

        return gender

    def transform_scores_to_predictions(self, age, gender):
        with torch.no_grad():
            age = self.softmax(age)
            age_pred = torch.argmax(age, dim=1)

            gender_pred = gender > 0.5

        return age_pred, gender_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x, training=True)
        loss = self.gender_loss_fn(preds, y)
        acc = (preds.round() == y).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    # TODO: figure out what this does?
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x, training=True)
        acc = (preds.round() == y).float().mean()

        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x, training=True)
        acc = (preds.round() == y).float().mean()

        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=False)

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
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, **dict(self.config.model_training_config.scheduler_config)
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "interval": "epoch",
            #     "monitor": "val_loss_total",
            #     "frequency": 1,
            # },
        }


class MultiTaskNetWorking(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        super().__init__()
        self.save_hyperparameters()
        self.model = self.create_model()
        self.loss_module = nn.CrossEntropyLoss()

    def create_model(self):
        model = resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        return model

    def _parse_output_to_preds(self, out):
        preds = torch.argmax(out, dim=1)
        return preds

    def forward(self, X):
        # Forward function that is run when visualizing the graph
        return self.model(X)

    def configure_optimizers(self):

        optimizer = optim.AdamW(self.parameters(), lr=0.1, weight_decay=0.0001)

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        X, y_age, y_gender = batch
        out = self.model(X)
        loss = self.loss_module(out, y_gender)

        preds = self._parse_output_to_preds(out)
        acc = (y_gender == preds).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # TODO: reduce code duplication
    def validation_step(self, batch, batch_idx):
        X, y_age, y_gender = batch
        out = self.model(X)
        preds = self._parse_output_to_preds(out)

        acc = (y_gender == preds).float().mean()
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        X, y_age, y_gender = batch
        out = self.model(X)
        preds = self._parse_output_to_preds(out)

        acc = (y_gender == preds).float().mean()
        self.log("test_acc", acc)
