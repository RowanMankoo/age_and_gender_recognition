import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision.models import resnet18
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
        self.gender_loss_component_weight = 0.99
        self.age_loss_component_weight = 0.01

        self.loss_module_gender = nn.CrossEntropyLoss()
        self.loss_module_age = nn.MSELoss()

        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.rand(1, 3, 224, 224)

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
        age_out = self.age_head(features)
        return gender_out, age_out

    def _parse_output_to_preds(self, *, out_gender, out_age):
        preds_gender = torch.argmax(out_gender, dim=1)
        preds_age = torch.round(out_age)
        return preds_gender, preds_age

    def configure_optimizers(self):

        optimizer = optim.AdamW(self.parameters(), lr=0.1, weight_decay=0.0001)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.2,
                    patience=10,
                    threshold=0.001,
                    threshold_mode="rel",
                ),
                "interval": "epoch",
                "monitor": "val_loss_combined",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def _shared_step(self, batch, batch_idx, prefix):
        X, y_age, y_gender = batch
        out_gender, out_age = self.forward(X)

        loss_gender = self.loss_module_gender(out_gender, y_gender)
        loss_age = self.loss_module_age(out_age.view(-1), y_age)
        combined_loss = (
            loss_gender * self.gender_loss_component_weight
            + loss_age * self.age_loss_component_weight
        )

        preds_gender, _ = self._parse_output_to_preds(
            out_gender=out_gender, out_age=out_age
        )

        acc_gender = (y_gender == preds_gender).float().mean()
        mse_age = F.mse_loss(out_age.view(-1), y_age)

        self.log_dict(
            {
                f"{prefix}_loss_gender": loss_gender,
                f"{prefix}_loss_age": loss_age,
                f"{prefix}_loss_combined": combined_loss,
                f"{prefix}_acc_gender": acc_gender,
                f"{prefix}_mse_age": mse_age,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return combined_loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "train")
        if self.global_step % 50 == 0:
            X, _, _ = batch
            grid = make_grid(X)
            self.logger.experiment.add_image("images", grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")
