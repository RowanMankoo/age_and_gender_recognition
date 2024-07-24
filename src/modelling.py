import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torchvision.utils import make_grid


class HeadBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


class MultiTaskNet(L.LightningModule):
    def __init__(self, production_model: bool, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.production_model = production_model

        self.base_model, num_ftrs = self.create_base_model_and_get_num_features()
        self.gender_head = HeadBlock(num_ftrs, self.hparams["gender_hidden_head_dim"], 2)
        self.age_head = HeadBlock(num_ftrs, self.hparams["age_hidden_head_dim"], 1)
        self.gender_loss_component_weight = 0.99
        self.age_loss_component_weight = 0.01

        self.loss_module_gender = nn.CrossEntropyLoss()
        self.loss_module_age = nn.MSELoss()

        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.rand(1, 3, 224, 224)

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/loss_gender": 0,
                "hp/loss_age": 0,
                "hp/loss_combined": 0,
            },
        )

    def create_base_model_and_get_num_features(self):
        base_model = getattr(torchvision.models, self.hparams.resnet_model)(pretrained=not self.production_model)

        num_ftrs = base_model.fc.in_features  # Get the number of input features of the final layer
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

        optimizer = optim.AdamW(self.parameters(), lr=self.hparams["learning_rate"], weight_decay=0.0001)

        if self.production_model:
            scheduler = MultiStepLR(
                optimizer,
                milestones=self.hparams["scheduler_milestones"],
                gamma=0.2,
            )
        else:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=self.hparams["scheduler_patience"],
                threshold=0.001,
                threshold_mode="rel",
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss_combined",
                "frequency": 1,
            },
        }

    def _shared_step(self, batch, batch_idx, prefix):
        X, y_age, y_gender = batch
        out_gender, out_age = self.forward(X)

        loss_gender = self.loss_module_gender(out_gender, y_gender)
        loss_age = self.loss_module_age(out_age.view(-1), y_age)
        loss_combined = loss_gender * self.gender_loss_component_weight + loss_age * self.age_loss_component_weight

        preds_gender, _ = self._parse_output_to_preds(out_gender=out_gender, out_age=out_age)

        acc_gender = (y_gender == preds_gender).float().mean()
        mse_age = F.mse_loss(out_age.view(-1), y_age)

        self.log_dict(
            {
                f"{prefix}/loss_gender": loss_gender,
                f"{prefix}/loss_age": loss_age,
                f"{prefix}/loss_combined": loss_combined,
                f"{prefix}/acc_gender": acc_gender,
                f"{prefix}/mse_age": mse_age,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        if prefix == "test":

            self.log_dict(
                {
                    "hp/loss_gender": loss_gender,
                    "hp/loss_age": loss_age,
                    "hp/loss_combined": loss_combined,
                }
            )

        return loss_combined

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "train")
        if self.global_step % 500 == 0:
            X, _, _ = batch
            grid = make_grid(X)
            self.logger.experiment.add_image("images", grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")
