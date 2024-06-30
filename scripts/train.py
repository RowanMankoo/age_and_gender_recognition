import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping

from src.core import load_and_validate_config
from src.data import get_dataloaders
from src.modelling import MultiTaskNet
from src.training import get_age_class_weights, split_dataset

# You are using a CUDA device ('NVIDIA GeForce RTX 3070') that has Tensor Cores.
# To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')`
# which will trade-off precision for performance.

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")

    config = load_and_validate_config(
        path_to_model_config="config/model_config.yaml",
        path_to_model_training_config="config/model_training_config.yaml",
    )

    # TODO: move this to conf?
    df_metadata = pd.read_csv("Data/metadata.csv")
    df_train, df_val, df_test = split_dataset(df_metadata, train_size=0.6, val_size=0.2)

    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
        df_train, df_test, config
    )

    age_class_weights = get_age_class_weights(
        df_train, config.model_config.age_labels_to_bins
    ).to(device)

    model = MultiTaskNet(config, age_class_weights)

    early_stop_callback = EarlyStopping(
        monitor="val_loss_total", patience=200, verbose=False, mode="min"
    )

    # training
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",
        callbacks=[early_stop_callback],
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_dataloader, test_dataloader)

    trainer.test(model, test_dataloader)

    # TODO: test preds on no target data
    # preds = trainer.predict(model, test_dataloader)

    print("done")
