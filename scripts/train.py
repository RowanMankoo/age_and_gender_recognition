import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping

from src.core import load_and_validate_config
from src.data import get_dataloaders
from src.modelling import MultiTaskNet
from src.training import get_age_class_weights, train_test_split_celeb_ids

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_and_validate_config(
        path_to_metadata_cleaning_config="config/metadata_cleaning_config.yaml",
        path_to_model_config="config/model_config.yaml",
        path_to_model_training_config="config/model_training_config.yaml",
    )

    df_metadata = pd.read_csv("Data/Metadata/metadata_cleaned.csv")
    df_train, df_test = train_test_split_celeb_ids(df_metadata, test_size=0.2, random_state=42)

    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(df_train, df_test, config)

    age_class_weights = get_age_class_weights(df_train, config.model_config.age_labels_to_bins).to(device)

    model = MultiTaskNet(config, age_class_weights)

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=False, mode="min")

    # training
    trainer = pl.Trainer(max_epochs=5, gpus=1, callbacks=[early_stop_callback])
    trainer.fit(model, train_dataloader, test_dataloader)

    class_weights = get_age_class_weights(df_train).to(device)
    age_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    gender_loss_fn = nn.BCELoss()

    print("done")
