import pandas as pd
import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader


from config.core import load_and_validate_config
from src.datasets import get_dataloaders, UTKFaceDataset
from src.modelling import MultiTaskNet, MultiTaskNetWorking
from src.training import get_age_class_weights, split_dataset, calculate_mean_std
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

# You are using a CUDA device ('NVIDIA GeForce RTX 3070') that has Tensor Cores.
# To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')`
# which will trade-off precision for performance.

# TODO: logger
 
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

    train_dataset = UTKFaceDataset(df_train, transforms=ToTensor())
    DATA_MEANS, DATA_STD = calculate_mean_std(train_dataset)
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    # TODO: reimplement this back into transforms func
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(DATA_MEANS, DATA_STD)]
    )
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(DATA_MEANS, DATA_STD),
        ]
    )
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = UTKFaceDataset(df=df_train, transforms=train_transform)
    val_dataset = UTKFaceDataset(df=df_val, transforms=test_transform)
    test_dataset = UTKFaceDataset(df=df_test, transforms=test_transform)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0
    )

    # training
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=200,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_acc"
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
        ],
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
    )

    # TODO: figure out tensorboard computation graph
    trainer.logger._log_graph = True

    # TODO: send config into this
    # TODO: allow for loading of checkpointed model
    model = MultiTaskNetWorking()

    trainer.fit(model, train_loader, val_loader)

    val_result = trainer.validate(model, val_loader)
    test_result = trainer.test(model, test_loader)

    print("done")
