import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

from config.core import load_and_validate_config
from src.datasets import UTKFaceDataset
from src.modelling import MultiTaskNet
from src.training import calculate_mean_std, split_dataset

# You are using a CUDA device ('NVIDIA GeForce RTX 3070') that has Tensor Cores.
# To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')`
# which will trade-off precision for performance.

# TODO: figure out tensorboard hparams optimisation?
# TODO: filter data to begin with so equal ages?
if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="my_model_1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")

    config = load_and_validate_config(
        path_to_model_config="config/model_config.yaml",
        path_to_model_training_config="config/model_training_config.yaml",
    )

    # TODO: move this to conf?
    # TODO: readd option which trains on whole dataset
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

    # TODO: add early stopping?

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=200,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_loss_combined"
            ),
            LearningRateMonitor("epoch"),
        ],
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
    )

    trainer.logger._log_graph = True

    # TODO: send config into this
    model = MultiTaskNet()

    trainer.fit(model, train_loader, val_loader)

    val_result = trainer.validate(model, val_loader)
    test_result = trainer.test(model, test_loader)

    print("done")
