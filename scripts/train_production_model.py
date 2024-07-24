import argparse

from src.training import train_production_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--early_stopping_patience", type=int, default=30)
    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=50)

    # add extra hparams which we have peformed our hyperparameter tuning on
    parser.add_argument("--learning_rate", type=float, default=0.00032849)
    parser.add_argument("--age_hidden_head_dim", type=int, default=66)
    parser.add_argument("--gender_hidden_head_dim", type=int, default=27)
    parser.add_argument("--resnet_model", type=str, default="resnet34")

    parser.add_argument("--scheduler_milestones", type=int, nargs="+", default=[24, 36, 48])

    args = parser.parse_args()

    train_production_model(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
        learning_rate=args.learning_rate,
        age_hidden_head_dim=args.age_hidden_head_dim,
        gender_hidden_head_dim=args.gender_hidden_head_dim,
        resnet_model=args.resnet_model,
        scheduler_patience=args.scheduler_patience,
        scheduler_milestones=args.scheduler_milestones,
    )
