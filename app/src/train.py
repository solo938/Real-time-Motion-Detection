import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from app.src.lstm import ActionClassificationLSTM, PoseDataModule, WINDOW_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",     default="app/datasets/retrain/")
    parser.add_argument("--out",           default="app/models/saved_model_v2.ckpt")
    parser.add_argument("--epochs",        type=int,   default=200)
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--hidden_dim",    type=int,   default=50)
    args = parser.parse_args()

    pl.seed_everything(42)

    # ensure data_root ends with /
    data_root = args.data_root
    if not data_root.endswith("/"):
        data_root += "/"

    print(f"Data root : {data_root}")
    print(f"Output    : {args.out}")
    print(f"Epochs    : {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device    : {'cuda' if torch.cuda.is_available() else 'cpu'}")

    model = ActionClassificationLSTM(
        input_features=34,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
    )

    data_module = PoseDataModule(
        data_root=data_root,
        batch_size=args.batch_size,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath="app/models/",
        filename="retrained_{epoch:02d}_{val_loss:.4f}",
        save_top_k=1,
        monitor="avg_val_loss",
        mode="min",
    )
    early_stop_cb = EarlyStopping(
        monitor="train_loss",
        patience=20,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop_cb, checkpoint_cb, lr_monitor],
        enable_progress_bar=True,
    )

    trainer.fit(model, data_module)

    # Save final checkpoint to the specified output path
    trainer.save_checkpoint(args.out)
    print(f"\nSaved final model to: {args.out}")
    print("Update app/__init__.py to load this new checkpoint.")


if __name__ == "__main__":
    main()