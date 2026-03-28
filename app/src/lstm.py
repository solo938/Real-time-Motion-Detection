import numpy as np
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

WINDOW_SIZE = 16   
NUM_CLASSES = 6
openpose_to_detectron_mapping = []


class PoseDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PoseDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size):
        super().__init__()
        self.data_root    = data_root
        self.batch_size   = batch_size
        self.X_train_path = self.data_root + "X_train.txt"
        self.X_test_path  = self.data_root + "X_test.txt"
        self.Y_train_path = self.data_root + "y_train.txt"
        self.Y_test_path  = self.data_root + "y_test.txt"

    def load_X(self, X_path):
        file = open(X_path, 'r')
        X = np.array(
            [list(map(float, row.strip().split(','))) for row in file if row.strip()],
            dtype=np.float32
        )
        file.close()
        blocks = int(len(X) / WINDOW_SIZE)
        X_ = np.array(np.split(X, blocks))
        return X_

    def load_y(self, y_path):
        file = open(y_path, 'r')
        y = np.array(
            [int(row.strip()) for row in file if row.strip()],
            dtype=np.int32
        )
        file.close()
        blocks = int(len(y) / WINDOW_SIZE)
        y = np.array(np.split(y, blocks))
        # take label of first frame in each window
        y = y[:, 0]
        return y

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        X_train = self.load_X(self.X_train_path)
        X_test  = self.load_X(self.X_test_path)
        y_train = self.load_y(self.Y_train_path)
        y_test  = self.load_y(self.Y_test_path)
        self.train_dataset = PoseDataset(X_train, y_train)
        self.eval_dataset  = PoseDataset(X_test,  y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False)


class ActionClassificationLSTM(pl.LightningModule):

    def __init__(self, input_features, hidden_dim, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.lstm   = nn.LSTM(input_features, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, NUM_CLASSES)

        # accumulators for epoch-end averaging (PL v2 style)
        self._train_losses = []
        self._train_accs   = []
        self._val_losses   = []
        self._val_accs     = []

    def forward(self, X):
        lstm_out, (ht, ct) = self.lstm(X)
        return self.linear(ht[-1])

    def training_step(self, batch, batch_idx):
        x, y   = batch
        y      = torch.squeeze(y).long()
        y_pred = self(x)
        loss   = F.cross_entropy(y_pred, y)
        prob   = F.softmax(y_pred, dim=1)
        pred   = prob.data.max(dim=1)[1]
        acc    = torchmetrics.functional.accuracy(
            pred, y, task='multiclass', num_classes=NUM_CLASSES)
        self.log('batch_train_loss', loss, prog_bar=True)
        self.log('batch_train_acc',  acc,  prog_bar=True)
        self._train_losses.append(loss.detach())
        self._train_accs.append(acc.detach())
        return loss

    def on_train_epoch_end(self):
        if self._train_losses:
            avg_loss = torch.stack(self._train_losses).mean()
            avg_acc  = torch.stack(self._train_accs).mean()
            self.log('train_loss', avg_loss, prog_bar=True)
            self.log('train_acc',  avg_acc,  prog_bar=True)
            self._train_losses.clear()
            self._train_accs.clear()

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        y      = torch.squeeze(y).long()
        y_pred = self(x)
        loss   = F.cross_entropy(y_pred, y)
        prob   = F.softmax(y_pred, dim=1)
        pred   = prob.data.max(dim=1)[1]
        acc    = torchmetrics.functional.accuracy(
            pred, y, task='multiclass', num_classes=NUM_CLASSES)
        self.log('batch_val_loss', loss, prog_bar=True)
        self.log('batch_val_acc',  acc,  prog_bar=True)
        self._val_losses.append(loss.detach())
        self._val_accs.append(acc.detach())

    def on_validation_epoch_end(self):
        if self._val_losses:
            avg_loss = torch.stack(self._val_losses).mean()
            avg_acc  = torch.stack(self._val_accs).mean()
            self.log('avg_val_loss', avg_loss, prog_bar=True)
            self.log('avg_val_acc',  avg_acc,  prog_bar=True)
            self._val_losses.clear()
            self._val_accs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-15)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "frequency": 1,
                "monitor":   "avg_val_loss",
            }
        }





        
