import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from data.manager import get_DataLoader
from train.optimizer import get_optim

import wandb


class Ginv(nn.Module):
  def __init__(self, dropout_rate):
    super(Ginv, self).__init__()

    # Model
    self.encoder = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),

        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),

        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),

        nn.Linear(512, 512),
    )

  def forward(self, x):
    x = self.encoder(x)
    return x


if __name__ == "__main__":
    ###
    # Config
    ###

    wandb.init(
        project="diffeo",
        name=f"MLP",
        config={
            "dropout_rate": 0.2,
            "batch_size": 1,
            "epochs": 1500,
            "optimizer" : {
                "name": "Adam",
                "lr": 1e-5,
            },
            "dataset": "resnet18_imagenet1k_train",
        },
    )
    config = wandb.config

    ### 
    # Code
    ###
    train_loader, val_loader = get_DataLoader(config.batch_size, config.dataset)

    model = Ginv(dropout_rate=config.dropout_rate)
    model.to(device)

    optimizer = get_optim(
        model=model,
        **config.optimizer_options
    )
    criterion = nn.MSELoss()

    for epoch in range(config.epochs):
        # ---- Training ----
        model.train()
        total_train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            # Move to GPU if available
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Average training loss over the epoch
        avg_train_loss = total_train_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)

                predictions = model(batch_X)
                loss = criterion(predictions, batch_Y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # ---- Logging ----
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch": epoch
        })