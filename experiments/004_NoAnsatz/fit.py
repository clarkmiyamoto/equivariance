import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from data.manager import get_DataLoader
from model.selector import get_model
from train.optimizer import get_optim
from train.scheduler import get_scheduler
from train.tracker import get_gradient_norm, get_update_norm

import wandb
import json
import argparse


def load_config():
    parser = argparse.ArgumentParser(description="Config JSON in String Form")
    parser.add_argument(
        "config_json",
        type=str,
        help="JSON string with hyperparameters (e.g. '{\"epochs\":10, ...}')"
    )
    args = parser.parse_args()
    config = json.loads(args.config_json)

    model_name = config["model"].pop("name", None) if "model" in config else 'Not Specified'

    return config, model_name



if __name__ == "__main__":
    ##################
    # --- Config --- #
    ##################

    config_dict, model_name = load_config()
    wandb.init(
        project="diffeo",
        name=model_name,
        config=config_dict,
    )
    config = wandb.config

    ################
    # --- Code --- #
    ################
    # ---- Initalization ----
    train_loader, val_loader = get_DataLoader(config.batch_size, config.dataset)

    model = get_model(str(wandb.run.name))(**config.model)
    model.to(device)

    optimizer = get_optim(
        model=model,
        **config.optimizer
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        **config.scheduler
    )
    criterion = nn.MSELoss()

    # ---- Tracking Parameters ----
    

    # ---- Run ----
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

        

        # ---- Logging ----
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        
        grad_norm = get_gradient_norm(model)
        
        prev_params = [param.clone().detach() for param in model.parameters()]
        update_norm = get_update_norm(model=model, prev_params=prev_params)
        
        # ---- Step ----
        scheduler.step(avg_val_loss)
        
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch": epoch,
            "current_lr": current_lr,
            "grad_norm": grad_norm,
            "update_norm": update_norm,
        })
