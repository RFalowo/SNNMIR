import optuna
from argparse import ArgumentParser
from encoders import lyon_cochleagram
from PIL import Image
import io
import os
import wandb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from GTZANBEAT import GTZANDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from spikingjelly_beat_detection import RSNNBeatDetection

def get_args():
    parser = ArgumentParser(description="SNN Beat Detection")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer to use; 'adam' or 'sgd'")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--model", type=str, default="RSNN", choices=["plain", "stateful", "feedback"], help="Model to use; 'plain', 'stateful', or 'feedback'")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging (default: False)")
    parser.add_argument("--encoding", type=str, default="lyon", choices=["lyon"], help="Audio encoding method (default: 'lyon')")

    return parser.parse_args()


def objective(trial):
    # Define the hyperparameters to tune
    v_threshold = trial.suggest_uniform('v_threshold', 0.1, 1.0)
    tau = trial.suggest_uniform('tau', 1.0, 10.0)

    # Create the model with the current hyperparameters
    model = RSNNBeatDetection(optimizer_name=args.optimizer, learning_rate=args.learning_rate, transform=transform)
    model.S1[1].v_threshold = v_threshold
    model.S1[1].tau = tau

    # Train the model and calculate the validation loss
    trainer = pl.Trainer(max_epochs=args.epochs)
    trainer.fit(model, train_loader, val_loader)
    val_loss = trainer.callback_metrics['val_loss'].item()

    return val_loss

# Create the Optuna study and start the optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters
print(study.best_params)