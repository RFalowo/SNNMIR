from spikingjelly.activation_based import neuron, functional, surrogate, layer
import spikingjelly.clock_driven as cd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchaudio
import pytorch_lightning as pl
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
#transfrom = T.MFCC(n_mfcc=96, melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 96, 'center': False})
# Adjust this transform according to your needs. For beat detection, Mel Spectrogram can be very useful.

def get_args():
    parser = ArgumentParser(description="SNN Beat Detection")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer to use; 'adam' or 'sgd'")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--model", type=str, default="RSNN", choices=["plain", "stateful", "feedback"], help="Model to use; 'plain', 'stateful', or 'feedback'")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging (default: False)")
    parser.add_argument("--encoding", type=str, default="lyon", choices=["lyon"], help="Audio encoding method (default: 'lyon')")

    return parser.parse_args()


class RSNNBeatDetection(pl.LightningModule):
    def __init__(self, optimizer_name="adam", learning_rate=0.001, transform=None):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        print(f"Transform argument: {transform}")
        self.transform = transform
        # Defining the RSNN architecture. Adjust sizes and parameters according to your dataset and task.
        self.S1 = nn.Sequential(
            layer.Linear(96, 256, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True, v_threshold=0.4, v_reset=0.2),
        )
        self.rsnn_block = layer.LinearRecurrentContainer(
                neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True, v_threshold=0.4, v_reset=0.2),
                in_features=256,
                out_features=128,
                step_mode='s',            )
             # Another spiking neuron layer
        # self.scnn = layer.Conv1d(1,1, 3)
        # self.AP = layer.AdaptiveAvgPool1d(2)
        self.S2 = nn.Sequential(
            layer.Linear(256, 128, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True, v_threshold=0.4, v_reset=0.2),
        )
        self.S3 = nn.Sequential(
            layer.Linear(128, 64, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True, v_threshold=0.4, v_reset=0.2),
        )
        self.output = nn.Sequential(
            layer.Linear(64, 1, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True, v_threshold=0.4, v_reset=0.2),
        )

       #final output spiking layer with one output neuron that will spike when the network predicts a beat


    def forward(self, x : torch.Tensor):
        # Assuming x is initially [batch_size, channels, MFCC features, time steps]
        # Remove the channels dimension if it's always 1 and not needed
        x = x.squeeze(1)  # Now x is [batch_size, MFCC features, time steps]

        # Permute to put time steps first for iteration
        x = x.permute(2, 0, 1)  # Now x is [time steps, batch_size, MFCC features]
        # print('input:',float(x.sum()), x.shape,x.min(),x.max())
        # x = torch.rand(x.shape).to(x.device)
        v = x.cpu().squeeze().detach().numpy()
        x0 = v
        x = self.S1(x)  # First spiking layer
        # print('S1: ',float(x.sum()), x.shape,x.min(),x.max())
        v = x.cpu().squeeze().detach().numpy()
        x = self.rsnn_block(x)  # RSNN block
        # print ('s2: ',float(x.sum()), x.shape,x.min(),x.max())
        v = x.cpu().squeeze().detach().numpy()
        x = self.S2(x)
        # print ('s3: ',float(x.sum()), x.shape,x.min(),x.max())
        v = x.cpu().squeeze().detach().numpy()
        x = self.S3(x)
        # print ('s4: ',float(x.sum()), x.shape,x.min(),x.max())
        v = x.cpu().squeeze().detach().numpy()
        x = self.output(x)
        v = x.cpu().squeeze().detach().numpy()

        # print (float(x.sum()))

        x = x.squeeze(2) # Remove the last dimension
        # reset neuron states
        functional.reset_net(self.S1)
        functional.reset_net(self.rsnn_block)
        functional.reset_net(self.S2)
        functional.reset_net(self.S3)
        functional.reset_net(self.output)
        return x

    def log_beats_to_wandb(self, predicted_spikes, y, sample_index=0, step=None):
        # Ensure tensors are on CPU for processing
        print(f"Before .cpu() - Predicted Spikes: {predicted_spikes.sum()}, True Beats: {y.sum()}")

        predicted_spikes = predicted_spikes.cpu().detach().numpy()
        y = y.cpu().detach().numpy()

        plt.figure(figsize=(20, 5))
        # Extract beat times
        for beat_time in range(len(y[sample_index])):
            if y[sample_index, beat_time].item() == 1:
                plt.axvline(x=beat_time, color='g', linestyle='-', label='True Beat' if beat_time == 0 else "")
        for beat_time in range(len(predicted_spikes[sample_index])):
            if predicted_spikes[sample_index, beat_time].item() == 1:
                plt.axvline(x=beat_time, color='r', linestyle='--', label='Predicted Beat' if beat_time == 0 else "")
        print(
            f"True Beats: {y[sample_index].sum().item()}, Predicted Beats: {predicted_spikes[sample_index].sum().item()}")

        plt.title(f'Predicted Beats vs. True Beats (Sample {sample_index})')
        plt.xlabel('Time Step')
        plt.yticks([])
        plt.legend(loc="upper right")

        # Use BytesIO buffer to save plot as an image in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image = Image.open(buf)

        # Log the image to wandb
        # wandb.log({"Beat Detection Plot": [wandb.Image(image, caption=f"Beat Detection (Sample {sample_index})")]},
        #           step=step)


    def training_step(self, batch, batch_idx):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            y = y.permute(1, 0)  # Flatten the target tensor
            y_hat = self.forward(x)  # Flatten the output tensor
            predicted_spikes = y_hat  # Predicted spikes are the output tensor values greater than 0
            if predicted_spikes.cpu().detach().numpy().sum() > 0:
                print(f"Predicted Spikes: {predicted_spikes.cpu().detach().numpy().sum()}")
            # if batch_idx%20 == 3:  #limit the number of plots
            #     # out_vs_gt = wandb.Image(torch.cat([y_hat, y], dim=1).cpu().detach().numpy())
            #     # self.log('out_vs_gt', out_vs_gt)
            #     self.log_beats_to_wandb(predicted_spikes, y, sample_index=batch_idx, step=batch_idx)

            loss = F.binary_cross_entropy_with_logits(y_hat, y)
            self.log('train_loss', loss)
            return loss

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer

    def train_dataloader(self):
        # Make sure to point 'dataset_path' to your GTZAN dataset location
        dataset = GTZANDataset(audio_dir='data/Data/genres_original', beat_dir='data/gtzan_tempo_beat/beats', transform=self.transform, normalization_file='normalizationMM_values.txt',gaussian_width=2)
        return DataLoader(dataset, batch_size=1, shuffle=True)

    def val_dataloader(self):
        dataset = GTZANDataset(audio_dir='data/Data/genres_original', beat_dir='data/gtzan_tempo_beat/beats', transform=self.transform, normalization_file='normalizationMM_values.txt',gaussian_width=2)
        return DataLoader(dataset, batch_size=1)

    def test_dataloader(self):
        dataset = GTZANDataset(audio_dir='data/Data/genres_original', beat_dir='data/gtzan_tempo_beat/beats', transform=self.transform, normalization_file='normalizationMM_values.txt',gaussian_width=2)
        return DataLoader(dataset, batch_size=1)





def main():
    args = get_args()

    dataset_args = {
        'audio_dir': 'data/Data/genres_original',
        'beat_dir': 'data/gtzan_tempo_beat/beats',
        'normalization_file': 'normalizationMM_values.txt',
        'gaussian_width': 2
    }

    if args.encoding == "lyon":
        transform = lyon_cochleagram
    else:
        raise ValueError(f"Unsupported encoding method: {args.encoding}")
    if args.model == "RSNN":
        model = RSNNBeatDetection(optimizer_name=args.optimizer, learning_rate=args.learning_rate, transform=transform)

    train_dataset = GTZANDataset(**dataset_args)
    val_dataset = GTZANDataset(**dataset_args)
    test_dataset = GTZANDataset(**dataset_args)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    logger = WandbLogger(project="SNN_Beat_Detection", log_model="all") if args.use_wandb else False
    trainer = pl.Trainer(max_epochs=args.epochs, logger=logger, callbacks=[ModelCheckpoint(monitor="train_loss")])
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

if __name__ == '__main__':
    main()
