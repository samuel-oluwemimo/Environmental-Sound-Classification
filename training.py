import sys
from importlib.metadata import metadata

import modal
import torch
import torchaudio
from model import AudioCNN
import pandas as pd
import torch.nn as nn
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

app = modal.App("audio_cnn")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
         .run_commands([
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ])
         .add_local_python_source("model"))

volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)

class Esc50Dataset(Dataset):
    def __init__(self, root_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split == "train":
            self.metadata = self.metadata[self.metadata["fold"] != 5]
        else:
            self.metadata = self.metadata[self.metadata["fold"] == 5]

        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.root_dir / "audio" / row['filename']
        wav, sample_rate = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(wav, dim=0, keepdim=True)  # Convert to mono if stereo

        if self.transform:
            spectrogram = self.transform(wav)
        else:
            spectrogram = wav

        return spectrogram, row["label"]

@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60 * 60 * 3)
def train():
    esc50_dir = Path("/opt/esc50-data")

    train_transform = nn.Sequential(
     T.MelSpectrogram(
         sample_rate=22050,
         n_fft=1024,
         hop_length=512,
         n_mels=128,
         f_min=0,
         f_max=11025
     ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )

    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB())

    train_set = Esc50Dataset(root_dir=esc50_dir,
                             metadata_file= esc50_dir / "meta" / "esc50.csv",
                             split="train", transform=train_transform)
    val_set = Esc50Dataset(root_dir=esc50_dir,
                             metadata_file=esc50_dir / "meta" / "esc50.csv",
                             split="val", transform=val_transform)

    print(f"..........Training set length: {len(train_set)}")
    print(f"..........Validation set length: {len(val_set)}")

    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = AudioCNN(num_classes=len(train_set.classes))
    model.to(device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1
    )


    print("Training...")

@app.local_entrypoint()
def main():
    train.remote()


