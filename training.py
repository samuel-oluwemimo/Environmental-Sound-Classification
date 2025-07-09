import modal
import torch
import torchaudio
import numpy as np
from model import AudioCNN
import pandas as pd
import torch.nn as nn
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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

def mixup_data(x,y):
    """Apply mixup augmentation to the input data."""
    alpha = 0.2
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    else:
        return x, y, y, 1.0

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute the mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60 * 60 * 3)
def train():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'/models/tensorboards_logs/{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)

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

    best_accuracy = 0.0

    print("......Starting Training......")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            if np.random.random() > 0.7:
                data, target_a, target_b, lam = mixup_data(data, target)
                data, target_a, target_b = data.to(device), target_a.to(device), target_b.to(device)
                output = model(data)
                loss = mixup_criterion(
                    criterion,
                    output,
                    target_a,
                    target_b,
                    lam
                )
            else:
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_running_loss = running_loss / len(train_dataloader)
        writer.add_scalar('Loss/train', avg_running_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_dataloader)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', accuracy, epoch)

        print(f"Epoch {epoch+1} Loss: {avg_running_loss:.4f},"
              f" Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({'model_state_dict': model.state_dict(),
                        'accuracy': accuracy,
                        'epoch': epoch,
                        'classes': train_set.classes,
                        }, "/models/model.pth")
            print(f'New best model saved: {accuracy:.2f}%')
    writer.close()
    print(f"Training completed: Best accuracy: {best_accuracy:.2f}%")



@app.local_entrypoint()
def main():
    train.remote()


