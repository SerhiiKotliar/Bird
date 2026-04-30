import os, random, warnings, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import MultiLabelBinarizer
import librosa
from tqdm import tqdm
from transformers import ASTConfig, ASTForAudioClassification

warnings.filterwarnings('ignore')

# ======================== КОНФИГУРАЦИЯ ========================
CFG = {
    "sr": 32000,
    "duration": 5.0,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "f_min": 20,
    "f_max": 16000,
    "batch_size": 16,
    "epochs": 60,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_audio_dir": "datas/train_audio",
    "soundscape_dir": "datas/train_soundscapes",
    "test_soundscapes_dir": "datas/test_soundscapes",
    "label_csv": "datas/train_soundscapes_labels.csv",
    "taxonomy_csv": "datas/taxonomy.csv",
    "sample_submission_path": "datas/sample_submission.csv",
    "model_save_path": "ast_bird_model.pth",
    "submission_save_path": "submission.csv",
    "mlb_path": "mlb.pkl",
    "use_amp": True,
    "clean_samples_per_epoch": 15000,
    "mix_samples_per_epoch": 5000,
    "mixup_alpha": 0.3,
    "label_smoothing": 0.02,
    "num_workers": 1,
}

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])
    torch.backends.cudnn.benchmark = True


def time_to_seconds(t):
    if pd.isna(t): return 0.0
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def mixup_data(x, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam


# ======================== МОДЕЛЬ AST (РАБОЧАЯ ВЕРСИЯ) ========================
class ASTModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        config = ASTConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        config.num_labels = num_classes
        self.backbone = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            config=config,
            ignore_mismatched_sizes=True
        )
        in_features = self.backbone.classifier.dense.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.squeeze(1)  # (B, 128, T)
        x = F.interpolate(x, size=1024, mode='linear', align_corners=False)  # (B, 128, 1024)
        x = x.transpose(1, 2)  # (B, 1024, 128)
        return self.backbone(x).logits


# ======================== ДАТАСЕТ ========================
class MelDataset(Dataset):
    def __init__(self, file_info, augment=False, mlb=None):
        self.file_info = file_info
        self.augment = augment
        self.mlb = mlb
        self.target_len = int(CFG["duration"] * CFG["sr"])
        self.mel_transform = T.MelSpectrogram(
            sample_rate=CFG["sr"], n_fft=CFG["n_fft"],
            hop_length=CFG["hop_length"], n_mels=CFG["n_mels"],
            f_min=CFG["f_min"], f_max=CFG["f_max"]
        )

    def load_segment(self, path, dur):
        if dur > CFG["duration"]:
            start = random.uniform(0, dur - CFG["duration"])
            try:
                y, _ = librosa.load(path, sr=CFG["sr"], mono=True, offset=start, duration=CFG["duration"])
            except:
                y = np.zeros(self.target_len, dtype=np.float32)
        else:
            try:
                y, _ = librosa.load(path, sr=CFG["sr"], mono=True)
            except:
                y = np.zeros(self.target_len, dtype=np.float32)
        return y

    def process_audio(self, y):
        if len(y) < self.target_len:
            y = np.pad(y, (0, self.target_len - len(y)))
        elif len(y) > self.target_len:
            y = y[:self.target_len]

        if self.augment:
            if random.random() < 0.3:
                y = y + np.random.normal(0, 0.005, len(y))
            if random.random() < 0.2:
                rate = random.uniform(0.9, 1.1)
                y = librosa.effects.time_stretch(y, rate=rate)
                y = y[:self.target_len] if len(y) > self.target_len else np.pad(y, (0, self.target_len - len(y)))
            y = y * random.uniform(0.8, 1.2)

        waveform = torch.from_numpy(y).float().unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        return mel


class CleanDataset(MelDataset):
    def __len__(self):
        return CFG["clean_samples_per_epoch"]

    def __getitem__(self, idx):
        i = random.randrange(len(self.file_info))
        path, label, dur = self.file_info[i]
        y = self.load_segment(path, dur)
        mel = self.process_audio(y)

        smooth = CFG["label_smoothing"]
        target = torch.full((len(self.mlb.classes_),), smooth / (len(self.mlb.classes_) - 1))
        for lab in label:
            if lab in self.mlb.classes_:
                target[list(self.mlb.classes_).index(lab)] = 1.0 - smooth
        return mel, target


class MixDataset(MelDataset):
    def __len__(self):
        return CFG["mix_samples_per_epoch"]

    def __getitem__(self, idx):
        n = random.choices([1, 2, 3], weights=[0.15, 0.55, 0.3])[0]
        chosen = random.sample(range(len(self.file_info)), k=n)
        mix = np.zeros(self.target_len, dtype=np.float32)
        labels = set()

        for i in chosen:
            path, lbs, dur = self.file_info[i]
            labels.update(lbs)
            y = self.load_segment(path, dur)
            y = y[:self.target_len] if len(y) > self.target_len else np.pad(y, (0, self.target_len - len(y)))
            mix += y * random.uniform(0.4, 1.0)

        peak = np.abs(mix).max()
        if peak > 0:
            mix = mix / peak * 0.95

        waveform = torch.from_numpy(mix).float().unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)

        smooth = CFG["label_smoothing"]
        target = torch.full((len(self.mlb.classes_),), smooth / (len(self.mlb.classes_) - 1))
        for lab in labels:
            if lab in self.mlb.classes_:
                target[list(self.mlb.classes_).index(lab)] = 1.0 - smooth
        return mel, target


class ValDataset(Dataset):
    def __init__(self, df, mlb):
        self.mlb = mlb
        self.target_len = int(CFG["duration"] * CFG["sr"])
        self.mel_transform = T.MelSpectrogram(
            sample_rate=CFG["sr"], n_fft=CFG["n_fft"],
            hop_length=CFG["hop_length"], n_mels=CFG["n_mels"],
            f_min=CFG["f_min"], f_max=CFG["f_max"]
        )
        self.data = []
        for _, row in df.iterrows():
            if not row["labels_list"]:
                continue
            fpath = os.path.join(CFG["soundscape_dir"], row["filename"])
            if not os.path.exists(fpath):
                continue
            s = time_to_seconds(row["start"])
            e = time_to_seconds(row["end"])
            if e - s < CFG["duration"]:
                continue
            mid = s + (e - s - CFG["duration"]) / 2.0
            self.data.append((fpath, mid, row["labels_list"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fpath, start, labels = self.data[idx]
        try:
            y, _ = librosa.load(fpath, sr=CFG["sr"], mono=True, offset=start, duration=CFG["duration"])
        except:
            y = np.zeros(self.target_len, dtype=np.float32)
        y = y[:self.target_len] if len(y) > self.target_len else np.pad(y, (0, self.target_len - len(y)))

        waveform = torch.from_numpy(y).float().unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)

        target = torch.zeros(len(self.mlb.classes_))
        for lab in labels:
            if lab in self.mlb.classes_:
                target[list(self.mlb.classes_).index(lab)] = 1.0
        return mel, target


# ======================== СБОРКА ========================
def build_datasets():
    print("Индексация файлов...")
    clean_info = []
    all_classes = set()
    for class_dir in tqdm(os.listdir(CFG["train_audio_dir"]), desc="Классы"):
        class_path = os.path.join(CFG["train_audio_dir"], class_dir)
        if not os.path.isdir(class_path): continue
        for fname in os.listdir(class_path):
            if fname.endswith(".ogg"):
                path = os.path.join(class_path, fname)
                try:
                    dur = librosa.get_duration(path=path)
                except:
                    dur = 5.0
                clean_info.append((path, [class_dir], dur))
                all_classes.add(class_dir)

    all_classes = sorted(list(all_classes))
    mlb = MultiLabelBinarizer(classes=all_classes)
    mlb.fit([all_classes])

    train_clean = CleanDataset(clean_info, augment=True, mlb=mlb)
    train_mix = MixDataset(clean_info, augment=False, mlb=mlb)
    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset([train_clean, train_mix])

    df = pd.read_csv(CFG["label_csv"], dtype={"primary_label": str})
    df["labels_list"] = df["primary_label"].apply(lambda x: x.split(";") if pd.notna(x) else [])
    val_ds = ValDataset(df, mlb)

    print(f"✅ Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds, mlb


# ======================== ОБУЧЕНИЕ ========================
def train_epoch(model, loader, optimizer, criterion, device, scaler, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Train {epoch}", unit="batch")
    for data, target in pbar:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if random.random() < 0.5:
            data, y_a, y_b, lam = mixup_data(data, target, CFG["mixup_alpha"])
            mixed = True
        else:
            mixed = False

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=CFG["use_amp"]):
            outputs = model(data)
            if mixed:
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                loss = criterion(outputs, target)

        if CFG["use_amp"]:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        running_loss += loss.item()
        if pbar.n % 10 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    for data, target in tqdm(loader, desc="Val", unit="batch", leave=False):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast(enabled=CFG["use_amp"]):
            outputs = model(data)
            loss = criterion(outputs, target)
        running_loss += loss.item()
        all_preds.append(torch.sigmoid(outputs).cpu())
        all_targets.append(target.cpu())
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    acc = ((preds > 0.5) == targets).float().mean()
    return running_loss / len(loader), acc.item()


if __name__ == "__main__":
    print(f"Устройство: {CFG['device']}")

    train_ds, val_ds, mlb = build_datasets()

    with open(CFG["mlb_path"], "wb") as f:
        pickle.dump(mlb, f)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=1)

    model = ASTModel(num_classes=len(mlb.classes_)).to(CFG["device"])
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler(enabled=CFG["use_amp"])

    best_acc, best_loss = 0.0, float('inf')
    for epoch in range(1, CFG["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CFG["device"], scaler, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, CFG["device"])
        scheduler.step()

        print(f"Epoch {epoch:2d} | TrLoss: {train_loss:.4f} | VaLoss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss = val_acc, val_loss
            torch.save(model.state_dict(), CFG["model_save_path"])
            print(f"  >> Лучшая модель сохранена (acc={val_acc:.4f})")
        torch.cuda.empty_cache()

    print(f"\nГотово. Лучшая точность: {best_acc:.4f}")