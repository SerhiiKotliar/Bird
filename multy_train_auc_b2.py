import os, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import librosa
from tqdm import tqdm
import timm

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
    "batch_size": 48,
    "epochs": 40,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "drop_rate": 0.3,
    "drop_path_rate": 0.1,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_audio_dir": "datas/train_audio",
    "soundscape_dir": "datas/train_soundscapes",
    "label_csv": "datas/train_soundscapes_labels.csv",
    "model_save_path": "bird_model_best_auc_b2.pth",
    "num_workers": 2,
    "prefetch_factor": 2,
    "use_amp": True,
    "clean_samples_per_epoch": 20000,
    "soundscape_samples_per_epoch": 15000,
    "mixup_alpha": 0.0,
    "label_smoothing": 0.05,
    "model_name": "efficientnet_b0",
    "use_3channel": True,
    "lr_schedule": "cosine",
    "warmup_epochs": 0,
    "early_stop_patience": 5,
}

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])
    torch.backends.cudnn.benchmark = True


def time_to_seconds(t):
    if pd.isna(t):
        return 0.0
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# ================== DATASET ДЛЯ ЗВУКОВЫХ ЛАНДШАФТОВ ==================
class SoundscapeDataset(Dataset):
    def __init__(self, df, mlb, augment=True):
        self.mlb = mlb
        self.augment = augment
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
            if e - s >= CFG["duration"]:
                self.data.append((fpath, s, e, row["labels_list"]))
        print(f"  Soundscape dataset: {len(self.data)} valid intervals")

    def __len__(self):
        return min(CFG["soundscape_samples_per_epoch"], len(self.data) * 10)

    def __getitem__(self, idx):
        fpath, start, end, labels = random.choice(self.data)
        max_start = max(start, end - CFG["duration"])
        offset = random.uniform(start, max_start)
        try:
            y, _ = librosa.load(fpath, sr=CFG["sr"], mono=True, offset=offset, duration=CFG["duration"])
        except:
            y = np.zeros(self.target_len, dtype=np.float32)
        if len(y) < self.target_len:
            y = np.pad(y, (0, self.target_len - len(y)))
        else:
            y = y[:self.target_len]
        if self.augment and random.random() < 0.3:
            y = y * random.uniform(0.8, 1.2)
        waveform = torch.from_numpy(y).float().unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        if CFG["use_3channel"]:
            mel = mel.repeat(3, 1, 1)
        target = torch.zeros(len(self.mlb.classes_))
        for lab in labels:
            if lab in self.mlb.classes_:
                target[list(self.mlb.classes_).index(lab)] = 1.0
        return mel, target


# ================== ДАТАСЕТЫ ДЛЯ ЧИСТЫХ ЗАПИСЕЙ ==================
class CleanDataset(Dataset):
    def __init__(self, file_info, mlb, augment=True):
        self.file_info = file_info
        self.mlb = mlb
        self.augment = augment
        self.target_len = int(CFG["duration"] * CFG["sr"])
        self.mel_transform = T.MelSpectrogram(
            sample_rate=CFG["sr"], n_fft=CFG["n_fft"],
            hop_length=CFG["hop_length"], n_mels=CFG["n_mels"],
            f_min=CFG["f_min"], f_max=CFG["f_max"]
        )

    def __len__(self):
        return CFG["clean_samples_per_epoch"]

    def __getitem__(self, idx):
        path, label, dur = random.choice(self.file_info)
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
        if len(y) < self.target_len:
            y = np.pad(y, (0, self.target_len - len(y)))
        else:
            y = y[:self.target_len]
        if self.augment and random.random() < 0.5:
            y = y * random.uniform(0.7, 1.3)
        waveform = torch.from_numpy(y).float().unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        if CFG["use_3channel"]:
            mel = mel.repeat(3, 1, 1)
        target = torch.zeros(len(self.mlb.classes_))
        if label[0] in self.mlb.classes_:
            target[list(self.mlb.classes_).index(label[0])] = 1.0
        return mel, target


# ================== ВАЛИДАЦИОННЫЙ ДАТАСЕТ ==================
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
            if e - s >= CFG["duration"]:
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
        if len(y) < self.target_len:
            y = np.pad(y, (0, self.target_len - len(y)))
        else:
            y = y[:self.target_len]
        waveform = torch.from_numpy(y).float().unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        if CFG["use_3channel"]:
            mel = mel.repeat(3, 1, 1)
        target = torch.zeros(len(self.mlb.classes_))
        for lab in labels:
            if lab in self.mlb.classes_:
                target[list(self.mlb.classes_).index(lab)] = 1.0
        return mel, target


# ================== ПОСТРОЕНИЕ ДАТАСЕТОВ (с pos_weight) ==================
def build_datasets():
    print("📂 Indexing files...")
    clean_info = []
    all_classes = set()

    for class_dir in tqdm(os.listdir(CFG["train_audio_dir"]), desc="Classes"):
        class_path = os.path.join(CFG["train_audio_dir"], class_dir)
        if not os.path.isdir(class_path):
            continue
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
    print(f"✅ Найдено {len(clean_info)} чистых записей, {len(all_classes)} классов")

    mlb = MultiLabelBinarizer(classes=all_classes)
    mlb.fit([all_classes])

    # Разделение soundscape файлов
    df = pd.read_csv(CFG["label_csv"], dtype={"primary_label": str})
    df["labels_list"] = df["primary_label"].apply(lambda x: x.split(";") if pd.notna(x) else [])
    df = df[df["labels_list"].apply(len) > 0]

    unique_files = df['filename'].unique()
    train_files, val_files = train_test_split(unique_files, test_size=0.25, random_state=CFG["seed"])
    df_train = df[df['filename'].isin(train_files)]
    df_val = df[df['filename'].isin(val_files)]

    print(f"✅ Звуковые ландшафты: train файлов={len(train_files)}, val файлов={len(val_files)}")
    print(f"   Train интервалов: {len(df_train)}, Val интервалов: {len(df_val)}")

    train_clean = CleanDataset(clean_info, mlb, augment=True)
    train_soundscape = SoundscapeDataset(df_train, mlb, augment=True)
    val_ds = ValDataset(df_val, mlb)

    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset([train_clean, train_soundscape])

    print(f"✅ Train: {len(train_ds)} (clean: {len(train_clean)} / soundscape: {len(train_soundscape)})")
    print(f"✅ Val: {len(val_ds)}")

    # --- Подсчёт pos_weight по ВСЕЙ тренировочной выборке (clean + soundscape) ---
    class_counts = np.zeros(len(mlb.classes_))
    # 1) soundscape интервалы
    for _, row in df_train.iterrows():
        for lab in row["labels_list"]:
            if lab in mlb.classes_:
                class_counts[list(mlb.classes_).index(lab)] += 1
    # 2) чистые записи
    for path, label, dur in clean_info:
        if label[0] in mlb.classes_:
            class_counts[list(mlb.classes_).index(label[0])] += 1

    total_samples = len(df_train) + len(clean_info)
    # Вычисляем pos_weight, но ограничиваем максимальное значение 100
    pos_weight = (total_samples - class_counts) / (class_counts + 1)  # +1 чтобы избежать деления на 0
    pos_weight = np.clip(pos_weight, 1.0, 100.0)  # важное ограничение!
    pos_weight = torch.from_numpy(pos_weight).float().to(CFG["device"])

    print(f"Pos_weight: min={pos_weight.min().item():.2f}, max={pos_weight.max().item():.2f}")
    return train_ds, val_ds, mlb, pos_weight


# ================== МОДЕЛЬ ==================
def build_model(num_classes):
    model_name = CFG["model_name"]
    in_chans = 3 if CFG["use_3channel"] else 1

    if "efficientnet" in model_name:
        model = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_rate=CFG["drop_rate"],
            drop_path_rate=CFG["drop_path_rate"]
        )
    elif "convnext" in model_name:
        model = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_path_rate=0.2
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(CFG["device"])


# ================== Cosine annealing without warmup ==================
class CosineAnnealingScheduler:
    def __init__(self, optimizer, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        progress = (self.current_epoch - 1) / (self.total_epochs - 1)
        lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress)) / 2
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


# ================== ТРЕНИРОВКА ==================
def train_epoch(model, loader, optimizer, criterion, device, scaler, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Train {epoch}", unit="batch")

    for data, target in pbar:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if target.sum() == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=CFG["use_amp"]):
            outputs = model(data)
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
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader)


# ================== ВАЛИДАЦИЯ ==================
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

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    print(f"  Preds mean: {preds.mean():.4f}, Targets mean: {targets.mean():.4f}")

    auc_per_class = []
    for i in range(targets.shape[1]):
        if np.sum(targets[:, i]) > 0:
            try:
                auc = roc_auc_score(targets[:, i], preds[:, i])
                auc_per_class.append(auc)
            except:
                pass
    macro_auc = np.mean(auc_per_class) if auc_per_class else 0.0
    all_preds_flat = preds.ravel()
    all_targets_flat = targets.ravel()
    micro_auc = roc_auc_score(all_targets_flat, all_preds_flat)
    return running_loss / len(loader), macro_auc, micro_auc


# ================== ОСНОВНАЯ ФУНКЦИЯ ==================
if __name__ == "__main__":
    print(f"Device: {CFG['device']}, model: {CFG['model_name']}")

    train_ds, val_ds, mlb, pos_weight = build_datasets()
    print(f"Number of classes: {len(mlb.classes_)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=True
    )

    model = build_model(len(mlb.classes_))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("✅ Using BCEWithLogitsLoss with clipped pos_weight (max=100)")

    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scaler = GradScaler(enabled=CFG["use_amp"])

    if CFG["lr_schedule"] == "cosine":
        scheduler = CosineAnnealingScheduler(optimizer, total_epochs=CFG["epochs"], base_lr=CFG["lr"], min_lr=1e-6)
    else:
        scheduler = None

    best_auc = 0.0
    early_stop_counter = 0

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CFG["device"], scaler, epoch)
        val_loss, macro_auc, micro_auc = validate(model, val_loader, criterion, CFG["device"])
        if scheduler:
            current_lr = scheduler.step()
        else:
            current_lr = CFG["lr"]

        print(f"Epoch {epoch:2d}/{CFG['epochs']} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
              f"Macro AUC: {macro_auc:.4f} | Micro AUC: {micro_auc:.4f} | LR: {current_lr:.2e}")

        if macro_auc > best_auc:
            best_auc = macro_auc
            torch.save(model.state_dict(), CFG["model_save_path"])
            print(f"  >> Best model saved! Macro AUC: {macro_auc:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= CFG["early_stop_patience"]:
            print(f"  >> Early stopping triggered after {epoch} epochs.")
            break

        torch.cuda.empty_cache()

    print(f"\n🏆 Finished! Best Macro AUC: {best_auc:.4f}")