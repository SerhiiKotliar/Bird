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
    "batch_size": 32,  # Уменьшил для стабильности
    "epochs": 30,  # Уменьшил, чтобы избежать переобучения
    "lr": 1e-4,  # Увеличил начальную LR
    "weight_decay": 0.05,  # Усилил регуляризацию
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_audio_dir": "datas/train_audio",
    "soundscape_dir": "datas/train_soundscapes",
    "label_csv": "datas/train_soundscapes_labels.csv",
    "model_save_path": "bird_model_best_auc_b0.pth",
    "num_workers": 2,
    "prefetch_factor": 2,
    "use_amp": True,
    "clean_samples_per_epoch": 20000,  # Уменьшил
    "mix_samples_per_epoch": 5000,  # Уменьшил
    "soundscape_samples_per_epoch": 5000,  # Уменьшил
    "mixup_alpha": 0.2,
    "label_smoothing": 0.05,  # Увеличил smoothing
    # Модель: EfficientNet B0 быстрее и меньше переобучается
    "model_name": "efficientnet_b0",  # Изменил с convnext_tiny
    "use_3channel": True,
    # Стратегия обучения
    "use_auc_loss": False,  # Временно отключил AUC loss (он может мешать)
    "use_focal_loss": True,  # Включил Focal Loss для борьбы с дисбалансом
    "focal_gamma": 2.0,
    "focal_alpha": 0.75,
    # Learning rate стратегия
    "lr_schedule": "cosine",  # Cosine annealing
    "warmup_epochs": 2,
}

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])
    torch.backends.cudnn.benchmark = True


# ================== FOCAL LOSS для борьбы с дисбалансом ==================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def time_to_seconds(t):
    if pd.isna(t):
        return 0.0
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam


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
        # Фильтруем только интервалы с реальными метками
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
        return min(CFG.get("soundscape_samples_per_epoch", 5000), len(self.data) * 10)

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

        # Упрощённые аугментации (меньше шума)
        if self.augment and random.random() < 0.3:
            y = y * random.uniform(0.8, 1.2)

        waveform = torch.from_numpy(y).float().unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)

        if CFG["use_3channel"]:
            mel = mel.repeat(3, 1, 1)

        # Бинарная метка (без label smoothing для валидной AUC)
        target = torch.zeros(len(self.mlb.classes_))
        valid_labels = [lab for lab in labels if lab in self.mlb.classes_]
        for lab in valid_labels:
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
        valid_labels = [lab for lab in labels if lab in self.mlb.classes_]
        for lab in valid_labels:
            target[list(self.mlb.classes_).index(lab)] = 1.0

        return mel, target


# ================== ПОСТРОЕНИЕ ДАТАСЕТОВ ==================
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

    # Оставляем только строки с метками
    df = df[df["labels_list"].apply(len) > 0]

    unique_files = df['filename'].unique()
    train_files, val_files = train_test_split(unique_files, test_size=0.15, random_state=CFG["seed"])
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

    return train_ds, val_ds, mlb


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
            drop_rate=0.3,
            drop_path_rate=0.1
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


# ================== Cosine annealing with warmup ==================
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
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

        # Mixup с низкой вероятностью
        if random.random() < 0.3:
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

    # Вычисляем macro AUC (только для классов с положительными метками)
    auc_per_class = []
    for i in range(targets.shape[1]):
        if np.sum(targets[:, i]) > 0:
            try:
                auc = roc_auc_score(targets[:, i], preds[:, i])
                auc_per_class.append(auc)
            except:
                pass

    macro_auc = np.mean(auc_per_class) if auc_per_class else 0.0

    # Также считаем micro AUC
    all_preds_flat = preds.ravel()
    all_targets_flat = targets.ravel()
    micro_auc = roc_auc_score(all_targets_flat, all_preds_flat)

    return running_loss / len(loader), macro_auc, micro_auc


# ================== ОСНОВНАЯ ФУНКЦИЯ ==================
if __name__ == "__main__":
    print(f"Device: {CFG['device']}, model: {CFG['model_name']}")

    train_ds, val_ds, mlb = build_datasets()
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

    # Выбор loss функции
    if CFG["use_focal_loss"]:
        criterion = FocalLoss(gamma=CFG["focal_gamma"], alpha=CFG["focal_alpha"])
        print(f"✅ Using Focal Loss (gamma={CFG['focal_gamma']}, alpha={CFG['focal_alpha']})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("✅ Using BCE Loss")

    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scaler = GradScaler(enabled=CFG["use_amp"])

    if CFG["lr_schedule"] == "cosine":
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=CFG["warmup_epochs"],
            total_epochs=CFG["epochs"],
            base_lr=CFG["lr"],
            min_lr=1e-6
        )
    else:
        scheduler = None

    best_auc = 0.0

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CFG["device"], scaler, epoch)
        val_loss, macro_auc, micro_auc = validate(model, val_loader, criterion, CFG["device"])

        if scheduler:
            current_lr = scheduler.step()
        else:
            current_lr = CFG["lr"]

        print(f"Epoch {epoch:2d}/{CFG['epochs']} | "
              f"TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
              f"Macro AUC: {macro_auc:.4f} | Micro AUC: {micro_auc:.4f} | LR: {current_lr:.2e}")

        if macro_auc > best_auc:
            best_auc = macro_auc
            torch.save(model.state_dict(), CFG["model_save_path"])
            print(f"  >> Best model saved! Macro AUC: {macro_auc:.4f}")

        torch.cuda.empty_cache()

    print(f"\n🏆 Finished! Best Macro AUC: {best_auc:.4f}")