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
    "batch_size": 64,
    "epochs": 50,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_audio_dir": "datas/train_audio",
    "soundscape_dir": "datas/train_soundscapes",
    "label_csv": "datas/train_soundscapes_labels.csv",
    "model_save_path": "bird_model_tiny_auc.pth",
    "num_workers": 1,
    "prefetch_factor": 2,
    "use_amp": True,
    "clean_samples_per_epoch": 30000,
    "mix_samples_per_epoch": 8000,
    "soundscape_samples_per_epoch": 8000,
    "mixup_alpha": 0.3,
    "label_smoothing": 0.02,
    "model_name": "convnext_tiny",
    "use_3channel": True,
    "grad_low_thresh": 0.1,
    "grad_high_thresh": 1.0,
    "lr_increase_factor": 1.1,
    "lr_decay_factor": 0.9,
    "loss_increase_penalty": 0.95,
    "loss_improve_bonus": 1.05,
    "min_lr": 1e-7,
    "max_lr": 1e-2,
    # Новые параметры для AUC loss и LR scheduling
    "use_auc_loss": True,  # Использовать AUC loss
    "auc_loss_weight": 0.1,  # Вес AUC loss в комбинированной функции
    "auc_margin": 1.0,  # Margin для AUC loss
    "reduce_lr_patience": 3,  # Эпох без улучшения AUC для уменьшения LR
    "reduce_lr_factor": 0.5,  # Множитель уменьшения LR
    "reduce_lr_min_delta": 0.001,  # Минимальное изменение AUC для считать улучшением
}

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])
    torch.backends.cudnn.benchmark = True


# ================== DIFFERENTIABLE AUC LOSS ==================
class AUCMLoss(nn.Module):
    """
    Differentiable approximation of AUC using pairwise ranking loss.
    Based on: "AUC Optimization for Deep Learning" - optimizes AUC directly.
    """

    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes) - raw logits from model
            targets: (batch_size, num_classes) - binary labels (0 or 1)
        Returns:
            differentiable AUC loss
        """
        probs = torch.sigmoid(logits)
        loss = 0.0
        num_classes = targets.shape[1]
        valid_classes = 0

        for c in range(num_classes):
            pos_mask = targets[:, c] > 0.5
            neg_mask = targets[:, c] < 0.5

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            pos_probs = probs[pos_mask, c]
            neg_probs = probs[neg_mask, c]

            # Pairwise ranking loss: maximize (pos_prob - neg_prob)
            # Using squared hinge loss approximation for differentiability
            pos_expanded = pos_probs.unsqueeze(1)  # (n_pos, 1)
            neg_expanded = neg_probs.unsqueeze(0)  # (1, n_neg)

            # Compute pairwise differences
            diff = pos_expanded - neg_expanded

            # Smooth approximation of AUC using logistic loss
            # Instead of counting pairs where diff > 0, we use sigmoid
            pair_loss = torch.log(1 + torch.exp(-self.margin * diff))

            valid_classes += 1
            loss += pair_loss.mean()

        if valid_classes > 0:
            loss = loss / valid_classes

        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * num_classes
        return loss


class CombinedLoss(nn.Module):
    """
    Combined BCE and AUC loss for stable training with AUC optimization.
    """

    def __init__(self, bce_weight=1.0, auc_weight=0.1, auc_margin=1.0):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.auc_loss = AUCMLoss(margin=auc_margin)
        self.bce_weight = bce_weight
        self.auc_weight = auc_weight

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        auc = self.auc_loss(logits, targets)
        total = self.bce_weight * bce + self.auc_weight * auc
        return total


def time_to_seconds(t):
    if pd.isna(t):
        return 0.0
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
            if e - s < CFG["duration"]:
                continue
            self.data.append((fpath, s, e, row["labels_list"]))

    def __len__(self):
        return CFG.get("soundscape_samples_per_epoch", 8000)

    def __getitem__(self, idx):
        fpath, start, end, labels = random.choice(self.data)
        max_start = end - CFG["duration"]
        if max_start > start:
            offset = random.uniform(start, max_start)
        else:
            offset = start
        try:
            y, _ = librosa.load(fpath, sr=CFG["sr"], mono=True, offset=offset, duration=CFG["duration"])
        except:
            y = np.zeros(self.target_len, dtype=np.float32)

        if len(y) < self.target_len:
            y = np.pad(y, (0, self.target_len - len(y)))
        else:
            y = y[:self.target_len]

        if self.augment:
            if random.random() < 0.3:
                y = y + np.random.normal(0, 0.005, len(y))
            if random.random() < 0.2:
                rate = random.uniform(0.9, 1.1)
                y = librosa.effects.time_stretch(y, rate=rate)
                if len(y) < self.target_len:
                    y = np.pad(y, (0, self.target_len - len(y)))
                else:
                    y = y[:self.target_len]
            y = y * random.uniform(0.8, 1.2)

        waveform = torch.from_numpy(y).float().unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        if CFG["use_3channel"]:
            mel = mel.repeat(3, 1, 1)

        smooth = CFG["label_smoothing"]
        target = torch.full((len(self.mlb.classes_),), smooth / (len(self.mlb.classes_) - 1))
        for lab in labels:
            if lab in self.mlb.classes_:
                target[list(self.mlb.classes_).index(lab)] = 1.0 - smooth
        return mel, target


# ================== ДАТАСЕТЫ ДЛЯ ЧИСТЫХ И СМЕШАННЫХ ЗАПИСЕЙ ==================
class OnTheFlyDatasetBase(Dataset):
    def __init__(self, file_info, augment=False):
        self.file_info = file_info
        self.augment = augment
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
            s = random.randint(0, len(y) - self.target_len)
            y = y[s:s + self.target_len]
        if self.augment:
            if random.random() < 0.3:
                y = y + np.random.normal(0, 0.005, len(y))
            if random.random() < 0.2:
                rate = random.uniform(0.9, 1.1)
                y = librosa.effects.time_stretch(y, rate=rate)
                if len(y) < self.target_len:
                    y = np.pad(y, (0, self.target_len - len(y)))
                else:
                    y = y[:self.target_len]
            y = y * random.uniform(0.8, 1.2)
        waveform = torch.from_numpy(y).float().unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        if CFG["use_3channel"]:
            mel = mel.repeat(3, 1, 1)
        return mel


class CleanSegments(OnTheFlyDatasetBase):
    def __len__(self):
        return CFG["clean_samples_per_epoch"]

    def __getitem__(self, idx):
        i = random.randrange(len(self.file_info))
        path, label, dur = self.file_info[i]
        y = self.load_segment(path, dur)
        mel = self.process_audio(y)
        return mel, label


class MixSegments(OnTheFlyDatasetBase):
    def __init__(self, file_info, mlb):
        super().__init__(file_info, augment=False)
        self.mlb = mlb

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
            if len(y) < self.target_len:
                y = np.pad(y, (0, self.target_len - len(y)))
            else:
                s = random.randint(0, max(0, len(y) - self.target_len))
                y = y[s:s + self.target_len]
            mix += y * random.uniform(0.4, 1.0)
        peak = np.abs(mix).max()
        if peak > 0:
            mix = mix / peak * 0.95
        waveform = torch.from_numpy(mix).float().unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        if CFG["use_3channel"]:
            mel = mel.repeat(3, 1, 1)

        smooth = CFG["label_smoothing"]
        target = torch.full((len(self.mlb.classes_),), smooth / (len(self.mlb.classes_) - 1))
        for lab in labels:
            if lab in self.mlb.classes_:
                target[list(self.mlb.classes_).index(lab)] = 1.0 - smooth
        return mel, target


class BinaryWrapper(Dataset):
    def __init__(self, base_ds, mlb):
        self.base = base_ds
        self.mlb = mlb

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        mel, label_list = self.base[idx]
        smooth = CFG["label_smoothing"]
        target = torch.full((len(self.mlb.classes_),), smooth / (len(self.mlb.classes_) - 1))
        for lab in label_list:
            if lab in self.mlb.classes_:
                target[list(self.mlb.classes_).index(lab)] = 1.0 - smooth
        return mel, target


# ================== ВАЛИДАЦИОННЫЙ ДАТАСЕТ (ТОЛЬКО НА ОТДЕЛЬНЫХ ФАЙЛАХ) ==================
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


# ================== ПОСТРОЕНИЕ ДАТАСЕТОВ С РАЗДЕЛЕНИЕМ ПО ФАЙЛАМ ==================
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

    # Clean + Mix (без изменений)
    train_clean = CleanSegments(clean_info, augment=True)
    train_clean_bin = BinaryWrapper(train_clean, mlb)
    train_mix = MixSegments(clean_info, mlb)

    # ========== РАЗДЕЛЕНИЕ ЗВУКОВЫХ ЛАНДШАФТОВ ПО ФАЙЛАМ ==========
    df = pd.read_csv(CFG["label_csv"], dtype={"primary_label": str})
    df["labels_list"] = df["primary_label"].apply(lambda x: x.split(";") if pd.notna(x) else [])

    unique_files = df['filename'].unique()
    train_files, val_files = train_test_split(
        unique_files, test_size=0.15, random_state=CFG["seed"]
    )
    df_train = df[df['filename'].isin(train_files)]
    df_val = df[df['filename'].isin(val_files)]
    print(f"✅ Звуковые ландшафты: train файлов={len(train_files)}, val файлов={len(val_files)}")

    train_soundscape = SoundscapeDataset(df_train, mlb, augment=True)
    val_ds = ValDataset(df_val, mlb)  # валидация на отдельных файлах

    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset([train_clean_bin, train_mix, train_soundscape])

    print(
        f"✅ Train: {len(train_ds)} (clean: {len(train_clean_bin)} / mix: {len(train_mix)} / soundscape: {len(train_soundscape)})")
    print(f"✅ Val: {len(val_ds)} (на {len(val_files)} файлах)")
    return train_ds, val_ds, mlb


# ================== МОДЕЛЬ ==================
def build_model(num_classes):
    model_name = CFG["model_name"]
    in_chans = 3 if CFG["use_3channel"] else 1
    if model_name.startswith("efficientnet"):
        model = timm.create_model(model_name, pretrained=True, in_chans=in_chans,
                                  num_classes=num_classes, drop_rate=0.25, drop_path_rate=0.05)
    elif model_name.startswith("convnext"):
        model = timm.create_model(model_name, pretrained=True, in_chans=in_chans,
                                  num_classes=num_classes, drop_path_rate=0.1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(CFG["device"])


# ================== АДАПТИВНАЯ СКОРОСТЬ ОБУЧЕНИЯ ==================
class AdaptiveLRScheduler:
    """
    Комбинированный scheduler: градиенты + loss + plateau на AUC
    """

    def __init__(self, optimizer, patience=3, factor=0.5, min_delta=0.001):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_delta = min_delta
        self.best_auc = 0.0
        self.counter = 0

    def step(self, current_auc):
        if current_auc > self.best_auc + self.min_delta:
            self.best_auc = current_auc
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                # Уменьшаем LR
                for param_group in self.optimizer.param_groups:
                    new_lr = param_group["lr"] * self.factor
                    param_group["lr"] = new_lr
                print(f"  📉 Plateau detected! AUC didn't improve for {self.patience} epochs. "
                      f"Reducing LR to {new_lr:.2e}")
                self.counter = 0
                return True
        return False


def adjust_lr_by_grad_and_loss(optimizer, avg_grad_norm, current_lr, current_val_loss, prev_val_loss):
    new_lr = current_lr
    loss_improved = (current_val_loss < prev_val_loss) if prev_val_loss is not None else True

    if avg_grad_norm > CFG["grad_high_thresh"]:
        base_factor = CFG["lr_decay_factor"]
        if not loss_improved:
            base_factor *= CFG["loss_increase_penalty"]
            print(
                f"  ⚠️ Большие градиенты ({avg_grad_norm:.4f}) И loss не улучшился → LR уменьшаем сильнее (factor={base_factor:.3f})")
        else:
            print(
                f"  📉 Большие градиенты ({avg_grad_norm:.4f}), но loss улучшился → LR уменьшаем (factor={base_factor:.3f})")
        new_lr = max(CFG["min_lr"], current_lr * base_factor)
    elif avg_grad_norm < CFG["grad_low_thresh"]:
        base_factor = CFG["lr_increase_factor"]
        if loss_improved:
            base_factor *= CFG["loss_improve_bonus"]
            print(
                f"  🚀 Малые градиенты ({avg_grad_norm:.4f}) И loss улучшился → LR увеличиваем сильнее (factor={base_factor:.3f})")
        else:
            print(
                f"  📈 Малые градиенты ({avg_grad_norm:.4f}), но loss не улучшился → LR увеличиваем (factor={base_factor:.3f})")
        new_lr = min(CFG["max_lr"], current_lr * base_factor)
    else:
        print(f"  ✅ Градиенты в норме ({avg_grad_norm:.4f}) → LR не меняем ({current_lr:.2e})")

    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


# ================== ТРЕНИРОВКА ОДНОЙ ЭПОХИ ==================
def train_epoch(model, loader, optimizer, criterion, device, scaler, epoch):
    model.train()
    running_loss = 0.0
    total_grad_norm_sq = 0.0
    num_batches = 0
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
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = np.sqrt(grad_norm)
            total_grad_norm_sq += grad_norm ** 2
            num_batches += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = np.sqrt(grad_norm)
            total_grad_norm_sq += grad_norm ** 2
            num_batches += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        running_loss += loss.item()
        if pbar.n % 10 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grad_norm:.4f}")

    avg_grad_norm = np.sqrt(total_grad_norm_sq / num_batches) if num_batches > 0 else 0.0
    return running_loss / len(loader), avg_grad_norm


# ================== ВАЛИДАЦИЯ С ROC-AUC ==================
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

    # Макроусреднённый ROC-AUC с пропуском классов без положительных меток
    auc_per_class = []
    for i in range(targets.shape[1]):
        if np.sum(targets[:, i]) == 0:
            continue
        auc = roc_auc_score(targets[:, i], preds[:, i])
        auc_per_class.append(auc)
    macro_auc = np.mean(auc_per_class) if auc_per_class else 0.0

    # Битовая точность (для информации)
    acc = ((preds > 0.5) == targets).mean()

    return running_loss / len(loader), acc, macro_auc


# ================== ОСНОВНАЯ ФУНКЦИЯ ==================
if __name__ == "__main__":
    print(f"Device: {CFG['device']}, model: {CFG['model_name']}, 3ch: {CFG['use_3channel']}")
    print(f"AUC Loss: {CFG['use_auc_loss']}, AUC weight: {CFG['auc_loss_weight']}")

    train_ds, val_ds, mlb = build_datasets()

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"], pin_memory=False,
                              prefetch_factor=CFG["prefetch_factor"], persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                            num_workers=CFG["num_workers"], pin_memory=False,
                            persistent_workers=False)

    model = build_model(len(mlb.classes_))

    # Выбор функции потерь
    if CFG["use_auc_loss"]:
        criterion = CombinedLoss(
            bce_weight=1.0,
            auc_weight=CFG["auc_loss_weight"],
            auc_margin=CFG["auc_margin"]
        )
        print(f"✅ Using Combined Loss (BCE + AUC) with AUC weight={CFG['auc_loss_weight']}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("✅ Using standard BCEWithLogitsLoss")

    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scaler = GradScaler(enabled=CFG["use_amp"])

    # Scheduler для уменьшения LR при остановке роста AUC
    plateau_scheduler = AdaptiveLRScheduler(
        optimizer,
        patience=CFG["reduce_lr_patience"],
        factor=CFG["reduce_lr_factor"],
        min_delta=CFG["reduce_lr_min_delta"]
    )

    best_auc = 0.0
    best_loss = float('inf')
    current_lr = CFG["lr"]
    prev_val_loss = None

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss, avg_grad_norm = train_epoch(model, train_loader, optimizer, criterion,
                                                CFG["device"], scaler, epoch)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, CFG["device"])

        # 1. Адаптация LR на основе градиентов и loss
        current_lr = adjust_lr_by_grad_and_loss(optimizer, avg_grad_norm, current_lr, val_loss, prev_val_loss)
        prev_val_loss = val_loss

        # 2. Уменьшение LR при остановке роста AUC (plateau)
        plateau_scheduler.step(val_auc)

        print(f"Epoch {epoch:2d}/{CFG['epochs']} | TrLoss: {train_loss:.4f} | "
              f"ValLoss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | LR: {current_lr:.2e}")

        if val_auc > best_auc or (val_auc == best_auc and val_loss < best_loss):
            best_auc = val_auc
            best_loss = val_loss
            torch.save(model.state_dict(), CFG["model_save_path"])
            print(f"  >> Best model saved (AUC {val_auc:.4f})")
        torch.cuda.empty_cache()

    print(f"Finished. Best AUC: {best_auc:.4f}")