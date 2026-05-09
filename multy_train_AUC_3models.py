import os, random, warnings, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
import librosa
from tqdm import tqdm
import timm
from transformers import ASTConfig, ASTForAudioClassification
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data import ConcatDataset

warnings.filterwarnings('ignore')

# ======================== КОНФИГУРАЦИЯ ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    "lr": 1e-4,
    "weight_decay": 0.05,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "train_audio_dir": os.path.join(SCRIPT_DIR, "datas", "train_audio"),
    "soundscape_dir": os.path.join(SCRIPT_DIR, "datas", "train_soundscapes"),
    "label_csv": os.path.join(SCRIPT_DIR, "datas", "train_soundscapes_labels.csv"),

    "model_save_dir": SCRIPT_DIR,
    "switch_to_asl_after_epochs": 5,  # Переключиться на ASL после N эпох
    "initial_loss": "bce",  # 'bce' или 'asl'
    "num_workers": 2,
    "prefetch_factor": 2,
    "use_amp": True,
    "clean_samples_per_epoch": 30000,
    "mix_samples_per_epoch": 8000,
    "soundscape_samples_per_epoch": 8000,
    "mixup_alpha": 0.3,
    "label_smoothing": 0.02,
    "use_3channel": True,
    "asl_gamma_neg": 4,
    "asl_gamma_pos": 1,
    "asl_clip": 0.05,
    "patience": 3,
    "use_asl_loss": False,  # Сначала используем BCEWithLogitsLoss, потом переключим на True
    "ensemble_models": [
        {"name": "efficientnet_b2", "seed": 42},
        {"name": "convnext_tiny", "seed": 123},
        {"name": "efficientnet_b0", "seed": 456},
    ],
}

os.makedirs(CFG["model_save_dir"], exist_ok=True)

print(f"📁 Папка скрипта: {SCRIPT_DIR}")
print(f"📁 Сохранение моделей: {CFG['model_save_dir']}")


# ======================== ASL LOSS (ИСПРАВЛЕННАЯ ВЕРСИЯ) ========================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-7):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        """Asymmetric Loss для multi-label классификации с численной стабильностью"""
        # 1. Стабильность входных данных
        x = torch.clamp(x, -10, 10)

        # 2. Вычисляем вероятности
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        # 3. Защита от log(0)
        xs_pos = torch.clamp(xs_pos, self.eps, 1 - self.eps)
        xs_neg = torch.clamp(xs_neg, self.eps, 1 - self.eps)

        # 4. Базовые потери
        los_pos = y * torch.log(xs_pos)
        los_neg = (1 - y) * torch.log(xs_neg)

        # 5. Взвешивание
        pt = torch.where(y == 1, xs_pos, xs_neg)
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        one_sided_w = torch.clamp(one_sided_w, 0, 100)

        # 6. Финальные потери
        loss = -(los_pos + los_neg) * one_sided_w

        # 7. Нормализация по количеству положительных меток
        loss = loss.sum() / (y.sum() + 1)

        # 8. Защита от NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(1.0, device=x.device, requires_grad=True)

        return loss


# ======================== AST MODEL ========================
class ASTModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        config = ASTConfig()
        config.num_labels = num_classes
        self.backbone = ASTForAudioClassification(config)
        feat_dim = self.backbone.classifier.dense.in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        x = x.squeeze(1)
        x = F.interpolate(x, size=1024, mode='linear', align_corners=False)
        x = x.transpose(1, 2)
        features = self.backbone(x).logits
        return self.classifier(features)


# ======================== ОБЁРТКА ДЛЯ TIMM МОДЕЛЕЙ ========================
class TimmModelWrapper(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ======================== MODEL FACTORY ========================
def build_model(model_name, num_classes):
    in_chans = 3 if CFG["use_3channel"] else 1

    if model_name == 'ast':
        model = ASTModel(num_classes)
    elif model_name.startswith("efficientnet"):
        backbone = timm.create_model(model_name, pretrained=True, in_chans=in_chans,
                                     num_classes=0, drop_rate=0.25, drop_path_rate=0.05)
        model = TimmModelWrapper(backbone, num_classes)
    elif model_name.startswith("convnext"):
        backbone = timm.create_model(model_name, pretrained=True, in_chans=in_chans,
                                     num_classes=0, drop_path_rate=0.1)
        model = TimmModelWrapper(backbone, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(CFG["device"])


# ======================== ФУНКЦИИ ========================
def time_to_seconds(t):
    if pd.isna(t): return 0.0
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# ======================== БАЗОВЫЙ КЛАСС ========================
class MelTransformMixin:
    def __init__(self):
        self.target_len = int(CFG["duration"] * CFG["sr"])
        self.mel_transform = T.MelSpectrogram(
            sample_rate=CFG["sr"], n_fft=CFG["n_fft"],
            hop_length=CFG["hop_length"], n_mels=CFG["n_mels"],
            f_min=CFG["f_min"], f_max=CFG["f_max"]
        )

    def waveform_to_mel(self, y):
        if len(y) < self.target_len:
            y = np.pad(y, (0, self.target_len - len(y)))
        elif len(y) > self.target_len:
            y = y[:self.target_len]
        waveform = torch.from_numpy(y).float().unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        if CFG["use_3channel"]:
            mel = mel.repeat(3, 1, 1)
        return mel


class BaseAudioDataset(Dataset, MelTransformMixin):
    def __init__(self, augment=False):
        super().__init__()
        MelTransformMixin.__init__(self)
        self.augment = augment

    def load_segment(self, path, dur):
        target_len = int(CFG["duration"] * CFG["sr"])
        try:
            if dur > CFG["duration"]:
                start = random.uniform(0, dur - CFG["duration"])
                y, _ = librosa.load(path, sr=CFG["sr"], mono=True, offset=start, duration=CFG["duration"])
            else:
                y, _ = librosa.load(path, sr=CFG["sr"], mono=True)
        except Exception:
            y = np.zeros(target_len, dtype=np.float32)

        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        elif len(y) > target_len:
            y = y[:target_len]
        return y

    def augment_audio(self, y):
        target_len = int(CFG["duration"] * CFG["sr"])
        if random.random() < 0.3:
            y = y + np.random.normal(0, 0.005, len(y))
        if random.random() < 0.2:
            rate = random.uniform(0.9, 1.1)
            y = librosa.effects.time_stretch(y, rate=rate)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]
        y = y * random.uniform(0.8, 1.2)
        return y


# ======================== ДАТАСЕТЫ ========================
class CleanSegments(BaseAudioDataset):
    def __init__(self, file_info, mlb, augment=False):
        super().__init__(augment=augment)
        self.file_info = file_info
        self.mlb = mlb

    def __len__(self):
        return CFG["clean_samples_per_epoch"]

    def __getitem__(self, idx):
        path, label_list, dur = self.file_info[idx % len(self.file_info)]
        y = self.load_segment(path, dur)
        if self.augment:
            y = self.augment_audio(y)
        mel = self.waveform_to_mel(y)

        target = torch.zeros(len(self.mlb.classes_))
        for lab in label_list:
            if lab in self.mlb.classes_:
                target[list(self.mlb.classes_).index(lab)] = 1.0
        return mel, target


class MixSegments(BaseAudioDataset):
    def __init__(self, file_info, mlb):
        super().__init__(augment=False)
        self.file_info = file_info
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
            mix += y * random.uniform(0.4, 1.0)

        peak = np.abs(mix).max()
        if peak > 0:
            mix = mix / peak * 0.95

        mel = self.waveform_to_mel(mix)
        target = torch.zeros(len(self.mlb.classes_))
        for lab in labels:
            if lab in self.mlb.classes_:
                target[list(self.mlb.classes_).index(lab)] = 1.0
        return mel, target


class SoundscapeDataset(BaseAudioDataset):
    def __init__(self, df, mlb, train_files, augment=True):
        super().__init__(augment=augment)
        self.mlb = mlb
        self.data = []

        print(f"\n🔍 Загрузка soundscapes для обучения:")
        print(f"   Тренировочных файлов: {len(train_files)}")

        for _, row in df.iterrows():
            if not row["labels_list"]:
                continue
            if row["filename"] not in train_files:
                continue
            fpath = os.path.join(CFG["soundscape_dir"], row["filename"])
            if not os.path.exists(fpath):
                continue
            s = time_to_seconds(row["start"])
            e = time_to_seconds(row["end"])
            if e - s >= CFG["duration"]:
                self.data.append((fpath, s, e, row["labels_list"]))

        print(f"   Загружено интервалов для обучения: {len(self.data)}")
        self.use_fallback = len(self.data) == 0
        self.fallback_data = []

    def set_fallback_data(self, clean_info):
        self.fallback_data = clean_info
        if self.use_fallback:
            print(f"⚠️ Нет soundscapes для обучения. Используем fallback: {len(self.fallback_data)} чистых записей")

    def __len__(self):
        return max(CFG["soundscape_samples_per_epoch"], 1)

    def __getitem__(self, idx):
        if self.use_fallback and len(self.fallback_data) > 0:
            path, labels, dur = random.choice(self.fallback_data)
            y = self.load_segment(path, dur)
            if self.augment:
                y = self.augment_audio(y)
            mel = self.waveform_to_mel(y)
            target = torch.zeros(len(self.mlb.classes_))
            for lab in labels:
                if lab in self.mlb.classes_:
                    target[list(self.mlb.classes_).index(lab)] = 1.0
            return mel, target

        fpath, start, end, labels = random.choice(self.data)
        max_start = end - CFG["duration"]
        offset = random.uniform(start, max_start) if max_start > start else start

        try:
            y, _ = librosa.load(fpath, sr=CFG["sr"], mono=True, offset=offset, duration=CFG["duration"])
        except Exception:
            y = np.zeros(self.target_len, dtype=np.float32)

        if self.augment:
            y = self.augment_audio(y)

        mel = self.waveform_to_mel(y)
        target = torch.zeros(len(self.mlb.classes_))
        for lab in labels:
            if lab in self.mlb.classes_:
                target[list(self.mlb.classes_).index(lab)] = 1.0
        return mel, target


class ValDataset(BaseAudioDataset):
    def __init__(self, df, mlb, val_files=None):
        super().__init__(augment=False)
        self.mlb = mlb
        self.data = []

        for _, row in df.iterrows():
            if not row["labels_list"]:
                continue
            if val_files is not None and row["filename"] not in val_files:
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

        print(f"   Загружено интервалов для валидации: {len(self.data)}")

    def __len__(self):
        return max(len(self.data), 1)

    def __getitem__(self, idx):
        fpath, start, labels = self.data[idx % len(self.data)]
        try:
            y, _ = librosa.load(fpath, sr=CFG["sr"], mono=True, offset=start, duration=CFG["duration"])
        except Exception:
            y = np.zeros(self.target_len, dtype=np.float32)

        mel = self.waveform_to_mel(y)
        target = torch.zeros(len(self.mlb.classes_))
        for lab in labels:
            if lab in self.mlb.classes_:
                target[list(self.mlb.classes_).index(lab)] = 1.0
        return mel, target


# ======================== WEIGHTED SAMPLER ========================
def get_class_weights(clean_info, mlb):
    class_counts = np.zeros(len(mlb.classes_))
    for _, labels, _ in clean_info:
        for lab in labels:
            if lab in mlb.classes_:
                class_counts[list(mlb.classes_).index(lab)] += 1
    weights = 1.0 / (class_counts + 1)
    sample_weights = []
    for i in range(len(clean_info)):
        weight = sum(weights[list(mlb.classes_).index(lab)] for lab in clean_info[i][1] if lab in mlb.classes_)
        sample_weights.append(weight if weight > 0 else 1.0)
    return sample_weights


def build_datasets_for_model(model_name, seed):
    CFG["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"📂 Сбор данных для {model_name} с seed={seed}")
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
    mlb = MultiLabelBinarizer(classes=all_classes)
    mlb.fit([all_classes])

    sample_weights = get_class_weights(clean_info, mlb)
    sampler = WeightedRandomSampler(sample_weights, CFG["clean_samples_per_epoch"], replacement=True)

    df = pd.read_csv(CFG["label_csv"], dtype={"primary_label": str})
    df["labels_list"] = df["primary_label"].apply(lambda x: x.split(";") if pd.notna(x) else [])

    unique_files = df["filename"].unique()
    random.seed(seed)
    rng = random.Random(seed)
    shuffled_files = rng.sample(list(unique_files), len(unique_files))
    split_idx = int(len(shuffled_files) * 0.8)
    train_soundscape_files = set(shuffled_files[:split_idx])
    val_soundscape_files = set(shuffled_files[split_idx:])

    print(f"\n📂 Разделение soundscapes:")
    print(f"   Всего файлов: {len(unique_files)}")
    print(f"   На обучение: {len(train_soundscape_files)}")
    print(f"   На валидацию: {len(val_soundscape_files)}")

    train_clean = CleanSegments(clean_info, mlb, augment=True)
    train_clean_loader = DataLoader(train_clean, batch_size=CFG["batch_size"], sampler=sampler,
                                    num_workers=CFG["num_workers"], pin_memory=True)

    train_mix = MixSegments(clean_info, mlb)
    train_mix_loader = DataLoader(train_mix, batch_size=CFG["batch_size"], shuffle=True,
                                  num_workers=CFG["num_workers"], pin_memory=True)

    train_soundscape = SoundscapeDataset(df, mlb, train_files=train_soundscape_files, augment=True)
    train_soundscape.set_fallback_data(clean_info)
    train_soundscape_loader = DataLoader(train_soundscape, batch_size=CFG["batch_size"], shuffle=True,
                                         num_workers=CFG["num_workers"], pin_memory=True)

    val_ds = ValDataset(df, mlb, val_files=val_soundscape_files)

    train_ds = ConcatDataset([train_clean_loader.dataset, train_mix_loader.dataset, train_soundscape_loader.dataset])
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"], pin_memory=True)

    return train_loader, val_ds, mlb


# ======================== ВАЛИДАЦИЯ ========================
def validate_with_auc(model, val_ds, criterion):
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=CFG["num_workers"])

    for data, target in tqdm(val_loader, desc="Val", leave=False):
        data, target = data.to(CFG["device"]), target.to(CFG["device"])
        with autocast(enabled=CFG["use_amp"]):
            outputs = model(data)
            loss = criterion(outputs, target)
        val_loss += loss.item()
        # Добавляем .detach() перед .cpu().numpy()
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    aucs = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            aucs.append(roc_auc_score(targets[:, i], preds[:, i]))
    macro_auc = np.mean(aucs) if aucs else 0.0

    return val_loss / len(val_loader), macro_auc


# ======================== ФУНКЦИЯ ОБУЧЕНИЯ ========================
def train_model(model_name, seed):
    print(f"\n{'=' * 50}\nОбучение {model_name} с seed={seed}\n{'=' * 50}")

    train_loader, val_ds, mlb = build_datasets_for_model(model_name, seed)

    model = build_model(model_name, len(mlb.classes_))

    # Параметры для переключения loss
    switch_to_asl_after_epochs = CFG.get("switch_to_asl_after_epochs", 5)

    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-7)
    scaler = GradScaler(enabled=CFG["use_amp"])

    best_auc = 0.0
    best_model_path = os.path.join(CFG["model_save_dir"], f"{model_name}_seed{seed}_best.pth")
    patience_counter = 0
    train_losses, val_losses = [], []
    best_epoch = 0

    for epoch in range(1, CFG["epochs"] + 1):
        # Переключаем loss после заданной эпохи
        if epoch <= switch_to_asl_after_epochs:
            criterion = nn.BCEWithLogitsLoss()
            if epoch == 1:
                print("📉 Используем BCEWithLogitsLoss (первые {} эпох)".format(switch_to_asl_after_epochs))
        else:
            if epoch == switch_to_asl_after_epochs + 1:
                criterion = AsymmetricLoss(
                    gamma_neg=CFG["asl_gamma_neg"],
                    gamma_pos=CFG["asl_gamma_pos"],
                    clip=CFG["asl_clip"]
                )
                print(f"\n🔥 Переключение на ASL Loss с эпохи {epoch}")
                # Уменьшаем learning rate при переключении
                for param_group in optimizer.param_groups:
                    param_group['lr'] = CFG["lr"] * 0.5
                print(f"   Learning rate уменьшен до: {optimizer.param_groups[0]['lr']:.2e}")
            else:
                criterion = AsymmetricLoss(
                    gamma_neg=CFG["asl_gamma_neg"],
                    gamma_pos=CFG["asl_gamma_pos"],
                    clip=CFG["asl_clip"]
                )

        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch}")

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(CFG["device"]), target.to(CFG["device"])

            # Отладка на первой эпохе
            if batch_idx == 0 and epoch == 1:
                print(f"\n🔍 Отладка целевых меток:")
                print(f"   Target min: {target.min().item():.4f}")
                print(f"   Target max: {target.max().item():.4f}")
                print(f"   Target mean: {target.mean().item():.4f}")
                print(f"   Положительных меток: {(target > 0.5).sum().item()}")
                print(f"   Размер батча: {data.size(0)}")

            optimizer.zero_grad()
            with autocast(enabled=CFG["use_amp"]):
                outputs = model(data)
                loss = criterion(outputs, target)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ NaN/Inf loss на эпохе {epoch}, батче {batch_idx}, пропускаем")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss, val_auc = validate_with_auc(model, val_ds, criterion)
        scheduler.step(val_loss)

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        # Мониторинг переобучения
        if len(train_losses) > 5 and (train_losses[-1] - val_losses[-1]) > 0.5:
            print(f"⚠️ Переобучение! Увеличиваем weight_decay")
            CFG["weight_decay"] *= 1.1
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = CFG["weight_decay"]

        loss_type = "BCE" if epoch <= switch_to_asl_after_epochs else "ASL"
        print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Loss Type: {loss_type}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'model_name': model_name,
                'seed': seed,
                'mlb_classes': mlb.classes_,
                'loss_type': loss_type,
            }, best_model_path)
            print(f"  >> ✅ Лучшая модель сохранена (эпоха {epoch}, AUC={val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CFG["patience"]:
                print(f"🛑 Ранняя остановка на эпохе {epoch}. Лучшая AUC={best_auc:.4f} (эпоха {best_epoch})")
                break

    print(f"\n📊 Итог обучения {model_name}_seed{seed}:")
    print(f"   Лучший ROC-AUC: {best_auc:.4f} на эпохе {best_epoch}")

    checkpoint = torch.load(best_model_path, map_location=CFG["device"], weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, best_auc, best_model_path

# ======================== MAIN ========================
if __name__ == "__main__":
    print("🚀 Начало обучения ансамбля моделей\n")
    print(f"Устройство: {CFG['device']}")
    print(f"Количество моделей в ансамбле: {len(CFG['ensemble_models'])}")
    print(f"Функция потерь: {'ASL Loss' if CFG['use_asl_loss'] else 'BCEWithLogitsLoss'}")

    trained_models_info = []

    for i, model_cfg in enumerate(CFG["ensemble_models"], 1):
        print(f"\n📌 Модель {i}/{len(CFG['ensemble_models'])}")
        model, auc, model_path = train_model(model_cfg["name"], model_cfg["seed"])
        trained_models_info.append({
            'name': model_cfg["name"],
            'seed': model_cfg["seed"],
            'auc': auc,
            'path': model_path,
            'model': model
        })

    total_auc = sum(info['auc'] for info in trained_models_info)

    print(f"\n{'=' * 60}")
    print("📊 Результаты обучения моделей:")
    print(f"{'=' * 60}")
    for info in trained_models_info:
        weight = info['auc'] / total_auc
        print(f"   {info['name']}_seed{info['seed']}: AUC={info['auc']:.4f}, вес={weight:.3f}")

    ensemble_info = {
        'models': trained_models_info,
        'weights': [info['auc'] / total_auc for info in trained_models_info],
        'total_auc': total_auc
    }

    with open(os.path.join(CFG["model_save_dir"], 'ensemble_info.pkl'), 'wb') as f:
        pickle.dump(ensemble_info, f)

    print(f"\n✅ Информация об ансамбле сохранена")

    best_model = max(trained_models_info, key=lambda x: x['auc'])
    print(f"\n🏆 Лучшая отдельная модель: {best_model['name']}_seed{best_model['seed']} (AUC={best_model['auc']:.4f})")