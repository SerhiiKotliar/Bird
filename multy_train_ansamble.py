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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data import ConcatDataset

warnings.filterwarnings('ignore')

# ======================== КОНФИГУРАЦИЯ ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    # ОСНОВНЫЕ ПАРАМЕТРЫ (НЕ ИЗМЕНЯЕМЫЕ - 5 сек, 32000 Гц)
    "sr": 32000,
    "duration": 5.0,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "f_min": 20,
    "f_max": 16000,

    # ПАРАМЕТРЫ МОДЕЛИ
    "model_name": "efficientnet_b0",
    "seed": 456,
    "batch_size": 32,
    "epochs": 60,
    "lr": 2e-5,
    "weight_decay": 0.08,

    # АУГМЕНТАЦИИ
    "use_3channel": True,
    "mixup_alpha": 0.3,
    "mixup_prob": 0.5,

    # ASL LOSS
    "switch_to_asl_epoch": 5,
    "asl_gamma_neg": 4,
    "asl_gamma_pos": 1,
    "asl_clip": 0.05,

    # ОБУЧЕНИЕ
    "use_amp": True,
    "num_workers": 2,
    "clean_samples_per_epoch": 40000,
    "mix_samples_per_epoch": 10000,
    "soundscape_samples_per_epoch": 10000,
    "patience": 10,
    "grad_clip": 1.0,

    # ПУТИ
    "train_audio_dir": os.path.join(SCRIPT_DIR, "datas", "train_audio"),
    "soundscape_dir": os.path.join(SCRIPT_DIR, "datas", "train_soundscapes"),
    "label_csv": os.path.join(SCRIPT_DIR, "datas", "train_soundscapes_labels.csv"),
    "model_save_path": os.path.join(SCRIPT_DIR, "best_model_for_submission.pth"),
    "mlb_save_path": os.path.join(SCRIPT_DIR, "mlb.pkl"),

    # ========= НОВЫЕ ПАРАМЕТРЫ ДЛЯ ВАЛИДАЦИИ И ОБУЧЕНИЯ =========
    "val_mode": "random_offset",  # "central" или "random_offset"
    "use_soundscapes_in_train": False,  # True / False
    "val_random_seed": 123,  # для воспроизводимости random_offset
}

os.makedirs(os.path.dirname(CFG["model_save_path"]), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG["device"] = device

print(f"📁 Папка скрипта: {SCRIPT_DIR}")
print(f"🔧 Устройство: {device}")
print(f"🤖 Модель: {CFG['model_name']}")
print(f"🎯 Learning rate: {CFG['lr']}")
print(f"📊 Batch size: {CFG['batch_size']}")
print(f"🎲 Валидационный режим: {CFG['val_mode']}")
print(f"🎧 Использовать soundscapes в обучении: {CFG['use_soundscapes_in_train']}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(CFG["seed"])


# ======================== ASL LOSS ========================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-7):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x = torch.clamp(x, -10, 10)
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos
        xs_pos = torch.clamp(xs_pos, self.eps, 1 - self.eps)
        xs_neg = torch.clamp(xs_neg, self.eps, 1 - self.eps)
        los_pos = y * torch.log(xs_pos)
        los_neg = (1 - y) * torch.log(xs_neg)
        pt = torch.where(y == 1, xs_pos, xs_neg)
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        one_sided_w = torch.clamp(one_sided_w, 0, 100)
        loss = -(los_pos + los_neg) * one_sided_w
        loss = loss.sum() / (y.sum() + 1)
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(1.0, device=x.device, requires_grad=True)
        return loss


# ======================== МОДЕЛЬ ========================
def build_model(num_classes):
    in_chans = 3 if CFG["use_3channel"] else 1
    model = timm.create_model(
        CFG["model_name"],
        pretrained=True,
        in_chans=in_chans,
        num_classes=0,
        drop_rate=0.2,
        drop_path_rate=0.1
    )
    feat_dim = model.num_features
    classifier = nn.Linear(feat_dim, num_classes)
    nn.init.xavier_uniform_(classifier.weight)
    nn.init.zeros_(classifier.bias)

    class WrappedModel(nn.Module):
        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier

        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)

    return WrappedModel(model, classifier).to(device)


# ======================== ФУНКЦИИ ДЛЯ ДАННЫХ ========================
def time_to_seconds(t):
    if pd.isna(t): return 0.0
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


class MelTransformMixin:
    def __init__(self):
        self.target_len = int(CFG["duration"] * CFG["sr"])
        self.mel_transform = T.MelSpectrogram(
            sample_rate=CFG["sr"], n_fft=CFG["n_fft"],
            hop_length=CFG["hop_length"], n_mels=CFG["n_mels"],
            f_min=CFG["f_min"], f_max=CFG["f_max"]
        ).to(device)

    def waveform_to_mel(self, y):
        if len(y) < self.target_len:
            y = np.pad(y, (0, self.target_len - len(y)))
        elif len(y) > self.target_len:
            y = y[:self.target_len]
        waveform = torch.from_numpy(y).float().unsqueeze(0).to(device)
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
        except:
            y = np.zeros(target_len, dtype=np.float32)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        elif len(y) > target_len:
            y = y[:target_len]
        return y

    def augment_audio(self, y):
        target_len = int(CFG["duration"] * CFG["sr"])
        if random.random() < 0.3:
            y = y + np.random.normal(0, 0.003, len(y))
        if random.random() < 0.2:
            rate = random.uniform(0.9, 1.1)
            y = librosa.effects.time_stretch(y, rate=rate)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]
        y = y * random.uniform(0.7, 1.3)
        return y


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
            if len(y) < self.target_len:
                y = np.pad(y, (0, self.target_len - len(y)))
            else:
                y = y[:self.target_len]
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


class SoundscapeTrainDataset(BaseAudioDataset):
    def __init__(self, df, mlb, train_files, augment=True):
        super().__init__(augment=augment)
        self.mlb = mlb
        self.data = []
        for _, row in df.iterrows():
            if not row["labels_list"] or row["filename"] not in train_files:
                continue
            fpath = os.path.join(CFG["soundscape_dir"], row["filename"])
            if not os.path.exists(fpath):
                continue
            s = time_to_seconds(row["start"])
            e = time_to_seconds(row["end"])
            if e - s >= CFG["duration"]:
                self.data.append((fpath, s, e, row["labels_list"]))
        print(f"🔊 SoundscapeTrain: {len(self.data)} интервалов")

    def __len__(self):
        return CFG["soundscape_samples_per_epoch"]

    def __getitem__(self, idx):
        fpath, start, end, labels = random.choice(self.data)
        max_start = end - CFG["duration"]
        offset = random.uniform(start, max_start) if max_start > start else start
        try:
            y, _ = librosa.load(fpath, sr=CFG["sr"], mono=True, offset=offset, duration=CFG["duration"])
        except:
            y = np.zeros(self.target_len, dtype=np.float32)
        if self.augment:
            y = self.augment_audio(y)
        mel = self.waveform_to_mel(y)
        target = torch.zeros(len(self.mlb.classes_))
        for lab in labels:
            if lab in self.mlb.classes_:
                target[list(self.mlb.classes_).index(lab)] = 1.0
        return mel, target


class SoundscapeValDataset(BaseAudioDataset):
    def __init__(self, df, mlb, val_files, mode="central", random_seed=None):
        super().__init__(augment=False)
        self.mlb = mlb
        self.mode = mode  # "central" или "random_offset"
        self.rng = random.Random(random_seed) if random_seed is not None else random
        self.data = []
        for _, row in df.iterrows():
            if not row["labels_list"] or row["filename"] not in val_files:
                continue
            fpath = os.path.join(CFG["soundscape_dir"], row["filename"])
            if not os.path.exists(fpath):
                continue
            s = time_to_seconds(row["start"])
            e = time_to_seconds(row["end"])
            if e - s < CFG["duration"]:
                continue
            # Валидация: для каждого интервала создаём один сэмпл
            if mode == "central":
                mid = s + (e - s - CFG["duration"]) / 2.0
                self.data.append((fpath, mid, row["labels_list"]))
            else:  # random_offset
                # Сохраняем интервал и будем генерировать offset при каждом обращении
                self.data.append((fpath, s, e, row["labels_list"]))
        print(f"🔊 SoundscapeVal ({mode}): {len(self.data)} интервалов")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "central":
            fpath, start, labels = self.data[idx]
            try:
                y, _ = librosa.load(fpath, sr=CFG["sr"], mono=True, offset=start, duration=CFG["duration"])
            except:
                y = np.zeros(self.target_len, dtype=np.float32)
        else:  # random_offset
            fpath, s, e, labels = self.data[idx]
            max_start = e - CFG["duration"]
            if max_start > s:
                start = self.rng.uniform(s, max_start)
            else:
                start = s
            try:
                y, _ = librosa.load(fpath, sr=CFG["sr"], mono=True, offset=start, duration=CFG["duration"])
            except:
                y = np.zeros(self.target_len, dtype=np.float32)
        mel = self.waveform_to_mel(y)
        target = torch.zeros(len(self.mlb.classes_))
        for lab in labels:
            if lab in self.mlb.classes_:
                target[list(self.mlb.classes_).index(lab)] = 1.0
        return mel, target


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


def build_datasets():
    print("📂 Сбор данных...")
    # 1. Чистые записи
    clean_info = []
    all_classes = set()
    for class_dir in tqdm(os.listdir(CFG["train_audio_dir"]), desc="Clean classes"):
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
    print(f"✅ Всего классов: {len(all_classes)}")
    with open(CFG["mlb_save_path"], "wb") as f:
        pickle.dump(mlb, f)

    # 2. Soundscapes
    df = pd.read_csv(CFG["label_csv"], dtype={"primary_label": str})
    df["labels_list"] = df["primary_label"].apply(lambda x: x.split(";") if pd.notna(x) else [])
    unique_files = df["filename"].unique()
    random.seed(CFG["seed"])
    shuffled = random.sample(list(unique_files), len(unique_files))
    split = int(0.8 * len(shuffled))
    train_sound_files = set(shuffled[:split])
    val_sound_files = set(shuffled[split:])
    print(f"🎧 Soundscape: всего {len(unique_files)}, train {len(train_sound_files)}, val {len(val_sound_files)}")

    # 3. Датасеты
    train_clean = CleanSegments(clean_info, mlb, augment=True)
    train_mix = MixSegments(clean_info, mlb)

    # ----------------------- ВАЖНО -----------------------
    # Формируем список обучающих датасетов в зависимости от флага
    train_datasets = [train_clean, train_mix]
    if CFG["use_soundscapes_in_train"]:
        train_sound = SoundscapeTrainDataset(df, mlb, train_sound_files, augment=True)
        train_datasets.append(train_sound)
        print("🔊 Soundscapes ДОБАВЛЕНЫ в обучение")
    else:
        print("🔇 Soundscapes НЕ используются в обучении (только чистые + миксы)")

    # Валидация
    val_sound = SoundscapeValDataset(df, mlb, val_sound_files,
                                     mode=CFG["val_mode"],
                                     random_seed=CFG.get("val_random_seed", None))

    # 4. Сэмплер для чистых
    sample_weights = get_class_weights(clean_info, mlb)
    sampler = WeightedRandomSampler(sample_weights, CFG["clean_samples_per_epoch"], replacement=True)
    train_clean_loader = DataLoader(train_clean, batch_size=CFG["batch_size"], sampler=sampler,
                                    num_workers=CFG["num_workers"], pin_memory=True)
    train_mix_loader = DataLoader(train_mix, batch_size=CFG["batch_size"], shuffle=True,
                                  num_workers=CFG["num_workers"], pin_memory=True)

    # Собираем ConcatDataset из всех выбранных датасетов
    # Но DataLoader для каждого датасета уже есть. Удобнее создать один большой датасет через ConcatDataset
    # и затем DataLoader. Для простоты используем ConcatDataset напрямую.
    combined_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(combined_dataset, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_sound, batch_size=CFG["batch_size"], shuffle=False,
                            num_workers=CFG["num_workers"], pin_memory=True)

    print(f"📊 Train сэмплов за эпоху: {len(combined_dataset)}")
    print(f"📊 Val сэмплов: {len(val_sound)}")
    return train_loader, val_loader, mlb


@torch.no_grad()
def validate_auc(model, val_loader):
    model.eval()
    all_preds, all_targets = [], []
    for data, target in tqdm(val_loader, desc="Validation", leave=False):
        data, target = data.to(device), target.to(device)
        with autocast(enabled=CFG["use_amp"]):
            outputs = model(data)
            probs = torch.sigmoid(outputs)
        all_preds.append(probs.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    aucs = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            try:
                aucs.append(roc_auc_score(targets[:, i], preds[:, i]))
            except:
                pass
    return np.mean(aucs) if aucs else 0.0


def train_model():
    print("\n🚀 НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ\n" + "=" * 60)
    train_loader, val_loader, mlb = build_datasets()
    model = build_model(len(mlb.classes_))
    print(f"📊 Параметров: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler(enabled=CFG["use_amp"])

    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, CFG["epochs"] + 1):
        if epoch <= CFG["switch_to_asl_epoch"]:
            criterion = nn.BCEWithLogitsLoss()
            loss_name = "BCE"
        else:
            criterion = AsymmetricLoss(CFG["asl_gamma_neg"], CFG["asl_gamma_pos"], CFG["asl_clip"])
            loss_name = "ASL"
            if epoch == CFG["switch_to_asl_epoch"] + 1:
                print(f"\n🔥 Переключение на ASL Loss, уменьшаем LR")
                for g in optimizer.param_groups:
                    g['lr'] = CFG["lr"] * 0.5

        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG['epochs']}")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with autocast(enabled=CFG["use_amp"]):
                outputs = model(data)
                loss = criterion(outputs, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = train_loss / len(train_loader)
        val_auc = validate_auc(model, val_loader)
        scheduler.step()

        print(
            f"Epoch {epoch:3d} | Train Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | Loss: {loss_name}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': CFG["model_name"],
                'mlb_classes': mlb.classes_,
                'best_auc': best_auc,
                'best_epoch': epoch,
            }, CFG["model_save_path"])
            print(f"  >> ✅ НОВАЯ ЛУЧШАЯ МОДЕЛЬ! AUC={val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= CFG["patience"]:
                print(f"\n🛑 Ранняя остановка. Лучший AUC={best_auc:.4f} (эпоха {best_epoch})")
                break

        if epoch % 10 == 0:
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'auc': val_auc},
                       CFG["model_save_path"].replace(".pth", f"_checkpoint_epoch_{epoch}.pth"))

    print(f"\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО! Лучший AUC: {best_auc:.4f} на эпохе {best_epoch}")
    print(f"📁 Модель сохранена: {CFG['model_save_path']}")
    return model, mlb, best_auc


if __name__ == "__main__":
    print("=" * 60 + "\n🎯 ТРЕНИРОВКА МОДЕЛИ ДЛЯ BIRDCLEF 2026\n" + "=" * 60)
    model, mlb, best_auc = train_model()
    print(f"\n✨ МОДЕЛЬ ГОТОВА! Лучший AUC: {best_auc:.4f}")
    print("Запустите скрипт сабмита для генерации предсказаний.")