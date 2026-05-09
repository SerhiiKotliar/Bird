import os, warnings, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.cuda.amp import autocast
import librosa
from tqdm import tqdm
import timm

warnings.filterwarnings('ignore')

# ======================== КОНФИГУРАЦИЯ ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    # ОСНОВНЫЕ ПАРАМЕТРЫ (ДОЛЖНЫ СОВПАДАТЬ С ТРЕНИРОВКОЙ)
    "sr": 32000,
    "duration": 5.0,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "f_min": 20,
    "f_max": 16000,

    "model_name": "efficientnet_b0",
    "use_3channel": True,
    "use_amp": True,
    "batch_size": 32,  # Для инференса

    # ПУТИ
    "model_path": os.path.join(SCRIPT_DIR, "best_model_for_submission.pth"),
    "mlb_path": os.path.join(SCRIPT_DIR, "mlb.pkl"),
    "test_soundscapes_dir": os.path.join(SCRIPT_DIR, "datas", "test_soundscapes"),
    "taxonomy_csv": os.path.join(SCRIPT_DIR, "datas", "taxonomy.csv"),
    "submission_save_path": os.path.join(SCRIPT_DIR, "submission.csv"),
}

# Устанавливаем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG["device"] = device

print(f"📁 Папка скрипта: {SCRIPT_DIR}")
print(f"🔧 Устройство: {device}")
print(f"🤖 Модель: {CFG['model_name']}")
print(f"📁 Тестовые данные: {CFG['test_soundscapes_dir']}")


# ======================== МОДЕЛЬ ========================
def build_model(num_classes):
    in_chans = 3 if CFG["use_3channel"] else 1

    model = timm.create_model(
        CFG["model_name"],
        pretrained=False,
        in_chans=in_chans,
        num_classes=0,
        drop_rate=0.2,
        drop_path_rate=0.1
    )
    feat_dim = model.num_features
    classifier = nn.Linear(feat_dim, num_classes)

    class WrappedModel(nn.Module):
        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier

        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)

    wrapped_model = WrappedModel(model, classifier)
    return wrapped_model


def load_model(model_path, mlb):
    """Загрузка обученной модели"""
    print(f"📂 Загрузка модели из: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    num_classes = len(mlb.classes_)
    model = build_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✅ Модель загружена")
    print(f"   Модель: {checkpoint.get('model_name', 'unknown')}")
    print(f"   Эпоха: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Loss: {checkpoint.get('best_loss', 'unknown'):.4f}" if 'best_loss' in checkpoint else "")

    return model


# ======================== ФУНКЦИИ ДЛЯ ПРЕДСКАЗАНИЙ ========================
def create_mel_transform():
    """Создание mel-спектрограммы (должно совпадать с тренировкой)"""
    mel_transform = T.MelSpectrogram(
        sample_rate=CFG["sr"], n_fft=CFG["n_fft"],
        hop_length=CFG["hop_length"], n_mels=CFG["n_mels"],
        f_min=CFG["f_min"], f_max=CFG["f_max"]
    ).to(device)
    return mel_transform


def predict_segment(model, mel_transform, audio_chunk):
    """Предсказание для одного сегмента"""
    target_len = int(CFG["duration"] * CFG["sr"])

    # Подготовка аудио
    if len(audio_chunk) < target_len:
        audio_chunk = np.pad(audio_chunk, (0, target_len - len(audio_chunk)))
    elif len(audio_chunk) > target_len:
        audio_chunk = audio_chunk[:target_len]

    # Преобразование в mel
    waveform = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(device)
    mel = mel_transform(waveform)
    mel = torch.log(mel + 1e-6)

    if CFG["use_3channel"]:
        mel = mel.repeat(3, 1, 1)

    mel = mel.unsqueeze(0)  # Добавляем batch dimension

    # Инференс
    with torch.no_grad():
        with autocast(enabled=CFG["use_amp"]):
            output = model(mel)
            probs = torch.sigmoid(output).squeeze(0).cpu().numpy()

    return probs


# ======================== ГЕНЕРАЦИЯ САБМИТА ========================
def generate_submission(model, mlb, mel_transform):
    print("\n📝 ГЕНЕРАЦИЯ САБМИТА...")

    # Загрузка таксономии (234 класса для сабмита)
    taxonomy = pd.read_csv(CFG["taxonomy_csv"])
    submission_classes = taxonomy["primary_label"].tolist()
    print(f"📊 Классов в сабмите: {len(submission_classes)}")

    # Классы из обучения
    train_classes = mlb.classes_
    print(f"📊 Классов в модели: {len(train_classes)}")

    # Поиск тестовых файлов
    test_files = [f for f in os.listdir(CFG["test_soundscapes_dir"]) if f.endswith(".ogg")]
    print(f"🎵 Найдено тестовых файлов: {len(test_files)}")

    if len(test_files) == 0:
        print("❌ Ошибка: Не найдены тестовые файлы!")
        return None

    target_len = int(CFG["duration"] * CFG["sr"])
    all_rows = []

    for fname in tqdm(test_files, desc="Обработка файлов"):
        filepath = os.path.join(CFG["test_soundscapes_dir"], fname)
        basename = fname.replace(".ogg", "")

        try:
            y, sr = librosa.load(filepath, sr=CFG["sr"], mono=True)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {fname}: {e}")
            continue

        # Обработка каждого 5-секундного окна (с 5 по 60 секунду)
        for end_time in range(5, 65, 5):
            row_id = f"{basename}_{end_time}"
            start_time = end_time - CFG["duration"]
            start_smpl = int(start_time * sr)
            end_smpl = int(end_time * sr)

            # Извлечение сегмента
            if end_smpl > len(y):
                chunk = y[start_smpl:]
                if len(chunk) < target_len:
                    chunk = np.pad(chunk, (0, target_len - len(chunk)))
            else:
                chunk = y[start_smpl:end_smpl]

            # Предсказание
            probs = predict_segment(model, mel_transform, chunk)

            # Проекция на 234 класса (заполняем нулями отсутствующие классы)
            row_probs = np.zeros(len(submission_classes), dtype=np.float32)
            for i, cls in enumerate(train_classes):
                if cls in submission_classes:
                    idx = submission_classes.index(cls)
                    row_probs[idx] = probs[i]

            all_rows.append([row_id] + row_probs.tolist())

    # Создание DataFrame
    submission_df = pd.DataFrame(all_rows, columns=["row_id"] + submission_classes)
    submission_df = submission_df.sort_values("row_id").reset_index(drop=True)

    # Сохранение
    submission_df.to_csv(CFG["submission_save_path"], index=False)
    print(f"\n✅ САБМИТ СОХРАНЁН!")
    print(f"📁 Путь: {CFG['submission_save_path']}")
    print(f"📊 Строк: {len(submission_df)}")
    print(f"📊 Колонок: {len(submission_df.columns)}")

    return submission_df


# ======================== ВЕРИФИКАЦИЯ ========================
def verify_submission(submission_df):
    """Проверка корректности сабмита"""
    print("\n🔍 ПРОВЕРКА САБМИТА:")

    # Проверка формата
    assert "row_id" in submission_df.columns, "Нет колонки row_id"

    # Проверка значений (должны быть в [0, 1])
    prob_columns = [col for col in submission_df.columns if col != "row_id"]
    values = submission_df[prob_columns].values
    assert (values >= 0).all() and (values <= 1).all(), "Значения вероятностей вне диапазона [0, 1]"

    print(f"✅ Формат корректен")
    print(f"   Строк: {len(submission_df)}")
    print(f"   Колонок: {len(submission_df.columns)}")
    print(f"   Диапазон вероятностей: [{values.min():.4f}, {values.max():.4f}]")

    # Пример первых строк
    print("\n📋 Пример первых строк:")
    print(submission_df.head(3))


# ======================== MAIN ========================
if __name__ == "__main__":
    print("=" * 60)
    print("🎯 ГЕНЕРАЦИЯ САБМИТА ДЛЯ BIRDCLEF 2026")
    print("=" * 60)

    # 1. Загрузка MLB
    print(f"\n📂 Загрузка MLB из: {CFG['mlb_path']}")
    with open(CFG["mlb_path"], "rb") as f:
        mlb = pickle.load(f)
    print(f"✅ MLB загружен, классов: {len(mlb.classes_)}")

    # 2. Загрузка модели
    model = load_model(CFG["model_path"], mlb)

    # 3. Создание mel-преобразования
    mel_transform = create_mel_transform()

    # 4. Генерация сабмита
    submission_df = generate_submission(model, mlb, mel_transform)

    # 5. Проверка сабмита
    if submission_df is not None:
        verify_submission(submission_df)

    print("\n" + "=" * 60)
    print("✨ ГОТОВО! Файл submission.csv создан и готов к отправке!")
    print("=" * 60)