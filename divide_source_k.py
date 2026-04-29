import soundfile as sf
from tqdm import tqdm
import os
import pandas as pd
import subprocess
import numpy as np
import librosa
import torch
import torchaudio
# from tqdm import tqdm
import sys
sys.path.append('/kaggle/working/libs')
import asteroid
# ------------------------------------------------------

def time_to_seconds(t):
    if pd.isna(t):
        return 0.0
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)
# ---------------------------------------------------------

import subprocess

# === ПУТИ ===
CSV_PATH = "/kaggle/input/competitions/birdclef-2026/train_soundscapes_labels.csv"
AUDIO_DIR = "/kaggle/input/competitions/birdclef-2026/train_soundscapes"
OUTPUT_DIR = "chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. Читаем CSV ===
df = pd.read_csv(CSV_PATH, dtype={"primary_label": str})

# === 2. Разбиваем источники по ; ===
df["labels_list"] = df["primary_label"].apply(
    lambda x: x.split(";") if pd.notna(x) else []
)

# === 3. функция ffmpeg ===
def cut_audio(input_file, output_file, start, duration):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-ss", str(start),
        "-t", str(duration),
        "-c", "copy",
        output_file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_row(row):
    filename = row.filename
    start = time_to_seconds(row.start)
    end = time_to_seconds(row.end)
    labels = row.labels_list

    input_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(input_path):
        return None

    duration = end - start

    base = filename.replace(".ogg", "")
    chunk_name = f"{base}_{int(start)}_{int(end)}.ogg"
    output_path = os.path.join(OUTPUT_DIR, chunk_name)

    # --- резка ---
    cut_audio(input_path, output_path, start, duration)

    return output_path, labels


def process_all(df, max_workers=8):
    files = []
    sources_in_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_row, df.itertuples(index=False)),
            total=len(df),
            desc="Обработка аудио"
        ))

    for res in results:
        if res is None:
            continue
        f, labels = res
        files.append(f)
        sources_in_files.append(labels)
    # пути нарезанных файлов и списки их источников
    return files, sources_in_files
# -------------------------------------------------------------------

files, sources_in_files = process_all(df, max_workers=10)
print("\nКоличество 5 секундных многоисточниковых эталонных файлов:", len(files))
# -------------------------------------------------------------------------

import torch
import torchaudio
import torch.nn.functional as F

def separate_to_sources_fast(
    files,
    sources_in_files,
    train_audio_dir,
    output_dir,
    device="cuda",
    use_fp16=True
):
    os.makedirs(output_dir, exist_ok=True)

    counters = {}
    reference_cache = {}

    print("🔧 Processing files (GPU optimized)...")

    for file_path, src_list in tqdm(
        list(zip(files, sources_in_files)),
        total=len(files),
        desc="Processing"
    ):
        # === загрузка многоисточникового файла (CPU) ===
        waveform, sr = torchaudio.load(file_path)

        # сразу уменьшаем память
        if use_fp16:
            waveform = waveform.half()

        waveform = waveform.to(device, non_blocking=True)
        # выделение одного источника из списка для названия папки
        for src_name in src_list:

            src_folder = os.path.join(train_audio_dir, src_name)
            if not os.path.isdir(src_folder):
                continue

            # папка вывода для одного источника
            out_dir = os.path.join(output_dir, src_name)
            os.makedirs(out_dir, exist_ok=True)

            if src_name not in counters:
                counters[src_name] = 0

            # =========================
            # КЭШ ЭТАЛОНОВ (ВАЖНО!)
            # =========================
            if src_name not in reference_cache:

                ref_path = None
                for f in os.listdir(src_folder):
                    if f.endswith(".ogg"):
                        ref_path = os.path.join(src_folder, f)
                        break

                if ref_path is None:
                    continue

                ref_wave, _ = torchaudio.load(ref_path)

                if use_fp16:
                    ref_wave = ref_wave.half()

                reference_cache[src_name] = ref_wave  # храним на CPU

            ref_wave = reference_cache[src_name]

            # переносим на GPU только на время вычисления
            ref_wave = ref_wave.to(device, non_blocking=True)

            # =========================
            # ОБРАБОТКА
            # =========================
            min_len = min(waveform.shape[1], ref_wave.shape[1])

            x = waveform[:, :min_len]
            ref = ref_wave[:, :min_len]

            # нормализация
            x_norm = F.normalize(x, dim=1)
            ref_norm = F.normalize(ref, dim=1)

            similarity = (x_norm * ref_norm).sum(dim=0)

            # более мягкая маска (лучше качество)
            mask = torch.sigmoid(similarity * 5)

            separated = x * mask

            # нормализация
            max_val = separated.abs().max()
            if max_val > 0:
                separated = separated / max_val

            # =========================
            # СОХРАНЕНИЕ
            # =========================
            # out_name = f"etalon_{counters[src_name]}.ogg"
            out_name = f"{src_name}_{counters[src_name]}.ogg"
            out_path = os.path.join(out_dir, out_name)

            torchaudio.save(out_path, separated.float().cpu(), sr)

            counters[src_name] += 1

            # освобождаем GPU
            del ref_wave, ref, ref_norm, similarity, mask, separated
            torch.cuda.empty_cache()

        del waveform
        torch.cuda.empty_cache()

    print("✅ Done!")

separate_to_sources_fast(
    files,                      # список 5-сек файлов
    sources_in_files,           # список списков источников
    train_audio_dir="/kaggle/input/competitions/birdclef-2026/train_audio",
    output_dir="/kaggle/working/separated_output",
    device="cuda",
    use_fp16=True
)