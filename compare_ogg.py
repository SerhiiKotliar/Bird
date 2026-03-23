import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import faiss
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
DATASET_DIR = "dataset_ogg"
EMB_PATH = "embeddings.npy"
PATHS_PATH = "paths.npy"

TOP_K = 5

# ==============================
# LOAD MODEL
# ==============================
print("Loading YAMNet...")
model = hub.load("https://tfhub.dev/google/yamnet/1")


# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_embedding(path):
    try:
        y, sr = librosa.load(path, sr=16000)

        # если тишина / битый файл
        if len(y) == 0:
            return None

        scores, embeddings, spectrogram = model(y)

        emb = np.mean(embeddings.numpy(), axis=0)
        return emb.astype("float32")

    except Exception as e:
        print(f"Error with {path}: {e}")
        return None


# ==============================
# BUILD DATABASE (ОДИН РАЗ)
# ==============================
def build_database():
    embeddings = []
    paths = []

    files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".ogg")]

    print(f"Processing {len(files)} files...")

    for file in tqdm(files):
        path = os.path.join(DATASET_DIR, file)

        emb = extract_embedding(path)

        if emb is not None:
            embeddings.append(emb)
            paths.append(path)

    embeddings = np.array(embeddings)

    print("Saving...")
    np.save(EMB_PATH, embeddings)
    np.save(PATHS_PATH, np.array(paths))

    return embeddings, paths


# ==============================
# LOAD DATABASE
# ==============================
def load_database():
    embeddings = np.load(EMB_PATH)
    paths = np.load(PATHS_PATH, allow_pickle=True)
    return embeddings, paths


# ==============================
# BUILD FAISS INDEX
# ==============================
def build_index(embeddings):
    dim = embeddings.shape[1]

    # косинусное сходство → нормализация
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dim)  # inner product
    index.add(embeddings)

    return index


# ==============================
# SEARCH
# ==============================
def search(query_path, index, paths, top_k=TOP_K):
    emb = extract_embedding(query_path)

    if emb is None:
        return []

    emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)

    scores, indices = index.search(emb, top_k)

    results = []
    for i, score in zip(indices[0], scores[0]):
        results.append((paths[i], float(score)))

    return results


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    if not os.path.exists(EMB_PATH):
        embeddings, paths = build_database()
    else:
        embeddings, paths = load_database()

    print("Building FAISS index...")
    index = build_index(embeddings)

    print("Ready!")

    # ==========================
    # ПРИМЕР ЗАПРОСА
    # ==========================
    query_file = "query.ogg"

    results = search(query_file, index, paths)

    print("\nTop matches:")
    for path, score in results:
        print(f"{score:.3f} | {path}")