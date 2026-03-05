import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid

OUTPUT_DIR = "backend/outputs/spectrograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return torch.tensor(np.mean(mfcc, axis=1), dtype=torch.float)

def generate_waveform(filepath):
    y, _ = librosa.load(filepath, sr=16000)
    return y[:700]  # trimming for frontend plot

def generate_spectrogram(filepath):
    y, sr = librosa.load(filepath)
    S = librosa.feature.melspectrogram(y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    filename1 = f"{uuid.uuid4()}_spec1.png"
    filename2 = f"{uuid.uuid4()}_spec2.png"

    plt.figure(figsize=(6,3))
    librosa.display.specshow(S_DB, sr=sr)
    plt.savefig(f"{OUTPUT_DIR}/{filename1}")
    plt.close()

    return f"/static/spec/{filename1}", f"/static/spec/{filename2}"
