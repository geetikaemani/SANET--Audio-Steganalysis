import os
import uuid
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display

# No model used now – UI first!


BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_audio(path, sr=8000, segment_seconds=10):
    """Load audio and pad/trim to fixed length."""
    audio, _ = librosa.load(path, sr=sr)

    target_len = sr * segment_seconds
    if len(audio) > target_len:
        start = (len(audio) - target_len) // 2
        audio = audio[start:start + target_len]
    else:
        pad = target_len - len(audio)
        audio = np.pad(audio, (pad // 2, pad - pad // 2), mode="constant")

    return audio, sr


def _make_spectrograms(audio, sr):
    """Generate and save spectrogram + mel-spectrogram images."""
    uid = uuid.uuid4().hex

    # 1) STFT Spectrogram
    spec_file = f"{uid}_spectrogram.png"
    spec_path = os.path.join(RESULTS_DIR, spec_file)

    D = librosa.stft(y=audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(5, 3))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="log")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.savefig(spec_path)
    plt.close()

    # 2) Mel Spectrogram
    mel_file = f"{uid}_mel.png"
    mel_path = os.path.join(RESULTS_DIR, mel_file)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(5, 3))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(mel_path)
    plt.close()

    # Return just filenames – app.py will serve them via /results/<filename>
    return spec_file, mel_file


def run_prediction(filepath: str):
    """
    Dummy inference: just to make the frontend fully work.
    - Loads audio
    - Generates spectrograms
    - Returns fake prediction + confidence
    """
    audio, sr = _load_audio(filepath)

    # Simple dummy rule: energy-based classification (just for demo)
    energy = float(np.mean(audio ** 2))
    if energy > 0.01:
        pred_label = "stego"
        confidence = 78.3
    else:
        pred_label = "cover"
        confidence = 84.5

    spec_file, mel_file = _make_spectrograms(audio, sr)

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "spectrogram1": f"/results/{spec_file}",
        "spectrogram2": f"/results/{mel_file}",
        "waveform": audio.tolist(),  # for waveform canvas
    }
