"""Audio loading helpers."""

import wave

import numpy as np


def load_wav(path):
    with wave.open(path, "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        return data / 32768.0, handle.getframerate()
