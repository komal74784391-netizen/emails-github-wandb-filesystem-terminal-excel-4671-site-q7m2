"""Waveform augmentation transforms."""

import numpy as np


def speed_perturb(waveform, sample_rate, factors=(0.9, 1.0, 1.1)):
    factor = np.random.choice(factors)
    idx = np.arange(0, len(waveform), factor)
    return np.interp(idx, np.arange(len(waveform)), waveform)


def additive_noise(waveform, noise, snr_db=10.0):
    signal_power = np.mean(waveform ** 2)
    noise_power = np.mean(noise ** 2)
    scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10.0)))
    return waveform + scale * noise


def time_mask(spectrogram, max_width=20):
    width = np.random.randint(0, max_width)
    start = np.random.randint(0, max(1, spectrogram.shape[-1] - width))
    out = spectrogram.copy()
    out[..., start:start + width] = 0.0
    return out
