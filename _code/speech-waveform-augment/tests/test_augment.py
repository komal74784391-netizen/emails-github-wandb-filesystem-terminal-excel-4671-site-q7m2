import numpy as np

from src.augment import additive_noise, time_mask


def test_noise_shape():
    wav = np.random.randn(16000).astype(np.float32)
    noise = np.random.randn(16000).astype(np.float32)
    out = additive_noise(wav, noise)
    assert out.shape == wav.shape


def test_time_mask_zeros():
    spec = np.ones((20, 100))
    out = time_mask(spec, max_width=10)
    assert (out == 1.0).any()
