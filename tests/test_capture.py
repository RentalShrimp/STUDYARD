import numpy as np

from studyard.capture import mix_mono, resample_mono


def test_mix_mono_averages_and_clips():
    a = np.array([1.0, 0.5], dtype=np.float32)
    b = np.array([1.0, -0.5], dtype=np.float32)
    out = mix_mono(a, b)
    assert out[0] == 1.0
    assert abs(out[1] - 0.0) < 1e-6


def test_resample_mono_48000_to_16000_length():
    src = np.linspace(-0.5, 0.5, 4800, dtype=np.float32)
    out = resample_mono(src, 48000, 16000)
    assert abs(len(out) - 1600) <= 1


def test_resample_same_rate_is_identity():
    src = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    out = resample_mono(src, 16000, 16000)
    assert np.allclose(out, src)

