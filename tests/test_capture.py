import numpy as np

from studyard.capture import mix_mono


def test_mix_mono_averages_and_clips():
    a = np.array([1.0, 0.5], dtype=np.float32)
    b = np.array([1.0, -0.5], dtype=np.float32)
    out = mix_mono(a, b)
    assert out[0] == 1.0
    assert abs(out[1] - 0.0) < 1e-6
