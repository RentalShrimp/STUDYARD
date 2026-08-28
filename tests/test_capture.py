import numpy as np

from studyard.capture import (
    channel_count_from_info,
    mix_mono,
    resample_mono,
    to_mono,
)


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


def test_to_mono_stereo_averages_channels():
    frames = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
    out = to_mono(frames)
    assert out.shape == (2,)
    assert abs(out[0] - 0.0) < 1e-6
    assert abs(out[1] - 0.5) < 1e-6


def test_to_mono_keeps_1d():
    src = np.array([0.1, -0.2], dtype=np.float32)
    assert np.allclose(to_mono(src), src)


def test_channel_count_loopback_uses_output():
    info = {"max_input_channels": 0, "max_output_channels": 2}
    assert channel_count_from_info(info, loopback=True) == 2
    assert channel_count_from_info(info, loopback=False) == 1

