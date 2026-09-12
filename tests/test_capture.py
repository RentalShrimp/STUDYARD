import numpy as np

from studyard.capture import (
    channel_count_from_info,
    combine_capture_parts,
    is_analog_jack,
    is_mic_array,
    mix_mono,
    normalize_peak,
    resample_mono,
    select_mic_device_indices,
    sum_mono,
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


def test_analog_jack_name_is_p2_not_array():
    assert is_analog_jack("Microfone (Realtek HD Audio Mic input)")
    assert not is_analog_jack("Grupo de microfones (Realtek HD Audio Mic Array input)")
    assert is_mic_array("Grupo de microfones (Realtek(R) Audio)")
    assert not is_mic_array("Microfone (AB13X USB Audio)")


def test_select_mic_includes_wasapi_and_wdm_analog_skips_mme():
    devices = [
        {"index": 1, "name": "Microfone (AB13X USB Audio)", "hostapi_name": "MME", "max_input_channels": 1},
        {"index": 14, "name": "Microfone (AB13X USB Audio)", "hostapi_name": "Windows WASAPI", "max_input_channels": 1},
        {"index": 15, "name": "Grupo de microfones (Realtek(R) Audio)", "hostapi_name": "Windows WASAPI", "max_input_channels": 2},
        {"index": 13, "name": "Alto-falantes (AB13X USB Audio)", "hostapi_name": "Windows WASAPI", "max_input_channels": 0},
        {"index": 24, "name": "Microfone (Realtek HD Audio Mic input)", "hostapi_name": "Windows WDM-KS", "max_input_channels": 2},
        {"index": 23, "name": "Grupo de microfones (Realtek HD Audio Mic Array input)", "hostapi_name": "Windows WDM-KS", "max_input_channels": 2},
        {"index": 26, "name": "Microfone (AB13X USB Audio)", "hostapi_name": "Windows WDM-KS", "max_input_channels": 1},
    ]
    picked = select_mic_device_indices(devices)
    assert 14 in picked
    assert 15 not in picked
    assert 24 in picked
    assert 23 in picked
    assert 1 not in picked
    assert 13 not in picked
    assert 26 not in picked


def test_sum_mono_silent_source_does_not_attenuate():
    active = np.array([0.5, -0.4], dtype=np.float32)
    silent = np.array([0.0, 0.0], dtype=np.float32)
    out = sum_mono([active, silent])
    assert np.allclose(out, active)


def test_combine_parts_sums_same_kind_then_averages_mic_vs_system():
    parts = {
        "mic:0": np.array([0.5, 0.5], dtype=np.float32),
        "mic:1": np.array([0.0, 0.0], dtype=np.float32),
        "system:0": np.array([0.0, 0.0], dtype=np.float32),
        "system:1": np.array([0.8, 0.8], dtype=np.float32),
    }
    out = combine_capture_parts(parts)
    assert abs(out[0] - 0.65) < 1e-5


def test_combine_silent_system_keeps_mic_level():
    parts = {
        "mic:0": np.array([0.5, 0.5], dtype=np.float32),
        "system:0": np.array([0.0, 0.0], dtype=np.float32),
    }
    out = combine_capture_parts(parts)
    assert np.allclose(out, np.array([0.5, 0.5], dtype=np.float32))


def test_normalize_peak_boosts_quiet_speech():
    quiet = np.array([0.02, -0.01], dtype=np.float32)
    out = normalize_peak(quiet, target=0.4, max_gain=25.0)
    assert float(np.max(np.abs(out))) > 0.3


def test_normalize_peak_leaves_silence():
    silent = np.zeros(4, dtype=np.float32)
    assert np.allclose(normalize_peak(silent), silent)

