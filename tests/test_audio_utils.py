import pytest
import torch

from pocket_tts.data.audio_utils import convert_audio, normalize_rms


def test_normalize_rms_targets_level():
    audio = torch.ones(24000) * 0.01
    normalized = normalize_rms(audio, target_db=-20.0, max_gain_db=40.0)
    rms = torch.sqrt(torch.mean(normalized.pow(2))).item()
    assert abs(rms - 0.1) < 0.01


def test_convert_audio_rejects_non_2d():
    wav = torch.zeros(10)
    with pytest.raises(ValueError, match="shape"):
        convert_audio(wav, 24000, 24000, 1)


def test_convert_audio_rejects_fractional_sample_rate():
    wav = torch.zeros(1, 10)
    with pytest.raises(ValueError, match="sample rate"):
        convert_audio(wav, 24000.5, 24000, 1)


def test_convert_audio_accepts_integer_like_float_rate():
    wav = torch.zeros(1, 10)
    converted = convert_audio(wav, 24000.0, 24000.0, 1)
    assert converted.shape == wav.shape
