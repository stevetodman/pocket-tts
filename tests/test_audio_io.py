import wave

import numpy as np
import torch

from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio


def _write_wav(path, sample_rate: int, frames: np.ndarray) -> None:
    frames = np.asarray(frames, dtype=np.int16)
    if frames.ndim == 1:
        frames = frames[:, None]
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(frames.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames.tobytes())


def test_audio_read_mono(tmp_path):
    path = tmp_path / "mono.wav"
    _write_wav(path, 16000, np.array([[0], [32767]], dtype=np.int16))

    audio, sample_rate = audio_read(path)

    assert sample_rate == 16000
    assert audio.shape == (1, 2)
    expected = torch.tensor([[0.0, 32767 / 32768.0]])
    assert torch.allclose(audio, expected, atol=1e-6)


def test_audio_read_stereo(tmp_path):
    path = tmp_path / "stereo.wav"
    _write_wav(path, 22050, np.array([[0, 32767], [32767, 0]], dtype=np.int16))

    audio, sample_rate = audio_read(path)

    assert sample_rate == 22050
    assert audio.shape == (2, 2)
    expected = torch.tensor([[0.0, 32767 / 32768.0], [32767 / 32768.0, 0.0]])
    assert torch.allclose(audio, expected, atol=1e-6)


def test_convert_audio_downmix_to_mono():
    wav = torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])

    converted = convert_audio(wav, 16000, 16000, 1)

    assert converted.shape == (1, 3)
    expected = torch.tensor([[0.5, 0.0, -0.5]])
    assert torch.allclose(converted, expected, atol=1e-6)


def test_convert_audio_upmix_from_mono():
    wav = torch.tensor([[0.25, -0.25]])

    converted = convert_audio(wav, 24000, 24000, 2)

    assert converted.shape == (2, 2)
    expected = torch.tensor([[0.25, -0.25], [0.25, -0.25]])
    assert torch.allclose(converted, expected, atol=1e-6)
