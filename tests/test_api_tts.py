import pytest
import torch
from fastapi.testclient import TestClient

from pocket_tts import main as main_module


class _DummyMimiConfig:
    sample_rate = 24000


class _DummyConfig:
    mimi = _DummyMimiConfig()


class _DummyModel:
    def __init__(self) -> None:
        self.sample_rate = 24000
        self.config = _DummyConfig()
        self.temp = 0.5
        self.lsd_decode_steps = 2
        self.noise_clamp = 0.0
        self.eos_threshold = -3.0

    def generate_audio_stream(self, model_state, text_to_generate, frames_after_eos=None):
        yield torch.zeros(480)

    def get_state_for_audio_prompt(self, prompt, truncate=True):
        return {"voice": str(prompt), "truncate": truncate}

    def _cached_get_state_for_audio_prompt(self, prompt, truncate=True):
        return {"voice": str(prompt), "truncate": truncate, "cached": True}


def _client():
    main_module.tts_model = _DummyModel()
    main_module.global_model_state = {"voice": "default"}
    main_module.global_default_voice = None
    return TestClient(main_module.web_app)


def test_tts_success():
    client = _client()
    response = client.post("/tts", data={"text": "Hello from tests"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content.startswith(b"RIFF")


def test_tts_empty_text_validation():
    client = _client()
    response = client.post("/tts", data={"text": "   "})
    assert response.status_code == 400
    assert response.json()["detail"] == "Text cannot be empty"


def test_tts_rejects_conflicting_voice_inputs():
    client = _client()
    response = client.post(
        "/tts",
        data={"text": "Hello", "voice_url": "https://example.com/voice.wav"},
        files={"voice_wav": ("voice.wav", b"not-a-wav")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Cannot provide both voice_url and voice_wav"


def test_tts_rejects_invalid_voice_url():
    client = _client()
    response = client.post("/tts", data={"text": "Hello", "voice_url": "ftp://invalid"})
    assert response.status_code == 400
    assert response.json()["detail"] == "voice_url must start with http://, https://, or hf://"


def test_tts_rejects_unknown_preset():
    client = _client()
    response = client.post("/tts", data={"text": "Hello", "preset": "unknown"})
    assert response.status_code == 400
    assert "Unknown preset" in response.json()["detail"]


def test_tts_ssml_success_and_invalid_markup(monkeypatch):
    def _dummy_iter_audio_from_segments(*args, **kwargs):
        yield torch.zeros(480)

    monkeypatch.setattr(main_module, "_iter_audio_from_segments", _dummy_iter_audio_from_segments)
    client = _client()
    ok = client.post("/tts", data={"text": "<speak>Hello</speak>", "ssml": "true"})
    assert ok.status_code == 200
    assert ok.content.startswith(b"RIFF")

    bad = client.post("/tts", data={"text": "<speak><bad /></speak>", "ssml": "true"})
    assert bad.status_code == 400
    assert "Unsupported SSML tag" in bad.json()["detail"]


def test_tts_rejects_non_wav_upload():
    client = _client()
    response = client.post(
        "/tts",
        data={"text": "Hello"},
        files={"voice_wav": ("voice.txt", b"not-a-wav")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded voice file must be a WAV file"


def test_tts_rejects_large_voice_upload(monkeypatch):
    client = _client()
    monkeypatch.setattr(main_module, "_MAX_VOICE_BYTES", 4)
    response = client.post(
        "/tts",
        data={"text": "Hello"},
        files={"voice_wav": ("voice.wav", b"RIFF0000WAVE")},
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


def test_tts_rejects_long_voice_upload(monkeypatch):
    import io
    import wave

    client = _client()
    monkeypatch.setattr(main_module, "_MAX_VOICE_SECONDS", 0.01)
    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x00" * 24000)
    response = client.post(
        "/tts",
        data={"text": "Hello"},
        files={"voice_wav": ("voice.wav", wav_bytes.getvalue())},
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]
