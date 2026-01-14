import torch

from pocket_tts import main as main_module
from pocket_tts.presets import GenerationPreset


def test_apply_preset_effects_streaming_skips_time_scale(monkeypatch):
    def _raise_time_stretch(*args, **kwargs):
        raise AssertionError("time_stretch_audio should not be called in streaming mode")

    monkeypatch.setattr(main_module, "time_stretch_audio", _raise_time_stretch)

    preset = GenerationPreset(time_scale=1.2, gain_db=6.0)
    chunks = [torch.full((4,), 0.1)]

    output = list(main_module._apply_preset_effects(chunks, preset, 24000, streaming=True))
    assert len(output) == 1
    assert torch.allclose(output[0], torch.full((4,), 0.199526), atol=1e-4)


def test_apply_preset_effects_buffering_uses_time_scale(monkeypatch):
    called = {"value": False}

    def _fake_time_stretch(audio, time_scale):
        called["value"] = True
        return audio

    monkeypatch.setattr(main_module, "time_stretch_audio", _fake_time_stretch)

    preset = GenerationPreset(time_scale=1.2)
    chunks = [torch.zeros(4)]

    output = list(main_module._apply_preset_effects(chunks, preset, 24000, streaming=False))
    assert called["value"] is True
    assert len(output) == 1


def test_apply_preset_effects_streaming_normalizes_per_chunk(monkeypatch):
    calls = {"count": 0}

    def _fake_normalize(chunk, target_db=-20.0, max_gain_db=12.0):
        calls["count"] += 1
        return chunk + 1.0

    monkeypatch.setattr(main_module, "normalize_rms", _fake_normalize)

    preset = GenerationPreset()
    chunks = [torch.zeros(2), torch.zeros(3)]

    output = list(
        main_module._apply_preset_effects(chunks, preset, 24000, normalize=True, streaming=True)
    )
    assert calls["count"] == 2
    assert len(output) == 2
    assert torch.equal(output[0], torch.ones(2))
    assert torch.equal(output[1], torch.ones(3))
