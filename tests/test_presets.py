import pytest

from pocket_tts.presets import GenerationPreset, get_preset, list_presets, merge_presets


def test_list_presets_contains_defaults():
    presets = list_presets()
    assert "default" in presets
    assert "slow" in presets
    assert "fast" in presets


def test_get_preset_unknown():
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset("does-not-exist")


def test_merge_presets_override_values():
    base = GenerationPreset(temperature=0.7, lsd_decode_steps=1, gain_db=1.0)
    override = GenerationPreset(temperature=0.4, gain_db=None, time_scale=1.2)
    merged = merge_presets(base, override)
    assert merged.temperature == 0.4
    assert merged.lsd_decode_steps == 1
    assert merged.gain_db == 1.0
    assert merged.time_scale == 1.2
