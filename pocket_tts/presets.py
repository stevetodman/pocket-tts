from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationPreset:
    temperature: float | None = None
    lsd_decode_steps: int | None = None
    noise_clamp: float | None = None
    eos_threshold: float | None = None
    frames_after_eos: int | None = None
    gain_db: float | None = None
    time_scale: float | None = None
    word_pause_ms: int | None = None


PRESETS: dict[str, GenerationPreset] = {
    "default": GenerationPreset(),
    "broadcast": GenerationPreset(
        temperature=0.45,
        lsd_decode_steps=4,
        eos_threshold=-3.8,
    ),
    "calm": GenerationPreset(
        temperature=0.4,
        lsd_decode_steps=3,
        eos_threshold=-4.8,
    ),
    "expressive": GenerationPreset(
        temperature=1.0,
        lsd_decode_steps=2,
        eos_threshold=-3.2,
        gain_db=2.0,
    ),
    "dramatic": GenerationPreset(
        temperature=1.1,
        lsd_decode_steps=2,
        eos_threshold=-2.8,
        gain_db=3.5,
    ),
    "slow": GenerationPreset(
        temperature=0.5,
        lsd_decode_steps=2,
        eos_threshold=-3.2,
        time_scale=1.15,
        word_pause_ms=60,
    ),
    "fast": GenerationPreset(
        temperature=0.9,
        lsd_decode_steps=1,
        eos_threshold=-5.2,
        time_scale=0.88,
    ),
}


def list_presets() -> list[str]:
    return sorted(PRESETS)


def get_preset(name: str | None) -> GenerationPreset:
    if name is None:
        return PRESETS["default"]
    preset_name = name.strip().lower()
    if preset_name == "":
        return PRESETS["default"]
    if preset_name not in PRESETS:
        available = ", ".join(list_presets())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[preset_name]


def list_presets_data() -> list[dict[str, float | int | str | None]]:
    result = []
    for name in list_presets():
        preset = PRESETS[name]
        result.append(
            {
                "name": name,
                "temperature": preset.temperature,
                "lsd_decode_steps": preset.lsd_decode_steps,
                "noise_clamp": preset.noise_clamp,
                "eos_threshold": preset.eos_threshold,
                "frames_after_eos": preset.frames_after_eos,
                "gain_db": preset.gain_db,
                "time_scale": preset.time_scale,
                "word_pause_ms": preset.word_pause_ms,
            }
        )
    return result


def merge_presets(
    base: GenerationPreset | None,
    override: GenerationPreset | None,
) -> GenerationPreset | None:
    if base is None:
        return override
    if override is None:
        return base
    return GenerationPreset(
        temperature=(
            override.temperature if override.temperature is not None else base.temperature
        ),
        lsd_decode_steps=(
            override.lsd_decode_steps
            if override.lsd_decode_steps is not None
            else base.lsd_decode_steps
        ),
        noise_clamp=override.noise_clamp if override.noise_clamp is not None else base.noise_clamp,
        eos_threshold=(
            override.eos_threshold if override.eos_threshold is not None else base.eos_threshold
        ),
        frames_after_eos=(
            override.frames_after_eos
            if override.frames_after_eos is not None
            else base.frames_after_eos
        ),
        gain_db=override.gain_db if override.gain_db is not None else base.gain_db,
        time_scale=override.time_scale if override.time_scale is not None else base.time_scale,
        word_pause_ms=(
            override.word_pause_ms
            if override.word_pause_ms is not None
            else base.word_pause_ms
        ),
    )


@contextmanager
def use_preset(model, preset: GenerationPreset | None):
    if preset is None:
        yield
        return
    previous = (model.temp, model.lsd_decode_steps, model.noise_clamp, model.eos_threshold)
    if preset.temperature is not None:
        model.temp = preset.temperature
    if preset.lsd_decode_steps is not None:
        model.lsd_decode_steps = preset.lsd_decode_steps
    if preset.noise_clamp is not None:
        model.noise_clamp = preset.noise_clamp
    if preset.eos_threshold is not None:
        model.eos_threshold = preset.eos_threshold
    try:
        yield
    finally:
        model.temp, model.lsd_decode_steps, model.noise_clamp, model.eos_threshold = previous
