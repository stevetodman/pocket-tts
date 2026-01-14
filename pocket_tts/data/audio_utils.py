"""Various utilities for audio conversion (PCM format, sample rate and channels),
and volume normalization."""

import math
from fractions import Fraction

import torch
from scipy.signal import resample_poly


def _coerce_sample_rate(rate: int | float, name: str) -> int:
    if isinstance(rate, bool):
        raise TypeError(f"{name} must be an integer sample rate, got bool")
    if isinstance(rate, float):
        if not rate.is_integer():
            raise ValueError(f"{name} must be an integer sample rate, got {rate}")
        rate = int(rate)
    elif isinstance(rate, int):
        rate = int(rate)
    else:
        raise TypeError(f"{name} must be an integer sample rate, got {type(rate).__name__}")
    if rate <= 0:
        raise ValueError(f"{name} must be positive, got {rate}")
    return rate


def convert_audio(
    wav: torch.Tensor, from_rate: int | float, to_rate: int | float, to_channels: int
) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels."""
    if wav.ndim != 2:
        raise ValueError("Audio tensor must have shape [channels, samples]")
    from_rate = _coerce_sample_rate(from_rate, "from_rate")
    to_rate = _coerce_sample_rate(to_rate, "to_rate")
    if from_rate != to_rate:
        # Convert to numpy for scipy resampling
        wav_np = wav.detach().cpu().numpy()

        # Calculate resampling parameters
        gcd = math.gcd(from_rate, to_rate)
        up = to_rate // gcd
        down = from_rate // gcd

        # Resample using scipy
        resampled_np = resample_poly(wav_np, up, down, axis=-1)

        # Convert back to torch tensor
        wav = torch.from_numpy(resampled_np).to(wav.device).to(wav.dtype)

    channels = wav.shape[-2]
    if channels != to_channels:
        if to_channels == 1 and channels > 1:
            wav = wav.mean(dim=-2, keepdim=True)
        elif to_channels > 1 and channels == 1:
            wav = wav.repeat(to_channels, 1)
        else:
            raise ValueError(f"Expected {to_channels} channel(s), got {channels}")
    return wav


def time_stretch_audio(wav: torch.Tensor, time_scale: float) -> torch.Tensor:
    """Stretch audio in time. time_scale > 1.0 makes audio longer (slower)."""
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")
    if time_scale == 1.0:
        return wav

    fraction = Fraction(time_scale).limit_denominator(100)
    up, down = fraction.numerator, fraction.denominator

    wav_np = wav.detach().cpu().numpy()
    resampled_np = resample_poly(wav_np, up, down, axis=-1)
    return torch.from_numpy(resampled_np).to(wav.device).to(wav.dtype)


def apply_gain_db(wav: torch.Tensor, gain_db: float) -> torch.Tensor:
    if gain_db == 0:
        return wav
    factor = 10 ** (gain_db / 20)
    return (wav * factor).clamp(-1, 1)


def normalize_rms(
    wav: torch.Tensor,
    target_db: float = -20.0,
    max_gain_db: float = 12.0,
) -> torch.Tensor:
    if wav.numel() == 0:
        return wav
    rms = torch.sqrt(torch.mean(wav.float().pow(2)))
    if rms <= 0:
        return wav
    target_rms = 10 ** (target_db / 20)
    max_gain = 10 ** (max_gain_db / 20)
    gain = min(target_rms / rms.item(), max_gain)
    return (wav * gain).clamp(-1, 1)
