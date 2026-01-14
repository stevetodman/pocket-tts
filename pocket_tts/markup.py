import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import torch

_BREAK_STRENGTH_MS = {
    "x-weak": 100,
    "weak": 200,
    "medium": 400,
    "strong": 800,
    "x-strong": 1200,
}
_DEFAULT_BREAK_MS = 400
_MAX_BREAK_MS = 10000


@dataclass(frozen=True)
class TextSegment:
    text: str
    voice: str | None
    preset: str | None
    normalize: bool


@dataclass(frozen=True)
class BreakSegment:
    duration_ms: int


def _strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _parse_break_ms(attrs: dict[str, str]) -> int:
    if "time" in attrs:
        value = attrs["time"].strip().lower()
        match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*(ms|s)?", value)
        if not match:
            raise ValueError(f"Invalid break time '{attrs['time']}'")
        amount = float(match.group(1))
        unit = match.group(2) or "ms"
        duration_ms = amount * (1000 if unit == "s" else 1)
    elif "strength" in attrs:
        strength = attrs["strength"].strip().lower()
        if strength not in _BREAK_STRENGTH_MS:
            raise ValueError(f"Invalid break strength '{attrs['strength']}'")
        duration_ms = _BREAK_STRENGTH_MS[strength]
    else:
        duration_ms = _DEFAULT_BREAK_MS

    duration_ms = max(0, min(int(duration_ms), _MAX_BREAK_MS))
    return duration_ms


def _parse_emphasis_preset(attrs: dict[str, str]) -> str | None:
    level = attrs.get("level") or attrs.get("strength")
    if not level:
        return None
    level = level.strip().lower()
    mapping = {
        "reduced": "calm",
        "moderate": "broadcast",
        "strong": "expressive",
        "x-strong": "dramatic",
        "none": None,
    }
    if level not in mapping:
        raise ValueError(f"Invalid emphasis level '{level}'")
    return mapping[level]


def _parse_prosody_preset(attrs: dict[str, str]) -> str | None:
    preset = attrs.get("preset")
    if preset is not None:
        preset = preset.strip()
        return preset or None
    rate = attrs.get("rate")
    if not rate:
        return None
    rate = rate.strip().lower()
    mapping = {
        "x-slow": "slow",
        "slow": "slow",
        "medium": None,
        "default": None,
        "fast": "fast",
        "x-fast": "fast",
    }
    if rate in mapping:
        return mapping[rate]
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*%?", rate)
    if match:
        value = float(match.group(1))
        if value <= 90:
            return "slow"
        if value >= 110:
            return "fast"
        return None
    raise ValueError(f"Invalid prosody rate '{rate}'")


def _normalize_text(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ")


_TAG_PREFIX_WHITESPACE_RE = re.compile(r"<\s*(/?)\s*([A-Za-z_][\w:.-]*)")


def _normalize_ssml_markup(xml_input: str) -> str:
    """Fix common copy/paste issues to help the strict XML parser.

    XML does not allow whitespace between "<" / "</" and the tag name (e.g. "</\nvoice>").
    Some shells allow multi-line strings inside quotes, which can accidentally introduce
    such whitespace when a long SSML line is split across lines.
    """

    return _TAG_PREFIX_WHITESPACE_RE.sub(r"<\1\2", xml_input)


def _append_text(
    segments: list,
    text: str,
    voice: str | None,
    preset: str | None,
    normalize: bool,
) -> None:
    normalized = _normalize_text(text)
    if not normalized.strip():
        return
    if (
        segments
        and isinstance(segments[-1], TextSegment)
        and segments[-1].voice == voice
        and segments[-1].preset == preset
        and segments[-1].normalize == normalize
    ):
        segments[-1] = TextSegment(segments[-1].text + normalized, voice, preset, normalize)
    else:
        segments.append(TextSegment(normalized, voice, preset, normalize))


def parse_ssml(text: str) -> list[TextSegment | BreakSegment]:
    """Parse a minimal SSML-like markup into text and break segments."""
    content = text.strip()
    if not content:
        raise ValueError("SSML input is empty")

    if re.match(r"^<\s*speak[\s>]", content):
        xml_input = content
    else:
        xml_input = f"<speak>{content}</speak>"

    xml_input = _normalize_ssml_markup(xml_input)

    try:
        root = ET.fromstring(xml_input)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid SSML: {exc}") from exc

    segments: list[TextSegment | BreakSegment] = []

    def walk(
        node: ET.Element,
        voice: str | None,
        preset: str | None,
        normalize: bool,
    ) -> None:
        tag = _strip_namespace(node.tag)
        if tag not in {"speak", "p", "s", "prosody", "emphasis", "voice"}:
            raise ValueError(f"Unsupported SSML tag '{tag}'")

        if node.text:
            _append_text(segments, node.text, voice, preset, normalize)

        for child in node:
            child_tag = _strip_namespace(child.tag)
            if child_tag == "voice":
                voice_attr = (
                    child.attrib.get("name")
                    or child.attrib.get("voice")
                    or child.attrib.get("src")
                    or child.attrib.get("url")
                )
                if not voice_attr:
                    raise ValueError("SSML <voice> tag requires a name/src/url attribute")
                walk(child, voice_attr, preset, normalize)
            elif child_tag == "prosody":
                next_preset = _parse_prosody_preset(child.attrib) or preset
                walk(child, voice, next_preset, True)
            elif child_tag == "emphasis":
                next_preset = _parse_emphasis_preset(child.attrib) or preset
                walk(child, voice, next_preset, True)
            elif child_tag == "break":
                duration_ms = _parse_break_ms(child.attrib)
                if duration_ms > 0:
                    segments.append(BreakSegment(duration_ms))
            elif child_tag in {"speak", "p", "s"}:
                walk(child, voice, preset, normalize)
            else:
                raise ValueError(f"Unsupported SSML tag '{child_tag}'")

            if child.tail:
                _append_text(segments, child.tail, voice, preset, normalize)

    walk(root, None, None, False)
    return segments


def iter_silence_chunks(sample_rate: int, duration_ms: int, chunk_ms: int = 200):
    total_samples = int(sample_rate * duration_ms / 1000)
    if total_samples <= 0:
        return
    chunk_samples = max(1, int(sample_rate * chunk_ms / 1000))
    remaining = total_samples
    while remaining > 0:
        size = min(chunk_samples, remaining)
        yield torch.zeros(size, dtype=torch.float32)
        remaining -= size
