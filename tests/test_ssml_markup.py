import pytest

from pocket_tts.markup import BreakSegment, TextSegment, iter_silence_chunks, parse_ssml


def test_parse_ssml_plain_text():
    segments = parse_ssml("Hello world")
    assert segments == [TextSegment("Hello world", None, None, False)]


def test_parse_ssml_break_time():
    segments = parse_ssml("<speak>Hello<break time='200ms'/>world</speak>")
    assert isinstance(segments[0], TextSegment)
    assert isinstance(segments[1], BreakSegment)
    assert isinstance(segments[2], TextSegment)
    assert segments[0].text == "Hello"
    assert segments[1].duration_ms == 200
    assert segments[2].text == "world"


def test_parse_ssml_voice_emphasis_prosody():
    segments = parse_ssml(
        "<speak>"
        "<voice name='alba'>Hi</voice>"
        "<emphasis level='strong'> there</emphasis>"
        "<prosody rate='slow'> friend</prosody>"
        "</speak>"
    )
    assert segments[0] == TextSegment("Hi", "alba", None, False)
    assert segments[1] == TextSegment(" there", None, "expressive", True)
    assert segments[2] == TextSegment(" friend", None, "slow", True)


def test_parse_ssml_invalid_tag():
    with pytest.raises(ValueError, match="Unsupported SSML tag"):
        parse_ssml("<speak><foo>bar</foo></speak>")


def test_parse_ssml_invalid_break():
    with pytest.raises(ValueError, match="Invalid break time"):
        parse_ssml("<speak><break time='abc'/></speak>")


def test_iter_silence_chunks_length():
    chunks = list(iter_silence_chunks(24000, 500, chunk_ms=200))
    assert sum(chunk.numel() for chunk in chunks) == 12000


def test_parse_ssml_prosody_preset_attr():
    segments = parse_ssml("<speak><prosody preset='broadcast'>Hello</prosody></speak>")
    assert segments == [TextSegment("Hello", None, "broadcast", True)]


def test_parse_ssml_tolerates_whitespace_in_tag_prefix():
    segments = parse_ssml("<speak>< voice name='alba'>Hi</\nvoice></speak>")
    assert segments == [TextSegment("Hi", "alba", None, False)]
