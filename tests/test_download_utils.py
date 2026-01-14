from pathlib import Path

import pytest

from pocket_tts.utils import utils as utils_module


class _DummyResponse:
    def __init__(self, content: bytes, headers: dict[str, str] | None = None):
        self._content = content
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1024):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_rejects_large_content_length(monkeypatch, tmp_path):
    monkeypatch.setattr(utils_module, "_DEFAULT_MAX_DOWNLOAD_BYTES", 4)
    monkeypatch.setattr(utils_module, "_DEFAULT_DOWNLOAD_RETRIES", 1)
    monkeypatch.setattr(utils_module, "make_cache_directory", lambda: tmp_path)

    def _fake_get(url, stream=True, timeout=None):
        return _DummyResponse(b"abcd", headers={"Content-Length": "10"})

    monkeypatch.setattr(utils_module.requests, "get", _fake_get)

    with pytest.raises(ValueError, match="exceeds limit"):
        utils_module.download_if_necessary("https://example.com/voice.wav")


def test_download_streams_to_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(utils_module, "_DEFAULT_MAX_DOWNLOAD_BYTES", 10)
    monkeypatch.setattr(utils_module, "_DEFAULT_DOWNLOAD_RETRIES", 1)
    monkeypatch.setattr(utils_module, "make_cache_directory", lambda: tmp_path)

    def _fake_get(url, stream=True, timeout=None):
        return _DummyResponse(b"data", headers={"Content-Length": "4"})

    monkeypatch.setattr(utils_module.requests, "get", _fake_get)

    cached = utils_module.download_if_necessary("https://example.com/voice.wav")
    assert isinstance(cached, Path)
    assert cached.exists()
    assert cached.read_bytes() == b"data"
