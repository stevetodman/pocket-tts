import hashlib
import logging
import os
import time
from pathlib import Path

import requests
import safetensors.torch
import torch
from huggingface_hub import hf_hub_download
from torch import nn

PROJECT_ROOT = Path(__file__).parent.parent.parent

_voices_names = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
PREDEFINED_VOICES = {
    # don't forget to change this
    x: f"hf://kyutai/pocket-tts-without-voice-cloning/embeddings/{x}.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"
    for x in _voices_names
}

_DEFAULT_DOWNLOAD_TIMEOUT_SEC = float(os.environ.get("POCKET_TTS_DOWNLOAD_TIMEOUT", "30"))
_DEFAULT_DOWNLOAD_RETRIES = int(os.environ.get("POCKET_TTS_DOWNLOAD_RETRIES", "3"))
_DEFAULT_MAX_DOWNLOAD_MB = int(os.environ.get("POCKET_TTS_MAX_DOWNLOAD_MB", "1024"))
_DEFAULT_MAX_DOWNLOAD_BYTES = _DEFAULT_MAX_DOWNLOAD_MB * 1024 * 1024
_DEFAULT_CONNECT_TIMEOUT_SEC = float(os.environ.get("POCKET_TTS_DOWNLOAD_CONNECT_TIMEOUT", "5"))


def make_cache_directory() -> Path:
    cache_dir = Path.home() / ".cache" / "pocket_tts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def print_nb_parameters(model: nn.Module, model_name: str):
    logger = logging.getLogger(__name__)
    state_dict = model.state_dict()
    total = 0
    for key, value in state_dict.items():
        logger.info("%s: %,d", key, value.numel())
        total += value.numel()
    logger.info("Total number of parameters in %s: %,d", model_name, total)


def size_of_dict(state_dict: dict) -> int:
    total_size = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total_size += value.numel() * value.element_size()
        elif isinstance(value, dict):
            total_size += size_of_dict(value)
    return total_size


class display_execution_time:
    def __init__(self, task_name: str, print_output: bool = True):
        self.task_name = task_name
        self.print_output = print_output
        self.start_time = None
        self.elapsed_time_ms = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        self.start_time = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.monotonic()
        self.elapsed_time_ms = int((end_time - self.start_time) * 1000)
        if self.print_output:
            self.logger.info("%s took %d ms", self.task_name, self.elapsed_time_ms)
        return False  # Don't suppress exceptions


def _download_http_with_retries(url: str, destination: Path) -> Path:
    max_bytes = _DEFAULT_MAX_DOWNLOAD_BYTES
    timeout = (_DEFAULT_CONNECT_TIMEOUT_SEC, _DEFAULT_DOWNLOAD_TIMEOUT_SEC)
    last_exc = None
    for attempt in range(_DEFAULT_DOWNLOAD_RETRIES):
        tmp_path = destination.with_suffix(destination.suffix + ".partial")
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_bytes:
                    raise ValueError(
                        f"Download size {content_length} exceeds limit {max_bytes} bytes for {url}"
                    )
                total = 0
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError(
                                f"Download exceeded limit {max_bytes} bytes for {url}"
                            )
                        f.write(chunk)
            tmp_path.replace(destination)
            return destination
        except Exception as exc:
            last_exc = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < _DEFAULT_DOWNLOAD_RETRIES - 1:
                time.sleep(0.5 * (attempt + 1))
    raise last_exc


def download_if_necessary(file_path: str) -> Path:
    if file_path.startswith("http://") or file_path.startswith("https://"):
        cache_dir = make_cache_directory()
        cached_file = cache_dir / (
            hashlib.sha256(file_path.encode()).hexdigest() + "." + file_path.split(".")[-1]
        )
        if not cached_file.exists():
            _download_http_with_retries(file_path, cached_file)
        return cached_file
    elif file_path.startswith("hf://"):
        file_path = file_path.removeprefix("hf://")
        splitted = file_path.split("/")
        repo_id = "/".join(splitted[:2])
        filename = "/".join(splitted[2:])
        if "@" in filename:
            filename, revision = filename.split("@")
        else:
            revision = None
        cached_file = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        return Path(cached_file)
    else:
        return Path(file_path)


def load_predefined_voice(voice_name: str) -> torch.Tensor:
    if voice_name not in PREDEFINED_VOICES:
        raise ValueError(
            f"Predefined voice '{voice_name}' not found"
            f", available voices are {list(PREDEFINED_VOICES)}."
        )
    voice_file = download_if_necessary(PREDEFINED_VOICES[voice_name])
    # There is only one tensor in the file.
    return safetensors.torch.load_file(voice_file)["audio_prompt"]
