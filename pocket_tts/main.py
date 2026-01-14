import io
import logging
import os
import re
import tempfile
import threading
import wave
from pathlib import Path
from queue import Queue

import torch
import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from typing_extensions import Annotated

from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.data.audio_utils import apply_gain_db, normalize_rms, time_stretch_audio
from pocket_tts.default_parameters import (
    DEFAULT_AUDIO_PROMPT,
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_FRAMES_AFTER_EOS,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_PRESET,
    DEFAULT_TEMPERATURE,
    DEFAULT_VARIANT,
)
from pocket_tts.markup import BreakSegment, TextSegment, iter_silence_chunks, parse_ssml
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.presets import (
    GenerationPreset,
    get_preset,
    list_presets,
    list_presets_data,
    merge_presets,
    use_preset,
)
from pocket_tts.utils.logging_utils import enable_logging
from pocket_tts.utils.utils import PREDEFINED_VOICES, size_of_dict

logger = logging.getLogger(__name__)

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool", pretty_exceptions_show_locals=False
)
PRESET_HELP = (
    f"Preset to use (default: {DEFAULT_PRESET}). Available: {', '.join(list_presets())}"
)
_MAX_VOICE_BYTES = int(os.environ.get("POCKET_TTS_MAX_VOICE_BYTES", str(20 * 1024 * 1024)))
_MAX_VOICE_SECONDS = float(os.environ.get("POCKET_TTS_MAX_VOICE_SECONDS", "30"))


# ------------------------------------------------------
# The pocket-tts server implementation
# ------------------------------------------------------

# Global model instance
tts_model = None
global_model_state = None
global_default_voice = None
tts_model_lock = threading.Lock()

web_app = FastAPI(
    title="Kyutai Pocket TTS API", description="Text-to-Speech generation API", version="1.0.0"
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://pod1-10007.internal.kyutai.org",
        "https://kyutai.org",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/")
async def root():
    """Serve the frontend."""
    static_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(static_path)


@web_app.get("/health")
async def health():
    return {"status": "healthy"}


@web_app.get("/presets")
async def presets():
    return {"presets": list_presets_data(), "default": DEFAULT_PRESET}


@web_app.get("/voices")
async def voices():
    voices_list = [
        {"name": name, "url": PREDEFINED_VOICES[name]} for name in sorted(PREDEFINED_VOICES)
    ]
    return {"voices": voices_list, "default": DEFAULT_AUDIO_PROMPT}


@web_app.get("/metadata")
async def metadata():
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {
        "sample_rate": tts_model.sample_rate,
        "device": tts_model.device,
        "has_voice_cloning": tts_model.has_voice_cloning,
    }


def _write_chunks_to_queue(queue, audio_chunks):
    """Allows writing to the StreamingResponse as if it were a file."""

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue):
            self.queue = queue

        def write(self, data):
            self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    stream_audio_chunks(FileLikeToQueue(queue), audio_chunks, tts_model.config.mimi.sample_rate)


def _generate_data_from_chunks(audio_chunks):
    queue = Queue()

    # Run your function in a thread
    thread = threading.Thread(target=_write_chunks_to_queue, args=(queue, audio_chunks))
    thread.start()

    # Yield data as it becomes available
    i = 0
    while True:
        data = queue.get()
        if data is None:
            break
        i += 1
        yield data

    thread.join()


def _iter_audio_chunks(audio: torch.Tensor, sample_rate: int, chunk_ms: int = 200):
    if audio.numel() == 0:
        return
    chunk_samples = max(1, int(sample_rate * chunk_ms / 1000))
    for start in range(0, audio.shape[0], chunk_samples):
        yield audio[start : start + chunk_samples]


def _iter_gain_chunks(audio_chunks, gain_db: float | None):
    if gain_db is None or gain_db == 0:
        yield from audio_chunks
        return
    factor = 10 ** (gain_db / 20)
    for chunk in audio_chunks:
        yield (chunk * factor).clamp(-1, 1)


def _apply_preset_effects(
    audio_chunks,
    preset: GenerationPreset | None,
    sample_rate: int,
    normalize: bool = False,
    streaming: bool = True,
):
    needs_buffer = normalize or (preset is not None and preset.time_scale not in (None, 1.0))
    if not needs_buffer:
        yield from _iter_gain_chunks(audio_chunks, preset.gain_db if preset else None)
        return

    if streaming:
        if preset is not None and preset.time_scale not in (None, 1.0):
            logger.warning("Skipping time_scale in streaming mode to preserve low latency.")
        for chunk in audio_chunks:
            if normalize:
                chunk = normalize_rms(chunk)
            if preset is not None and preset.gain_db is not None:
                chunk = apply_gain_db(chunk, preset.gain_db)
            yield chunk
        return

    chunks = list(audio_chunks)
    if not chunks:
        return
    audio = torch.cat(chunks, dim=0)
    if preset is not None and preset.time_scale not in (None, 1.0):
        audio = time_stretch_audio(audio, preset.time_scale)
    if normalize:
        audio = normalize_rms(audio)
    if preset is not None and preset.gain_db is not None:
        audio = apply_gain_db(audio, preset.gain_db)
    yield from _iter_audio_chunks(audio, sample_rate)


def _split_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def generate_data_with_state(
    text_to_generate: str,
    model_state: dict,
    preset: GenerationPreset | None = None,
    frames_after_eos: int | None = None,
):
    frames_override = (
        preset.frames_after_eos
        if preset is not None and preset.frames_after_eos is not None
        else frames_after_eos
    )

    def iterator():
        with tts_model_lock:
            with use_preset(tts_model, preset):
                raw_chunks = tts_model.generate_audio_stream(
                    model_state=model_state,
                    text_to_generate=text_to_generate,
                    frames_after_eos=frames_override,
                )
                yield from _apply_preset_effects(
                    raw_chunks,
                    preset,
                    tts_model.sample_rate,
                    streaming=True,
                )

    yield from _generate_data_from_chunks(iterator())


def _iter_audio_from_segments(
    model: TTSModel,
    segments: list[TextSegment | BreakSegment],
    default_state: dict,
    default_voice: str | None,
    default_preset: GenerationPreset | None,
    frames_after_eos: int | None,
    truncate_voice: bool,
):
    voice_state_cache: dict[str, dict] = {}

    def resolve_state(voice: str | None) -> dict:
        if not voice or (default_voice is not None and voice == default_voice):
            return default_state
        if voice in voice_state_cache:
            return voice_state_cache[voice]
        voice_state = model.get_state_for_audio_prompt(voice, truncate=truncate_voice)
        voice_state_cache[voice] = voice_state
        return voice_state

    def resolve_preset(preset_name: str | None) -> tuple[GenerationPreset | None, bool]:
        if preset_name is None:
            return default_preset, False
        return merge_presets(default_preset, get_preset(preset_name)), True

    with tts_model_lock:
        for segment in segments:
            if isinstance(segment, BreakSegment):
                yield from iter_silence_chunks(model.sample_rate, segment.duration_ms)
                continue
            state = resolve_state(segment.voice)
            preset, is_segment_override = resolve_preset(segment.preset)
            frames_override = (
                preset.frames_after_eos
                if preset is not None and preset.frames_after_eos is not None
                else frames_after_eos
            )

            def generate_with_effects(text: str):
                with use_preset(model, preset):
                    raw_chunks = model.generate_audio_stream(
                        model_state=state,
                        text_to_generate=text,
                        frames_after_eos=frames_override,
                    )
                    yield from _apply_preset_effects(
                        raw_chunks,
                        preset,
                        model.sample_rate,
                        normalize=segment.normalize,
                        streaming=True,
                    )

            apply_word_pause = (
                is_segment_override
                and preset is not None
                and preset.word_pause_ms is not None
                and preset.word_pause_ms > 0
            )
            if apply_word_pause:
                words = _split_words(segment.text)
                for index, word in enumerate(words):
                    yield from generate_with_effects(word)
                    if index < len(words) - 1:
                        yield from iter_silence_chunks(model.sample_rate, preset.word_pause_ms)
            else:
                yield from generate_with_effects(segment.text)


def _validate_segment_presets(segments: list[TextSegment | BreakSegment]) -> None:
    for segment in segments:
        if isinstance(segment, TextSegment) and segment.preset is not None:
            get_preset(segment.preset)


def _validate_voice_wav(content: bytes) -> None:
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded voice file is empty")
    if len(content) > _MAX_VOICE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded voice file exceeds {_MAX_VOICE_BYTES} bytes",
        )
    if len(content) < 12 or content[:4] != b"RIFF" or content[8:12] != b"WAVE":
        raise HTTPException(status_code=400, detail="Uploaded voice file must be a WAV file")
    try:
        with wave.open(io.BytesIO(content), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
    except wave.Error as exc:
        raise HTTPException(status_code=400, detail="Uploaded voice file is not a valid WAV") from exc
    if sample_rate <= 0:
        raise HTTPException(status_code=400, detail="Uploaded voice file has invalid sample rate")
    duration = frames / sample_rate
    if duration > _MAX_VOICE_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded voice file exceeds {_MAX_VOICE_SECONDS} seconds",
        )


@web_app.post("/tts")
def text_to_speech(
    text: str = Form(...),
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
    ssml: bool = Form(False),
    preset: str | None = Form(None),
):
    """
    Generate speech from text using the pre-loaded voice prompt or a custom voice.

    Args:
        text: Text to convert to speech
        voice_url: Optional voice URL (http://, https://, or hf://)
        voice_wav: Optional uploaded voice file (mutually exclusive with voice_url)
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if voice_url is not None and voice_wav is not None:
        raise HTTPException(status_code=400, detail="Cannot provide both voice_url and voice_wav")

    # Use the appropriate model state
    default_voice = None
    default_preset = get_preset(DEFAULT_PRESET)
    if preset is not None:
        try:
            default_preset = get_preset(preset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if voice_url is not None:
        if not (
            voice_url.startswith("http://")
            or voice_url.startswith("https://")
            or voice_url.startswith("hf://")
            or voice_url in PREDEFINED_VOICES
        ):
            raise HTTPException(
                status_code=400, detail="voice_url must start with http://, https://, or hf://"
            )
        with tts_model_lock:
            model_state = tts_model._cached_get_state_for_audio_prompt(voice_url, truncate=True)
        default_voice = voice_url
        logging.warning("Using voice from URL: %s", voice_url)
    elif voice_wav is not None:
        # Use uploaded voice file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = voice_wav.file.read()
            _validate_voice_wav(content)
            temp_file.write(content)
            temp_file.flush()

            try:
                with tts_model_lock:
                    model_state = tts_model.get_state_for_audio_prompt(
                        Path(temp_file.name), truncate=True
                    )
            finally:
                os.unlink(temp_file.name)
    else:
        # Use default global model state
        model_state = global_model_state
        default_voice = global_default_voice

    if ssml:
        try:
            segments = parse_ssml(text)
            _validate_segment_presets(segments)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        audio_chunks = _iter_audio_from_segments(
            model=tts_model,
            segments=segments,
            default_state=model_state,
            default_voice=default_voice,
            default_preset=default_preset,
            frames_after_eos=None,
            truncate_voice=True,
        )
        return StreamingResponse(
            _generate_data_from_chunks(audio_chunks),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=generated_speech.wav",
                "Transfer-Encoding": "chunked",
            },
        )

    return StreamingResponse(
        generate_data_with_state(text, model_state, default_preset),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


@cli_app.command()
def serve(
    voice: Annotated[
        str, typer.Option(help="Path to voice prompt audio file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
):
    """Start the FastAPI server."""

    global tts_model, global_model_state, global_default_voice
    tts_model = TTSModel.load_model(DEFAULT_VARIANT)

    # Pre-load the voice prompt
    global_model_state = tts_model.get_state_for_audio_prompt(voice)
    global_default_voice = voice
    logger.info(f"The size of the model state is {size_of_dict(global_model_state) // 1e6} MB")

    uvicorn.run("pocket_tts.main:web_app", host=host, port=port, reload=reload)


# ------------------------------------------------------
# The pocket-tts single generation CLI implementation
# ------------------------------------------------------


@cli_app.command()
def generate(
    text: Annotated[
        str, typer.Option(help="Text to generate")
    ] = "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me.",
    voice: Annotated[
        str, typer.Option(help="Path to audio conditioning file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    variant: Annotated[str, typer.Option(help="Model signature")] = DEFAULT_VARIANT,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Temperature for generation")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS threshold")] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    preset: Annotated[
        str | None,
        typer.Option(help=PRESET_HELP),
    ] = None,
    ssml: Annotated[bool, typer.Option(help="Interpret text as SSML-lite markup")] = False,
    output_path: Annotated[
        str, typer.Option(help="Output path for generated audio")
    ] = "./tts_output.wav",
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
):
    """Generate speech using Kyutai Pocket TTS."""
    if "cuda" in device:
        # Cuda graphs capturing does not play nice with multithreading.
        os.environ["NO_CUDA_GRAPH"] = "1"

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(
            variant, temperature, lsd_decode_steps, noise_clamp, eos_threshold
        )
        tts_model.to(device)

        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)
        if preset is None:
            preset_value = get_preset(DEFAULT_PRESET)
        else:
            try:
                preset_value = get_preset(preset)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
        frames_override = (
            preset_value.frames_after_eos
            if preset_value is not None and preset_value.frames_after_eos is not None
            else frames_after_eos
        )
        if ssml:
            try:
                segments = parse_ssml(text)
                _validate_segment_presets(segments)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            audio_chunks = _iter_audio_from_segments(
                model=tts_model,
                segments=segments,
                default_state=model_state_for_voice,
                default_voice=voice,
                default_preset=preset_value,
                frames_after_eos=frames_override,
                truncate_voice=False,
            )
        else:
            # Stream audio generation directly to file or stdout
            def iterator():
                with use_preset(tts_model, preset_value):
                    raw_chunks = tts_model.generate_audio_stream(
                        model_state=model_state_for_voice,
                        text_to_generate=text,
                        frames_after_eos=frames_override,
                    )
                    yield from _apply_preset_effects(
                        raw_chunks,
                        preset_value,
                        tts_model.sample_rate,
                        streaming=False,
                    )

            audio_chunks = iterator()

        stream_audio_chunks(output_path, audio_chunks, tts_model.config.mimi.sample_rate)

        # Only print the result message if not writing to stdout
        if output_path != "-":
            logger.info("Results written in %s", output_path)
        logger.info(
            "If you want to try multiple voices and prompts quickly, try the `serve` command."
        )


if __name__ == "__main__":
    cli_app()
