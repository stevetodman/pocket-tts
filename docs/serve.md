# Serve Command Documentation

The `serve` command starts a FastAPI web server that provides both a web interface and HTTP API for text-to-speech generation.

## Basic Usage

```bash
uvx pocket-tts serve
# or if installed manually:
pocket-tts serve
```

This starts a server on `http://localhost:8000` with the default voice model.

## Command Options

- `--voice VOICE`: Path to voice prompt audio file (voice to clone) (default: "hf://kyutai/tts-voices/alba-mackenna/casual.wav")
- `--host HOST`: Host to bind to (default: "localhost")
- `--port PORT`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development

## Examples

### Basic Server

```bash
# Start with default settings
pocket-tts serve

# Custom host and port
pocket-tts serve --host "localhost" --port 8080
```

### Custom Voice

```bash
# Use different voice
pocket-tts serve --default-voice "hf://kyutai/tts-voices/jessica-jian/casual.wav"

# Use local voice file
pocket-tts serve --default-voice "./my_voice.wav"
```

## Web Interface

Once the server is running, navigate to `http://localhost:8000` to access the web interface.

For more advanced usage, see the [Python API documentation](python-api.md) for direct integration with the TTS model.

## SSML-lite in the API

The `/tts` endpoint accepts an optional `ssml=true` form field to interpret `text` as SSML-lite.
Supported tags: `<voice>`, `<break>`, `<prosody>`, and `<emphasis>`.
You can also pass `preset` to apply a sampling preset.
`<prosody>` supports a `preset` attribute to set it directly.
`rate="slow"` inserts short pauses between words and time-stretches the segment. `<emphasis>` adds a small gain boost.
Segments under `<emphasis>` and `<prosody>` are RMS-normalized. Time-stretching uses resampling and will also shift pitch slightly.
For streaming responses, time-stretching is skipped to preserve low latency; use the CLI for full time-scale effects.

```bash
curl -X POST http://localhost:8000/tts \
  -F 'text=<speak>Hello <break time="250ms"/> there.</speak>' \
  -F 'ssml=true' \
  -F 'preset=broadcast' \
  --output tts_output.wav
```

## Metadata Endpoints

Use these to populate UIs or validate inputs:

```bash
curl http://localhost:8000/presets
curl http://localhost:8000/voices
curl http://localhost:8000/metadata
```
