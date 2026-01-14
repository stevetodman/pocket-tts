# Generate Command Documentation

The `generate` command allows you to generate speech from text directly from the command line using Kyutai Pocket TTS.

## Basic Usage

```bash
uvx pocket-tts generate
# or if installed manually:
pocket-tts generate
```

This will generate a WAV file `./tts_output.wav` with the default text and voice.

## Command Options

### Core Options

- `--text TEXT`: Text to generate (default: "Hello world! I am Kyutai Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me.")
- `--voice VOICE`: Path to audio conditioning file (voice to clone) (default: "hf://kyutai/tts-voices/alba-mackenna/casual.wav"). Urls and local paths are supported.
- `--output-path OUTPUT_PATH`: Output path for generated audio (default: "./tts_output.wav")
- `--ssml`: Interpret `--text` as SSML-lite markup (supports `<voice>`, `<break>`, `<prosody>`, and `<emphasis>`).
- `--preset PRESET`: Sampling preset to apply (examples: `broadcast`, `calm`, `expressive`, `dramatic`, `slow`, `fast`).

### Generation Parameters

- `--variant VARIANT`: Model signature (default: "b6369a24")
- `--lsd-decode-steps LSD_DECODE_STEPS`: Number of generation steps (default: 1)
- `--temperature TEMPERATURE`: Temperature for generation (default: 0.7)
- `--noise-clamp NOISE_CLAMP`: Noise clamp value (default: None)
- `--eos-threshold EOS_THRESHOLD`: EOS threshold (default: -4.0)
- `--frames-after-eos FRAMES_AFTER_EOS`: Number of frames to generate after EOS (default: None, auto-calculated based on the text length). Each frame is 80ms.

### Performance Options

- `--device DEVICE`: Device to use (default: "cpu", you may not get a speedup by using a gpu since it's a small model)
- `--quiet`, `-q`: Disable logging output

## Examples

### Basic Generation

```bash
# Generate with default settings
pocket-tts generate

# Custom text
pocket-tts generate --text "Hello, this is a custom message."

# Custom output path
pocket-tts generate --output-path "./my_audio.wav"
```

### Voice Selection

```bash
# Use different voice from HuggingFace
pocket-tts generate --voice "hf://kyutai/tts-voices/jessica-jian/casual.wav"

# Use local voice file
pocket-tts generate --voice "./my_voice.wav"
```

### SSML-lite Markup

```bash
pocket-tts generate --ssml --text '<speak>Hello <break time="300ms"/> world.</speak>'
pocket-tts generate --ssml --text '<speak><voice name="alba">Hi.</voice> <voice name="marius">Hello.</voice></speak>'
pocket-tts generate --ssml --text '<speak><emphasis level="strong">Big moment.</emphasis></speak>'
pocket-tts generate --ssml --text '<speak><prosody rate="slow">Take it easy.</prosody></speak>'
```

`<emphasis>` and `<prosody>` map to sampling presets. You can also set `<prosody preset="broadcast">` to pick a preset directly.
`rate="slow"` inserts short pauses between words and time-stretches the segment; `rate="fast"` shortens it. `<emphasis>` adds a small gain boost.
Segments under `<emphasis>` and `<prosody>` are RMS-normalized; time-stretching uses resampling and will shift pitch slightly.

### Quality Tuning

```bash
# Higher quality (more steps)
pocket-tts generate --lsd-decode-steps 5 --temperature 0.5

# More expressive (higher temperature)
pocket-tts generate --temperature 1.0

# Adjust EOS threshold, smaller means finishing earlier.
pocket-tts generate --eos-threshold -3.0
```

## Output Format

The generate command always outputs WAV files in the following format:
- **Sample Rate**: 24kHz
- **Channels**: Mono
- **Bit Depth**: 16-bit PCM
- **Format**: Standard WAV file

For more advanced usage, see the [Python API documentation](python-api.md) or consider using the [serve command](serve.md) for web-based generation and quick iteration.
