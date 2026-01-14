# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- SSML-lite and preset support in the server UI and docs, with `/presets`, `/voices`, and new `/metadata` endpoint for clients.
- Convenience Python helpers: `generate_audio_from_text` and `generate_audio_stream_from_text`.
- Audio streaming effect safeguards and tests covering SSML, API validation, streaming behavior, and download safeguards.

### Changed
- Streaming responses now skip time-stretching to preserve low latency; CLI/file output keeps full preset effects.
- Long text generation carries conditioning across chunks for smoother continuity.
- Voice prompt handling enforces mono shape, non-empty input, and improved WAV parsing.
- Download logic now streams, retries, and enforces size caps (HTTP downloads).

### Fixed
- Concurrency safety for server-side generation through request serialization.
- Device/dtype initialization for streaming convolution and attention state buffers.
- Audio read now handles 8/16/32-bit PCM with channel validation.

### Security
- Upload validation for `/tts` now rejects empty, oversized, or invalid WAV files and enforces max duration.

### Configuration
- New env knobs: `POCKET_TTS_TORCH_THREADS`, `POCKET_TTS_VOICE_CACHE_SIZE`,
  `POCKET_TTS_MAX_VOICE_BYTES`, `POCKET_TTS_MAX_VOICE_SECONDS`,
  `POCKET_TTS_MAX_DOWNLOAD_MB`, `POCKET_TTS_DOWNLOAD_TIMEOUT`,
  `POCKET_TTS_DOWNLOAD_RETRIES`, `POCKET_TTS_DOWNLOAD_CONNECT_TIMEOUT`.
