import types

import torch
from torch import nn

from pocket_tts.models import tts_model as tts_model_module
from pocket_tts.presets import GenerationPreset


def _make_model() -> tts_model_module.TTSModel:
    dummy_flow = types.SimpleNamespace(conditioner=types.SimpleNamespace(tokenizer=None))
    dummy_config = types.SimpleNamespace(mimi=types.SimpleNamespace(sample_rate=24000))
    model = tts_model_module.TTSModel.__new__(tts_model_module.TTSModel)
    nn.Module.__init__(model)
    model.flow_lm = dummy_flow
    model.temp = 0.0
    model.lsd_decode_steps = 1
    model.noise_clamp = None
    model.eos_threshold = 0.0
    model.config = dummy_config
    model.has_voice_cloning = True
    return model


def test_generate_audio_stream_applies_continuation_between_chunks(monkeypatch):
    model = _make_model()
    monkeypatch.setattr(
        tts_model_module,
        "split_into_best_sentences",
        lambda _tokenizer, _text: ["First chunk.", "Second chunk."],
    )

    generated_chunks = []
    continuation_inputs = []

    def _fake_generate(
        self,
        model_state,
        text_to_generate,
        frames_after_eos,
        copy_state,
        audio_accumulator=None,
    ):
        value = float(len(generated_chunks) + 1)
        audio_chunk = torch.full((4,), value)
        generated_chunks.append(text_to_generate)
        if audio_accumulator is not None:
            audio_accumulator.append(audio_chunk)
        yield audio_chunk

    def _fake_apply(self, model_state, audio):
        continuation_inputs.append(audio)

    monkeypatch.setattr(
        tts_model_module.TTSModel, "_generate_audio_stream_short_text", _fake_generate
    )
    monkeypatch.setattr(tts_model_module.TTSModel, "_apply_audio_continuation", _fake_apply)

    state = {"seed": 1}
    chunks = list(model.generate_audio_stream(state, "Long text", copy_state=True))

    assert len(chunks) == 2
    assert len(continuation_inputs) == 1
    assert torch.equal(continuation_inputs[0], torch.full((4,), 1.0))


def test_generate_audio_stream_skips_continuation_for_single_chunk(monkeypatch):
    model = _make_model()
    monkeypatch.setattr(
        tts_model_module,
        "split_into_best_sentences",
        lambda _tokenizer, _text: ["Only one chunk."],
    )

    continuation_inputs = []

    def _fake_generate(
        self,
        model_state,
        text_to_generate,
        frames_after_eos,
        copy_state,
        audio_accumulator=None,
    ):
        audio_chunk = torch.zeros(4)
        if audio_accumulator is not None:
            audio_accumulator.append(audio_chunk)
        yield audio_chunk

    def _fake_apply(self, model_state, audio):
        continuation_inputs.append(audio)

    monkeypatch.setattr(
        tts_model_module.TTSModel, "_generate_audio_stream_short_text", _fake_generate
    )
    monkeypatch.setattr(tts_model_module.TTSModel, "_apply_audio_continuation", _fake_apply)

    state = {"seed": 1}
    chunks = list(model.generate_audio_stream(state, "Short text", copy_state=True))

    assert len(chunks) == 1
    assert continuation_inputs == []


def test_voice_prompt_cache_respects_env(monkeypatch):
    model = _make_model()
    calls = {"count": 0}

    def _fake_get_state(prompt, truncate=False):
        calls["count"] += 1
        return {"prompt": str(prompt), "truncate": truncate}

    model.get_state_for_audio_prompt = _fake_get_state
    monkeypatch.setenv("POCKET_TTS_VOICE_CACHE_SIZE", "1")
    model._init_voice_prompt_cache()

    model._cached_get_state_for_audio_prompt("voice-a", truncate=False)
    model._cached_get_state_for_audio_prompt("voice-b", truncate=False)
    model._cached_get_state_for_audio_prompt("voice-a", truncate=False)

    assert calls["count"] == 3


def test_voice_prompt_cache_skips_tensor_inputs(monkeypatch):
    model = _make_model()
    calls = {"count": 0}

    def _fake_get_state(prompt, truncate=False):
        calls["count"] += 1
        return {"prompt": prompt, "truncate": truncate}

    model.get_state_for_audio_prompt = _fake_get_state
    monkeypatch.setenv("POCKET_TTS_VOICE_CACHE_SIZE", "2")
    model._init_voice_prompt_cache()

    prompt = torch.zeros(1)
    model._cached_get_state_for_audio_prompt(prompt, truncate=False)
    model._cached_get_state_for_audio_prompt(prompt, truncate=False)

    assert calls["count"] == 2


def test_generate_audio_from_text_resolves_voice(monkeypatch):
    model = _make_model()
    calls = {}

    def _fake_get_state(prompt, truncate=False):
        calls["voice"] = prompt
        calls["truncate"] = truncate
        return {"state": 1}

    def _fake_generate(model_state, text_to_generate, frames_after_eos=None, copy_state=True):
        calls["text"] = text_to_generate
        calls["state"] = model_state
        return torch.ones(3)

    model.get_state_for_audio_prompt = _fake_get_state
    model.generate_audio = _fake_generate

    audio = model.generate_audio_from_text(
        "Hello there",
        voice="voice.wav",
        preset=GenerationPreset(temperature=0.5),
        truncate_voice=True,
    )

    assert torch.equal(audio, torch.ones(3))
    assert calls["voice"] == "voice.wav"
    assert calls["truncate"] is True
    assert calls["text"] == "Hello there"
    assert calls["state"] == {"state": 1}


def test_generate_audio_from_text_accepts_state():
    model = _make_model()

    def _fake_generate(model_state, text_to_generate, frames_after_eos=None, copy_state=True):
        return torch.full((2,), 2.0)

    model.generate_audio = _fake_generate

    audio = model.generate_audio_from_text("Hello", voice={"state": 2})
    assert torch.equal(audio, torch.full((2,), 2.0))


def test_generate_audio_stream_from_text_uses_stream(monkeypatch):
    model = _make_model()
    called = {"count": 0}

    def _fake_get_state(prompt, truncate=False):
        return {"state": 3}

    def _fake_stream(model_state, text_to_generate, frames_after_eos=None, copy_state=True):
        called["count"] += 1
        yield torch.tensor([1.0])

    model.get_state_for_audio_prompt = _fake_get_state
    model.generate_audio_stream = _fake_stream

    chunks = list(model.generate_audio_stream_from_text("Hello", voice="voice.wav"))
    assert called["count"] == 1
    assert torch.equal(chunks[0], torch.tensor([1.0]))
