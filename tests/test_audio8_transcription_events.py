import numpy as np

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.STT import audio8_handler
from speech_to_speech.STT.audio8_handler import Audio8STTHandler


def _handler():
    handler = object.__new__(Audio8STTHandler)
    handler.device = "cpu"
    handler.sample_rate = 16000
    handler._generate = lambda audio: "set the AC to twenty"
    return handler


def test_final_audio8_transcription(monkeypatch):
    monkeypatch.setattr(audio8_handler.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(audio8_handler.torch.mps, "empty_cache", lambda: None)

    result = list(
        _handler().process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="final",
                turn_id="turn_1",
                turn_revision=0,
                created_at_s=1.0,
            )
        )
    )
    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "set the AC to twenty"
    assert result[0].speech_stopped_at_s == 1.0


def test_progressive_audio8_partial(monkeypatch):
    monkeypatch.setattr(audio8_handler.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(audio8_handler.torch.mps, "empty_cache", lambda: None)

    result = list(
        _handler().process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="progressive",
                turn_id="turn_1",
                turn_revision=1,
            )
        )
    )
    assert isinstance(result[0], PartialTranscription)
    assert result[0].text == "set the AC to twenty"


def test_normalize_prediction_text_strips_control():
    from speech_to_speech.STT.audio8_handler import normalize_prediction_text

    raw = "<|text|>pay fifty to Mom<|im_end|>"
    assert normalize_prediction_text(raw) == "pay fifty to Mom"
