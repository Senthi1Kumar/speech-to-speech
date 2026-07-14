import numpy as np

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.STT import paraformer_handler
from speech_to_speech.STT.paraformer_handler import ParaformerSTTHandler


class _FakeParaformerModel:
    def generate(self, *args, **kwargs):
        return [{"text": " 今 天 天 气 不 错 "}]


def _handler(*, strip_spaces: bool = True, language: str = ""):
    handler = object.__new__(ParaformerSTTHandler)
    handler.model = _FakeParaformerModel()
    handler.device = "cpu"
    handler.language = language
    handler._strip_spaces = strip_spaces
    return handler


def test_progressive_paraformer_transcription_is_partial(monkeypatch):
    monkeypatch.setattr(paraformer_handler.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(paraformer_handler.torch.mps, "empty_cache", lambda: None)

    result = list(
        _handler().process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="progressive",
                turn_id="turn_1",
                turn_revision=2,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], PartialTranscription)
    assert result[0].text == "今天天气不错"
    assert result[0].turn_id == "turn_1"
    assert result[0].turn_revision == 2


def test_final_paraformer_transcription_is_final(monkeypatch):
    monkeypatch.setattr(paraformer_handler.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(paraformer_handler.torch.mps, "empty_cache", lambda: None)

    result = list(
        _handler().process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="final",
                turn_id="turn_1",
                turn_revision=2,
                created_at_s=123.0,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "今天天气不错"
    assert result[0].turn_id == "turn_1"
    assert result[0].turn_revision == 2
    assert result[0].speech_stopped_at_s == 123.0


def test_english_keeps_spaces(monkeypatch):
    monkeypatch.setattr(paraformer_handler.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(paraformer_handler.torch.mps, "empty_cache", lambda: None)

    class _EnModel:
        def generate(self, *args, **kwargs):
            assert kwargs.get("language") == "英文"
            return [{"text": "set the AC to twenty"}]

    handler = _handler(strip_spaces=False, language="英文")
    handler.model = _EnModel()
    result = list(
        handler.process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="final",
                turn_id="turn_1",
                turn_revision=0,
                created_at_s=1.0,
            )
        )
    )
    assert result[0].text == "set the AC to twenty"
