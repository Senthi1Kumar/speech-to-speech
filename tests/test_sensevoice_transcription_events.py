import numpy as np

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.STT import sensevoice_handler
from speech_to_speech.STT.sensevoice_handler import SenseVoiceSTTHandler


class _FakeSenseVoiceModel:
    def generate(self, *args, **kwargs):
        assert kwargs.get("language") == "en"
        assert kwargs.get("use_itn") is True
        return [{"text": "<|en|><|NEUTRAL|><|Speech|><|withitn|>set the AC to twenty"}]


def _handler():
    handler = object.__new__(SenseVoiceSTTHandler)
    handler.model = _FakeSenseVoiceModel()
    handler.device = "cpu"
    handler.language = "en"
    handler.use_itn = True
    handler._postprocess = lambda text: text.split("|>")[-1] if "|>" in text else text
    return handler


def test_final_sensevoice_strips_tags(monkeypatch):
    monkeypatch.setattr(sensevoice_handler.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(sensevoice_handler.torch.mps, "empty_cache", lambda: None)

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


def test_progressive_sensevoice_partial(monkeypatch):
    monkeypatch.setattr(sensevoice_handler.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(sensevoice_handler.torch.mps, "empty_cache", lambda: None)

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
