from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np
import torch
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


class SenseVoiceSTTHandler(BaseSTTHandler):
    """
    FunASR SenseVoiceSmall STT.

    Silero VAD already segments turns in the s2s pipeline, so this handler loads
    SenseVoice without FunASR's fsmn-vad (avoids double-VAD and extra VRAM).
    Raw SenseVoice text includes emotion/event tags; those are stripped via
    rich_transcription_postprocess before yielding.
    """

    def setup(
        self,
        model_name: str = "FunAudioLLM/SenseVoiceSmall",
        device: str = "cuda",
        language: str = "en",
        use_itn: bool = True,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        del gen_kwargs  # unused; SenseVoice uses explicit generate kwargs
        self.device = device
        self.language = (language or "en").strip() or "en"
        self.use_itn = bool(use_itn)
        try:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SenseVoice STT requires funasr. "
                "Install with `pip install speech-to-speech[paraformer]` "
                "(or `uv add funasr`)."
            ) from exc

        self._postprocess = rich_transcription_postprocess
        # No FunASR vad_model: Silero already owns turn boundaries.
        self.model = AutoModel(
            model=model_name,
            device=device,
            hub="hf",
            disable_update=True,
            disable_pbar=True,
        )
        self.warmup()

    def _generate(self, audio: np.ndarray) -> str:
        result = self.model.generate(
            input=audio,
            cache={},
            language=self.language,
            use_itn=self.use_itn,
            batch_size=1,
        )
        raw = (result[0].get("text") or "").strip()
        return self._postprocess(raw).strip()

    def warmup(self) -> None:
        logger.info("Warming up %s", self.__class__.__name__)
        dummy = np.zeros(16000, dtype=np.float32)
        _ = self._generate(dummy)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering sensevoice...")
        pred_text = self._generate(vad_audio.audio)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("finished sensevoice inference")
        if not pred_text:
            logger.debug("no text detected. skipping...")
            return

        console.print(f"[yellow]USER: {pred_text}")

        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
        else:
            yield Transcription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )
