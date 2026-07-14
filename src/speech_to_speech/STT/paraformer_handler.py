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


def _is_fun_asr_nano(model_name: str) -> bool:
    name = model_name.lower()
    return "fun-asr-nano" in name or "funasr-nano" in name


class ParaformerSTTHandler(BaseSTTHandler):
    """
    FunASR STT handler (Paraformer aliases, Fun-ASR-Nano, etc.).
    Default model is Chinese-oriented; Nova passes Fun-ASR-Nano + language=英文.
    """

    def setup(
        self,
        model_name: str = "paraformer-zh",
        device: str = "cuda",
        language: str = "",
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        print(model_name)
        # Keep org/name for HF hubs (FunAudioLLM/Fun-ASR-Nano-2512). Only strip
        # ModelScope-style aliases that historically used a trailing path segment.
        if (
            not _is_fun_asr_nano(model_name)
            and len(model_name.split("/")) > 1
            and not model_name.startswith("FunAudioLLM/")
        ):
            model_name = model_name.split("/")[-1]
        self.device = device
        self.language = (language or "").strip()
        # Classic paraformer-zh emits spaced CJK tokens; Fun-ASR / English must keep spaces.
        self._strip_spaces = model_name in {"paraformer-zh", "paraformer"} or model_name.endswith(
            "paraformer-zh"
        )
        try:
            from funasr import AutoModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Paraformer STT requires the optional 'paraformer' extra. "
                "Install it with `pip install speech-to-speech[paraformer]`."
            ) from exc

        load_kwargs: dict[str, Any] = {
            "model": model_name,
            "device": device,
            "disable_update": True,
            "disable_pbar": True,
        }
        if _is_fun_asr_nano(model_name) or model_name.startswith("FunAudioLLM/"):
            # Per FunASR / Fun-ASR-Nano docs: HF hub + trust_remote_code.
            load_kwargs.update(
                {
                    "hub": "hf",
                    "trust_remote_code": True,
                }
            )

        self.model = AutoModel(**load_kwargs)
        self.warmup()

    def _postprocess(self, text: str) -> str:
        text = (text or "").strip()
        if self._strip_spaces:
            text = text.replace(" ", "")
        return text

    def _generate(self, audio: np.ndarray) -> str:
        gen_kwargs: dict[str, Any] = {"input": audio, "cache": {}, "batch_size": 1}
        if self.language:
            gen_kwargs["language"] = self.language
        result = self.model.generate(**gen_kwargs)
        return self._postprocess(result[0]["text"])

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")

        n_steps = 1
        dummy_input = np.array([0] * 512, dtype=np.float32)
        for _ in range(n_steps):
            _ = self._generate(dummy_input)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering paraformer...")

        pred_text = self._generate(vad_audio.audio)
        # MPS-only cleanup (Apple Silicon). Unconditional torch.mps.empty_cache()
        # raises RuntimeError on CUDA/CPU Linux.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif (
            str(getattr(self, "device", "")).startswith("cuda")
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()

        logger.debug("finished paraformer inference")
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
