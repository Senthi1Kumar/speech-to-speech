"""AutoArk Audio8-ASR STT (non-streaming, full VAD segment).

Inference patterned on AutoArk/open-audio-opd
``scripts/infer/ark_asr_transformers.py`` (``ArkAsrTransformerInferencer``)
and the HF ``examples/transcribe.py`` for Audio8-ASR-0.1B.
Silero owns VAD; this handler re-transcribes each closed float32 segment.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import wave
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from rich.console import Console
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()

PROMPT = "Please transcribe this audio."
SPECIAL_TOKEN_PATTERN = re.compile(
    r"<\|(?:"
    r"bicodec_(?:semantic|global)_\d+|"
    r"(?:start|end)_(?:global_token|glm_token|semantic_token|content)"
    r")\|>"
)
TURN_END_MARKERS = ("<|user|>", "<|assistant|>", "<|im_end|>")
LEADING_NOISE_PATTERN = re.compile(r"^[\s,.;:!?-]+")
CONTROL_TOKEN_PATTERN = re.compile(r"^<.*>$")


class BlockTokenIdsFromLogitsProcessor(LogitsProcessor):
    """Mask token ids >= block_from_id and explicit control token ids."""

    def __init__(
        self, block_from_id: int | None, block_token_ids: Iterable[int] | None = None
    ):
        self.block_from_id = (
            None if block_from_id is None or int(block_from_id) < 0 else int(block_from_id)
        )
        self.block_token_ids = sorted(
            set(int(token_id) for token_id in (block_token_ids or []))
        )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        vocab_size = scores.shape[-1]
        if self.block_from_id is not None and self.block_from_id < vocab_size:
            scores[:, self.block_from_id :] = -float("inf")
        valid = [t for t in self.block_token_ids if 0 <= t < vocab_size]
        if valid:
            scores[:, valid] = -float("inf")
        return scores


def truncate_generation_text(text: str) -> str:
    if not text:
        return ""
    cut = len(text)
    for marker in TURN_END_MARKERS:
        index = text.find(marker)
        if index != -1 and index < cut:
            cut = index
    return text[:cut].strip()


def remove_special_tokens(text: str) -> str:
    if not text:
        return ""
    if "<|text|>" in text:
        text = text.split("<|text|>", 1)[1]
    return SPECIAL_TOKEN_PATTERN.sub("", text).strip()


def normalize_prediction_text(text: str) -> str:
    if not text:
        return ""
    text = truncate_generation_text(text)
    text = remove_special_tokens(text)
    text = re.sub(r"\s+", " ", text).strip()
    return LEADING_NOISE_PATTERN.sub("", text).strip()


def _normalize_token_ids(token_ids: Any) -> list[int]:
    if token_ids is None:
        return []
    if isinstance(token_ids, (list, tuple, set)):
        return [int(t) for t in token_ids if t is not None]
    return [int(token_ids)]


def _build_eos_token_ids(tokenizer: Any) -> list[int]:
    eos_ids = list(_normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    for marker in TURN_END_MARKERS:
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(token_id, int) and token_id >= 0:
            eos_ids.append(int(token_id))
    return list(dict.fromkeys(eos_ids))


def _build_asr_keep_token_ids(model: Any, tokenizer: Any) -> list[int]:
    keep: set[int] = set()
    keep.update(_normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    keep.update(
        _normalize_token_ids(getattr(getattr(model, "config", None), "eos_token_id", None))
    )
    keep.update(
        _normalize_token_ids(
            getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        )
    )
    return sorted(keep)


def _build_asr_extra_block_token_ids(
    tokenizer: Any,
    keep_token_ids: Iterable[int] | None = None,
    block_from_id: int | None = None,
) -> list[int]:
    keep = set(int(t) for t in (keep_token_ids or []))
    max_control = None if block_from_id is None or int(block_from_id) < 0 else int(block_from_id)
    block_token_ids = set(
        int(t) for t in getattr(tokenizer, "all_special_ids", []) if t is not None
    )
    added = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    for token_id, token_meta in added.items():
        tid = int(token_id)
        if max_control is not None and tid >= max_control:
            continue
        content = getattr(token_meta, "content", None)
        if content is None and isinstance(token_meta, dict):
            content = token_meta.get("content")
        if content and CONTROL_TOKEN_PATTERN.match(content):
            block_token_ids.add(tid)
    block_token_ids.difference_update(keep)
    return sorted(block_token_ids)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "keys") and hasattr(value, "__getitem__"):
        return {key: value[key] for key in value.keys()}
    raise TypeError(f"Unexpected processor output type: {type(value)}")


def _write_temp_wav(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Write mono float32 [-1,1] to a 16-bit PCM wav; return path."""
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())
    return path


class Audio8STTHandler(BaseSTTHandler):
    """Non-streaming Audio8-ASR via Transformers (trust_remote_code)."""

    def setup(
        self,
        model_name: str = "AutoArk-AI/Audio8-ASR-0.1B",
        device: str = "cuda",
        max_new_tokens: int = 128,
        attn_impl: str = "eager",
        block_token_id_from: int = 151670,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        del gen_kwargs
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.attn_impl = (attn_impl or "eager").strip() or "eager"
        self.block_token_id_from = int(block_token_id_from)
        self.sample_rate = 16000
        self.max_audio_seconds = 30

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Audio8 STT requires transformers. "
                "Install with the speech-to-speech stack dependencies."
            ) from exc

        torch_dtype = (
            torch.bfloat16 if str(device).startswith("cuda") and torch.cuda.is_available()
            else torch.float32
        )
        self.torch_dtype = torch_dtype

        candidates = [self.attn_impl]
        if self.attn_impl == "auto":
            candidates = (
                ["flash_attention_2", "sdpa", "eager"]
                if str(device).startswith("cuda")
                else ["eager"]
            )
        elif self.attn_impl == "flash_attention_2":
            candidates = ["flash_attention_2", "sdpa", "eager"]

        last_error: Exception | None = None
        model = None
        resolved_attn = self.attn_impl
        for candidate in candidates:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    attn_implementation=candidate,
                ).to(device)
                model.eval()
                resolved_attn = candidate
                break
            except (ImportError, RuntimeError, ValueError, OSError) as exc:
                last_error = exc
                if candidate == candidates[-1]:
                    raise
                logger.warning(
                    "Audio8 attn_impl=%s unavailable (%s); trying next",
                    candidate,
                    str(exc).splitlines()[0],
                )
        if model is None:
            raise RuntimeError(f"Failed to load Audio8 model: {last_error}")

        self.model = model
        self.resolved_attn_impl = resolved_attn
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.eos_token_ids = _build_eos_token_ids(self.tokenizer)
        keep = _build_asr_keep_token_ids(self.model, self.tokenizer)
        self.extra_block_token_ids = _build_asr_extra_block_token_ids(
            self.tokenizer,
            keep_token_ids=keep,
            block_from_id=self.block_token_id_from,
        )
        logger.info(
            "Audio8 loaded model=%s device=%s attn=%s dtype=%s",
            model_name,
            device,
            resolved_attn,
            torch_dtype,
        )
        self.warmup()

    def _generate(self, audio: np.ndarray) -> str:
        wav_path = _write_temp_wav(audio, self.sample_rate)
        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": wav_path},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]
            batch_raw = self.processor.apply_chat_template(
                conversation,
                return_tensors="pt",
                sampling_rate=self.sample_rate,
                audio_padding="longest",
                add_generation_prompt=True,
                audio_max_length=int(self.max_audio_seconds * self.sample_rate),
                text_kwargs={
                    "padding": "longest",
                    "truncation": True,
                    "max_length": 1000,
                },
            )
            if torch.is_tensor(batch_raw):
                raise RuntimeError("Audio8 apply_chat_template returned Tensor-only")
            batch = _as_dict(batch_raw)
            for key, value in list(batch.items()):
                if not torch.is_tensor(value):
                    continue
                if key == "audios":
                    batch[key] = value.to(device=self.device, dtype=self.torch_dtype)
                else:
                    batch[key] = value.to(self.device)

            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.eos_token_ids:
                generate_kwargs["eos_token_id"] = self.eos_token_ids
            if self.block_token_id_from >= 0 or self.extra_block_token_ids:
                generate_kwargs["logits_processor"] = LogitsProcessorList(
                    [
                        BlockTokenIdsFromLogitsProcessor(
                            block_from_id=self.block_token_id_from,
                            block_token_ids=self.extra_block_token_ids,
                        )
                    ]
                )

            with torch.inference_mode():
                output_ids = self.model.generate(**batch, **generate_kwargs)

            prompt_len = int(batch["input_ids"].shape[1])
            raw = self.tokenizer.decode(
                output_ids[0, prompt_len:], skip_special_tokens=False
            )
            text = normalize_prediction_text(raw)
            if not text:
                text = self.processor.decode(
                    output_ids[0, prompt_len:], skip_special_tokens=True
                ).strip()
            return text
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def warmup(self) -> None:
        logger.info("Warming up %s", self.__class__.__name__)
        dummy = np.zeros(self.sample_rate, dtype=np.float32)
        _ = self._generate(dummy)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering audio8...")
        pred_text = self._generate(vad_audio.audio)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("finished audio8 inference")
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
