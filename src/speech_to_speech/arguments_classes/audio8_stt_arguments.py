from dataclasses import dataclass, field


@dataclass
class Audio8STTHandlerArguments:
    audio8_stt_model_name: str = field(
        default="AutoArk-AI/Audio8-ASR-0.1B",
        metadata={
            "help": (
                "Audio8 ASR model id (HF). Default AutoArk-AI/Audio8-ASR-0.1B. "
                "CC-BY-NC-4.0 — research/demo only. "
                "See https://huggingface.co/AutoArk-AI/Audio8-ASR-0.1B"
            )
        },
    )
    audio8_stt_device: str = field(
        default="cuda",
        metadata={"help": "Device for Audio8 (cuda / cpu). Default cuda."},
    )
    audio8_stt_max_new_tokens: int = field(
        default=128,
        metadata={"help": "Max new tokens for ASR decode. Default 128 (short VAD turns)."},
    )
    audio8_stt_attn_impl: str = field(
        default="eager",
        metadata={
            "help": (
                "Attention implementation: eager (Audio8 HF default), sdpa, "
                "flash_attention_2, or auto. Default eager."
            )
        },
    )
    audio8_stt_block_token_id_from: int = field(
        default=151670,
        metadata={
            "help": (
                "Mask tokenizer ids >= this value during generate "
                "(ArkAsrTransformerInferencer default 151670). Set -1 to disable."
            )
        },
    )
