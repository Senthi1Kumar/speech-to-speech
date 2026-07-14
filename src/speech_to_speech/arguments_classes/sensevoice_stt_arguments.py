from dataclasses import dataclass, field


@dataclass
class SenseVoiceSTTHandlerArguments:
    sensevoice_stt_model_name: str = field(
        default="FunAudioLLM/SenseVoiceSmall",
        metadata={
            "help": (
                "FunASR SenseVoice model id. Default FunAudioLLM/SenseVoiceSmall "
                "(HF hub). See https://huggingface.co/FunAudioLLM/SenseVoiceSmall"
            )
        },
    )
    sensevoice_stt_device: str = field(
        default="cuda",
        metadata={"help": "Device for SenseVoice (cuda / cpu). Default cuda."},
    )
    sensevoice_stt_language: str = field(
        default="en",
        metadata={
            "help": (
                "SenseVoice language: auto, zn, en, yue, ja, ko, nospeech. "
                "Default en for Nova English."
            )
        },
    )
    sensevoice_stt_use_itn: bool = field(
        default=True,
        metadata={"help": "Inverse text normalization + punctuation. Default True."},
    )
