from speech_to_speech.LLM.utils import remove_unspeechable


def test_remove_unspeechable_normalizes_smart_apostrophes() -> None:
    assert remove_unspeechable("I’ll reply if here’s the plan.") == "I'll reply if here's the plan."


def test_remove_unspeechable_keeps_text_and_drops_emoji() -> None:
    assert remove_unspeechable("Hello 👋 lobster 🦞") == "Hello  lobster "


def test_remove_unspeechable_strips_tool_call_markup() -> None:
    """LFM sometimes emits tool-call tokens as plain text on the speak path."""
    raw = (
        "Amazon is near one eighty. "
        "<|tool_call_start|>web_search(place=Benin, temp_c=28)<|tool_call_end|>"
    )
    cleaned = remove_unspeechable(raw)
    assert "<|tool_call" not in cleaned
    assert "web_search" not in cleaned
    assert "Amazon is near one eighty." in cleaned


def test_remove_unspeechable_strips_tool_calls_section() -> None:
    raw = (
        "Here you go. "
        "<|tool_calls_section_begin|><|tool_call_start|>x()<|tool_call_end|>"
        "<|tool_calls_section_end|> Thanks."
    )
    cleaned = remove_unspeechable(raw)
    assert "tool_call" not in cleaned.lower()
    assert "Here you go." in cleaned
    assert "Thanks." in cleaned
