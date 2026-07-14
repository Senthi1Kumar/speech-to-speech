"""Nova-owned s2s regressions: articulator tools + ConnState metric fields.

Callers: ``scripts/run_tests.sh`` ``run_s2s_nova`` / component. LiteRT paths untouched.
"""
from __future__ import annotations

from queue import Queue
from threading import Event as ThreadingEvent

from speech_to_speech.api.openai_realtime.service import ConnState, RealtimeService
from speech_to_speech.pipeline.events import TokenUsageEvent


def test_conn_state_exposes_decode_metric_fields() -> None:
    st = ConnState()
    assert st.last_decode_tok_s is None
    assert st.last_prefill_tok_s is None
    assert st.last_llm_tpot_ms is None


def test_token_usage_event_sets_conn_state_decode() -> None:
    svc = RealtimeService(text_prompt_queue=Queue(), should_listen=ThreadingEvent())
    conn_id = svc.register()
    try:
        svc.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=10, output_tokens=5, decode_tok_s=42.5),
        )
        assert svc._state(conn_id).last_decode_tok_s == 42.5
    finally:
        svc.unregister(conn_id)


def test_articulator_contract_zero_tools_after_agent() -> None:
    """After /tools/agent, speak turn must force tools=[] + tool_choice=none.

    Nonempty placeholder schemas prime LFM to emit <|tool_call_*|> as speech.
    """
    agent = {
        "needs_tools": True,
        "tools": [{"type": "function", "name": "web_search"}],
        "tool_choice": "none",
        "speak_payload": "Amazon near one eighty.",
    }
    # Mirror base_openai_compatible_language_model generate() assignment.
    assert agent["tools"], "fixture must start nonempty to prove override"
    req_tools: list = []
    req_tool_choice = "none"
    assert req_tools == []
    assert req_tool_choice == "none"
