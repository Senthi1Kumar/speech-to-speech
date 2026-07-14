from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime
from typing import Any, Optional

import httpx
from nltk import sent_tokenize
from openai import OpenAI
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import (
    Chat,
    ChatItemError,
    SupportedItem,
    build_active_chat,
    make_system_message,
    make_user_message,
)
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn, build_compactor
from speech_to_speech.LLM.text_prompt import build_text_system_prompt
from speech_to_speech.LLM.utils import remove_unspeechable, resolve_auto_language
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    LLMResponseChunk,
    TokenUsage,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import _generate_id, is_out_of_band, response_wants_audio

logger = logging.getLogger(__name__)


# ── Normalised provider events ────────────────────────────────────────────────
# Each backend's stream/response is mapped to this small vocabulary so the shared
# speech-pipeline logic (sentence batching, cancellation, history, token usage)
# lives in one place. Subclasses differ only in how they produce these events.


class TextDelta(BaseModel):
    """Incremental assistant text. Always RAW (unfiltered); the base applies
    ``remove_unspeechable`` for the audio path."""

    text: str


class AssistantMessage(BaseModel):
    """A complete assistant turn to write back to history."""

    content: list[AssistantContent]


class ToolCall(BaseModel):
    """A complete function tool call (``call_id`` / ``id`` already regenerated)."""

    item: ResponseFunctionToolCall


class Usage(BaseModel):
    """Token accounting for the turn (+ optional llama.cpp timings)."""

    input_tokens: int
    output_tokens: int
    decode_tok_s: float | None = None
    prefill_tok_s: float | None = None
    llm_tpot_ms: float | None = None


ProviderEvent = TextDelta | AssistantMessage | ToolCall | Usage


def _llama_timings(obj: Any) -> dict[str, Any]:
    """Pull llama-server ``timings`` off an OpenAI SDK completion/chunk."""
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict) and isinstance(extra.get("timings"), dict):
        return extra["timings"]
    dumped = getattr(obj, "model_dump", None)
    if callable(dumped):
        t = dumped().get("timings")
        if isinstance(t, dict):
            return t
    return {}


def _usage_from_openai(usage_obj: Any, source: Any = None) -> Usage:
    timings = _llama_timings(source if source is not None else usage_obj)
    decode = timings.get("predicted_per_second")
    prefill = timings.get("prompt_per_second")
    tpot = timings.get("predicted_per_token_ms")
    return Usage(
        input_tokens=int(getattr(usage_obj, "prompt_tokens", None) or 0),
        output_tokens=int(getattr(usage_obj, "completion_tokens", None) or 0),
        decode_tok_s=float(decode) if decode is not None else None,
        prefill_tok_s=float(prefill) if prefill is not None else None,
        llm_tpot_ms=float(tpot) if tpot is not None else None,
    )


def _last_user_text(chat: Chat) -> str:
    """Most recent user utterance text, or '' if the buffer tip is not a user turn."""
    for item in reversed(chat.buffer):
        if isinstance(item, RealtimeConversationItemUserMessage):
            return " ".join(p.text for p in item.content if getattr(p, "text", None))
        # Tool follow-up / assistant tip: do not dig past non-user items for routing.
        break
    return ""


def _nova_route_override(query: str) -> tuple[list[Any] | None, Any]:
    """Ask Nova ``/tools/route`` for per-turn tools + tool_choice.

    Client-side ``session.update`` races the auto STT→LLM path, so Gemma often
    still sees ``tool_choice=auto`` and narrates. When ``NOVA_TOOLS_ROUTE_URL``
    is set (run_demo), override here immediately before the chat-completions
    call. Failures are ignored so a tools outage does not kill the voice loop.
    """
    url = os.environ.get("NOVA_TOOLS_ROUTE_URL", "").strip()
    if not url or not query.strip():
        return None, None
    try:
        resp = httpx.post(url, json={"query": query, "k": 8}, timeout=0.75)
        resp.raise_for_status()
        data = resp.json()
        tools = data.get("tools")
        if not isinstance(tools, list) or not tools:
            return None, None
        return tools, data.get("tool_choice", "auto")
    except Exception as exc:  # noqa: BLE001 — never block generation on route failure
        logger.warning("Nova tools route failed: %s", exc)
        return None, None



def _nova_auth_precheck(query: str, session_id: str | None) -> dict[str, Any] | None:
    """DriveAuth payment precheck before /tools/agent. None = bypass / disabled."""
    url = os.environ.get("NOVA_TOOLS_AUTH_PRECHECK_URL", "").strip()
    if not url or not query.strip():
        return None
    sid = session_id or os.environ.get("NOVA_SESSION_ID") or "s2s-default"
    try:
        resp = httpx.post(
            url,
            json={"session_id": sid, "transcript": query},
            timeout=3.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Nova DriveAuth precheck failed: %s", exc)
        return None


def _nova_tool_agent(query: str, session_id: str | None = None) -> dict[str, Any] | None:
    """Ask Nova ``/tools/agent`` (230M pick + execute). Speak path only.

    Returns the agent JSON dict, or None if disabled / failed / not model mode.
    """
    mode = (os.environ.get("NOVA_TOOL_ROUTE_MODE") or "").strip().lower()
    if mode not in {"model", "agent", "router"}:
        return None
    url = os.environ.get("NOVA_TOOLS_AGENT_URL", "").strip()
    if not url or not query.strip():
        return None
    try:
        payload = {"query": query, "execute": True}
        if session_id:
            payload["session_id"] = session_id
        resp = httpx.post(url, json=payload, timeout=20.0)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Nova tool agent failed: %s", exc)
        return None


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        return str(tool.get("name") or (tool.get("function") or {}).get("name") or "")
    return str(getattr(tool, "name", "") or "")


def _tool_parameters(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        params = tool.get("parameters")
        if params is None:
            params = (tool.get("function") or {}).get("parameters")
        return params if isinstance(params, dict) else {}
    params = getattr(tool, "parameters", None)
    return params if isinstance(params, dict) else {}


def _args_for_forced_tool(name: str, tool: Any, user_text: str) -> str | None:
    """Build JSON args for a Nova-forced tool, or None if the LLM must fill them.

    Gemma/llama often ignores ``tool_choice={type:function,name}`` and narrates
    (``Tools: []``). For empty-arg and simple-arg tools we emit the call ourselves.
    """
    params = _tool_parameters(tool)
    required = list(params.get("required") or [])
    q = user_text.lower()

    if name == "web_search":
        cats = (
            "politics", "traffic", "business", "sports", "crime", "tech", "weather",
        )
        category = next((c for c in cats if c in q), "")
        place = ""
        m = re.search(
            r"\bin\s+([A-Za-z][A-Za-z]*(?:\s+[A-Za-z]+)*)",
            user_text,
            re.I,
        )
        if m:
            place = m.group(1).strip().rstrip(".,?!")
        elif "bangalore" in q or "bengaluru" in q:
            place = "Bengaluru"
        if any(w in q for w in ("news", "headline", "headlines", "breaking")):
            query = f"{place} news today".strip() if place else "news headlines today"
            if category:
                query = f"{place} {category} news today".strip()
            payload: dict[str, str] = {"query": query}
            if place:
                payload["place"] = place
            if category:
                payload["category"] = category
            else:
                payload["category"] = "general"
            return json.dumps(payload)
        # Strip filler for general search — don't dump the whole utterance.
        cleaned = re.sub(
            r"^(?:hey[,.]?\s+|can you\s+|please\s+|tell me\s+|what(?:'s| is)\s+)",
            "",
            user_text.strip(),
            flags=re.I,
        ).rstrip("?.")
        return json.dumps({"query": cleaned or user_text.strip()})

    if name in ("set_windows", "set_sunroof"):
        # "roll down" / "open" → open; "close" / "roll up" → closed
        close = any(w in q for w in ("close", "shut", "roll up", "window up"))
        if close and "down" not in q:
            return json.dumps({"open": False})
        return json.dumps({"open": True})

    if name == "get_weather":
        m = re.search(r"\bin\s+(.+?)(?:\?|\.|$)", user_text, re.I)
        place = (m.group(1).strip() if m else user_text.strip()) or "Bangalore"
        return json.dumps({"place": place})

    if name == "play_music":
        m = re.search(r"play\s+(?:some\s+)?(.+)", user_text, re.I)
        query = (m.group(1).strip() if m else user_text.strip()).rstrip(".")
        return json.dumps({"query": query or "music"})

    if name == "start_research":
        topic = re.sub(r"^(?:research|look up|investigate)\s+", "", user_text, flags=re.I).strip()
        return json.dumps({"topic": topic or user_text.strip()})

    if name == "set_reminder":
        return json.dumps({"text": user_text.strip(), "when": user_text.strip()})

    if name == "check_calendar":
        # Order matters: "day after tomorrow" contains "tomorrow".
        if re.search(r"day\s+after\s+tomorrow", q):
            return json.dumps({"day": "day_after_tomorrow"})
        # Explicit calendar date: "July 16" / "2026-07-16" / "16th of July" / "16th July"
        iso = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", user_text)
        if iso:
            y, m, d = int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
            return json.dumps({"on_date": f"{y:04d}-{m:02d}-{d:02d}"})
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        }
        named = re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
            r"\s+(\d{1,2})(?:st|nd|rd|th)?\b",
            q,
        )
        if named:
            mon = months[named.group(1)]
            day_n = int(named.group(2))
            year = datetime.now().year
            return json.dumps({"on_date": f"{year:04d}-{mon:02d}-{day_n:02d}"})
        day_first = re.search(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?"
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            q,
        )
        if day_first:
            day_n = int(day_first.group(1))
            mon = months[day_first.group(2)]
            year = datetime.now().year
            return json.dumps({"on_date": f"{year:04d}-{mon:02d}-{day_n:02d}"})
        if "tomorrow" in q:
            return json.dumps({"day": "tomorrow"})
        if re.search(r"\btoday\b", q):
            return json.dumps({"day": "today"})
        return json.dumps({"day": "week"})

    if name == "check_email":
        if any(w in q for w in ("summar", "summary", "summaris", "what does it say", "what's it about", "read it", "body")):
            return json.dumps({"mode": "summarize"})
        if any(w in q for w in ("latest", "last", "recent", "newest")):
            return json.dumps({"mode": "latest"})
        return json.dumps({"mode": "unread"})

    if name == "list_drive_files":
        m = re.search(
            r"(?:drive|files?|docs?)\s+(?:for|named|called|about)\s+(.+)",
            user_text,
            re.I,
        )
        query = (m.group(1).strip().rstrip("?.") if m else "")
        return json.dumps({"query": query})

    if name == "create_drive_folder":
        m = re.search(
            r"(?:named|called)\s+as\s+(.+)$",
            user_text,
            re.I,
        ) or re.search(
            r"(?:named|called)\s+(.+)$",
            user_text,
            re.I,
        ) or re.search(
            r"(?:folder|directory|dir)\s+(.+)$",
            user_text,
            re.I,
        )
        folder = (m.group(1).strip().rstrip("?.") if m else user_text.strip())
        for prefix in ("as ", "named ", "called ", "folder ", "directory ", "dir "):
            if folder.lower().startswith(prefix):
                folder = folder[len(prefix) :].strip()
        return json.dumps({"name": folder or "New folder"})

    if name == "send_payment":
        am = re.search(r"\$\s*(\d+(?:\.\d+)?)", user_text) or re.search(
            r"(\d+(?:\.\d+)?)\s*(?:dollars?|rupees?|rs\.?)", user_text, re.I
        )
        if not am:
            return None
        amount = float(am.group(1))
        pm = re.search(
            r"(?:to|payee is|pay is|payee)\s+([A-Za-z][\w\s'-]+?)(?:\.|$|,|\s+the\s+amount)",
            user_text,
            re.I,
        )
        if not pm:
            pm = re.search(r"pay\s+\$?\d+(?:\.\d+)?\s+to\s+([A-Za-z][\w\s'-]+)", user_text, re.I)
        if not pm:
            return None
        payee = pm.group(1).strip().rstrip(".")
        return json.dumps({"payee": payee, "amount": amount})

    if not required:
        return "{}"

    return None


def _forced_tool_call(name: str, arguments: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        arguments=arguments,
        call_id=_generate_id("call"),
        name=name,
        type="function_call",
        id=_generate_id("fc"),
        status="completed",
    )


class _Turn(BaseModel):
    """Per-request context threaded through generation (immutable for the turn)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    language_code: Optional[str]
    gen: int | None
    runtime_config: Any
    response: Any
    turn_id: str | None
    turn_revision: int | None
    speech_stopped_at_s: float | None
    wants_audio: bool


class _GenState(BaseModel):
    """Mutable accumulators collected while consuming a turn's events."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tools: list[ResponseFunctionToolCall] = Field(default_factory=list)
    pending: list[SupportedItem] = Field(default_factory=list)
    clean_text: str = ""  # filtered text, kept only for the debug log
    input_tokens: int = 0
    output_tokens: int = 0
    decode_tok_s: float | None = None
    prefill_tok_s: float | None = None
    llm_tpot_ms: float | None = None


class BaseOpenAICompatibleHandler(BaseHandler[LLMIn, LLMOut], ABC):
    """Shared lifecycle for OpenAI-compatible LLM backends (Responses & Chat
    Completions).

    Subclasses implement four hooks — :meth:`warmup`,
    :meth:`_build_compaction_generate_fn`, :meth:`_serialize`, :meth:`_request`,
    :meth:`_iter_events` and :meth:`_build_optional_kwargs` — and inherit the
    request/response orchestration: speculative-turn gating, cancellation,
    sentence batching, text-only vs audio handling, history write-back, token
    usage, out-of-band handling and error termination.
    """

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(
        self,
        model_name: str = "gpt-5.4-mini",
        device: str = "cuda",
        gen_kwargs: dict[str, Any] = {},
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: bool = True,
        user_role: str = "user",
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        disable_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        request_timeout_s: float = float(os.environ.get("S2S_LLM_REQUEST_TIMEOUT_S", "20.0")),
        stream_batch_sentences: int = 3,
        enable_lang_prompt: bool = False,
        compact_history: bool = False,
        **_kwargs: Any,
    ) -> None:
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.model_name = model_name
        self.stream = stream
        self.stream_batch_sentences = max(1, stream_batch_sentences)
        self.enable_lang_prompt = enable_lang_prompt
        self.gen_kwargs = dict(gen_kwargs)
        self.request_timeout_s = float(request_timeout_s)
        self.request_timeout = httpx.Timeout(
            self.request_timeout_s,
            connect=min(10.0, self.request_timeout_s),
        )

        self.user_role = user_role
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._extra_body = self._build_extra_body(base_url, disable_thinking, reasoning_effort)
        self.compactor = build_compactor(self._build_compaction_generate_fn()) if compact_history else None
        self.warmup()

    @staticmethod
    def _is_official_openai(base_url: Optional[str]) -> bool:
        """Whether ``base_url`` points at the official OpenAI server.

        Normalises a trailing slash so ``https://api.openai.com/v1/`` is also
        recognised; the official server rejects the provider-specific extra_body
        keys we send to vLLM / the HF router.
        """
        if base_url is None:
            return False
        return base_url.rstrip("/") == "https://api.openai.com/v1"

    @classmethod
    def _build_extra_body(
        cls,
        base_url: Optional[str],
        disable_thinking: bool,
        reasoning_effort: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """Build the provider-specific ``extra_body`` used to disable reasoning.

        Providers differ in how reasoning is turned off: vLLM/Qwen honour
        ``chat_template_kwargs.enable_thinking=false``, while others (e.g. GLM via
        the HF router) ignore that and require ``reasoning_effort='none'``. A
        non-empty ``reasoning_effort`` therefore takes precedence; otherwise we fall
        back to the chat-template flag. None of this applies to the official
        OpenAI server, which rejects unknown extra_body keys.
        """
        if base_url is None or cls._is_official_openai(base_url):
            return None
        if reasoning_effort:
            return {"reasoning_effort": reasoning_effort}
        if disable_thinking:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None

    # ── subclass hooks ──────────────────────────────────────────────────────--

    @abstractmethod
    def warmup(self) -> None:
        """Issue a cheap request so the model/connection is ready before serving."""
        ...

    @abstractmethod
    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        """Return a ``(system, user) -> text`` fn used to compact long histories."""
        ...

    @abstractmethod
    def _serialize(self, active_chat: Chat) -> Any:
        """Serialise the chat to the backend's request payload (input/messages)."""
        ...

    @abstractmethod
    def _request(self, api_input: Any, optional_kwargs: dict[str, Any]) -> Any:
        """Issue the create() call and return the response or stream."""
        ...

    @abstractmethod
    def _iter_stream_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        """Map a streaming response to normalised :data:`ProviderEvent`s."""
        ...

    @abstractmethod
    def _iter_response_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        """Map a non-streaming response to normalised :data:`ProviderEvent`s."""
        ...

    def _iter_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        """Dispatch to the stream/non-stream mapper. ``self.stream`` is the single
        source of truth (it set the request's ``stream=`` flag), so the response
        type always matches it."""
        if self.stream:
            yield from self._iter_stream_events(api_response)
        else:
            yield from self._iter_response_events(api_response)

    @abstractmethod
    def _build_optional_kwargs(self, req_tools: Any, req_tool_choice: Any) -> dict[str, Any]:
        """Build the per-request tools/tool_choice kwargs in the backend's shape."""
        ...

    # ── speculative-turn / cancellation gating ─────────────────────────────────

    def _turn_is_latest(self, turn_id: str | None, turn_revision: int | None) -> bool:
        return self.speculative_turns is None or self.speculative_turns.is_latest(turn_id, turn_revision)

    def _generation_is_stale(self, gen: int | None) -> bool:
        return gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen)

    def _turn_output_allowed(self, turn_id: str | None, turn_revision: int | None) -> bool:
        if self.speculative_turns is None:
            return True
        return self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision)

    def _apply_config(
        self,
        chat: Chat,
        instructions: Optional[str],
        wants_audio: bool = True,
    ) -> None:
        if instructions:
            builder = build_voice_system_prompt if wants_audio else build_text_system_prompt
            full_instructions = builder(instructions)
            chat.add_item(make_system_message(full_instructions))

    # ── output helpers ──────────────────────────────────────────────────────--

    def _chunk(
        self,
        turn: _Turn,
        *,
        text: str = "",
        tools: list[ResponseFunctionToolCall] | None = None,
        language_code: Optional[str] = None,
    ) -> LLMResponseChunk:
        return LLMResponseChunk(
            text=text,
            language_code=language_code if language_code is not None else turn.language_code,
            tools=tools or [],
            runtime_config=turn.runtime_config,
            response=turn.response,
            turn_id=turn.turn_id,
            turn_revision=turn.turn_revision,
            speech_stopped_at_s=turn.speech_stopped_at_s,
            cancel_generation=turn.gen,
        )

    def _record_tool_call(self, state: _GenState, turn: _Turn, item: ResponseFunctionToolCall) -> Iterator[LLMOut]:
        """Emit a tool call, persisting it (and any assistant text seen so far)
        to history *before* it is forwarded to the client.

        The function_call must already exist in the conversation by the time the
        client returns its ``function_call_output``; otherwise a fast client
        races ahead of the deferred end-of-turn write-back and the output is
        rejected ("No function_call with call_id ... found"), which makes the
        model re-issue the same tool call. The call lands in ``_pending_tool_calls``
        (not serialized until its output pairs it), so eager recording is safe.

        Out-of-band turns never touch the default conversation, and a stale turn
        records nothing (it is not forwarded to the client either)."""
        state.tools.append(item)
        fc_item = RealtimeConversationItemFunctionCall(
            type="function_call",
            name=item.name,
            arguments=item.arguments,
            call_id=item.call_id,
            id=item.id,
            status=item.status,
        )
        if self._generation_is_stale(turn.gen) or not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
            logger.info("LLM generation cancelled (stale speculative turn)")
            return
        if not is_out_of_band(turn.response):
            # Flush assistant text accumulated before this call first (so history
            # order matches what the client received), then persist the call —
            # all before the chunk leaves for the client.
            chat = turn.runtime_config.chat
            for pending_item in state.pending:
                chat.add_item(pending_item)
            state.pending.clear()
            chat.add_item(fc_item)
        yield self._chunk(turn, tools=[item])

    # ── consumption ─────────────────────────────────────────────────────────--

    def _consume_streaming(self, events: Iterator[ProviderEvent], state: _GenState, turn: _Turn) -> Iterator[LLMOut]:
        cancelled = False
        printable_text = ""
        sentence_batch: list[str] = []

        def _flush(batch: list[str]) -> Iterator[LLMOut]:
            if not batch:
                return
            if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                logger.info("LLM generation cancelled (stale speculative turn)")
                return
            yield self._chunk(turn, text=" ".join(batch))

        for event in events:
            if self._generation_is_stale(turn.gen) or not self._turn_is_latest(turn.turn_id, turn.turn_revision):
                logger.info("LLM generation cancelled (interruption)")
                cancelled = True
                break

            if isinstance(event, Usage):
                state.input_tokens = event.input_tokens
                state.output_tokens = event.output_tokens
                if event.decode_tok_s is not None:
                    state.decode_tok_s = event.decode_tok_s
                if event.prefill_tok_s is not None:
                    state.prefill_tok_s = event.prefill_tok_s
                if event.llm_tpot_ms is not None:
                    state.llm_tpot_ms = event.llm_tpot_ms
            elif isinstance(event, AssistantMessage):
                state.pending.append(
                    RealtimeConversationItemAssistantMessage(type="message", role="assistant", content=event.content)
                )
            elif isinstance(event, ToolCall):
                # Flush any pending spoken text before emitting the tool call.
                if printable_text.strip():
                    sentence_batch.append(printable_text.strip())
                    printable_text = ""
                if sentence_batch:
                    if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                        logger.info("LLM generation cancelled (stale speculative turn)")
                        cancelled = True
                        break
                    yield from _flush(sentence_batch)
                    sentence_batch = []
                yield from self._record_tool_call(state, turn, event.item)
            elif isinstance(event, TextDelta):
                if not turn.wants_audio:
                    # Text-only: forward verbatim. Keep every character (no
                    # remove_unspeechable, which strips TTS-unfriendly symbols) and
                    # don't sentence-split (sent_tokenize collapses newlines/markdown).
                    state.clean_text += event.text
                    if event.text:
                        if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                            logger.info("LLM generation cancelled (stale speculative turn)")
                            cancelled = True
                            break
                        yield self._chunk(turn, text=event.text)
                    continue
                new_text = remove_unspeechable(event.text)
                state.clean_text += new_text
                printable_text += new_text
                sentences = sent_tokenize(printable_text)
                if len(sentences) > 1:
                    for s in sentences[:-1]:
                        sentence_batch.append(s)
                        if len(sentence_batch) >= self.stream_batch_sentences:
                            if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                                logger.info("LLM generation cancelled (stale speculative turn)")
                                cancelled = True
                                break
                            yield from _flush(sentence_batch)
                            sentence_batch = []
                    if cancelled:
                        break
                    printable_text = sentences[-1]

        if not cancelled:
            if printable_text.strip():
                sentence_batch.append(printable_text.strip())
            if sentence_batch:
                if self._generation_is_stale(turn.gen):
                    logger.info("LLM generation cancelled (interruption)")
                else:
                    logger.debug(f"Clean text: {state.clean_text}")
                    yield from _flush(sentence_batch)
            logger.info(f"Tools: {state.tools}")


    def _consume_nonstreaming(self, events: Iterator[ProviderEvent], state: _GenState, turn: _Turn) -> Iterator[LLMOut]:
        if self._generation_is_stale(turn.gen) or not self._turn_is_latest(turn.turn_id, turn.turn_revision):
            logger.info("LLM generation cancelled (interruption)")
            return
        for event in events:
            if isinstance(event, Usage):
                state.input_tokens = event.input_tokens
                state.output_tokens = event.output_tokens
                if event.decode_tok_s is not None:
                    state.decode_tok_s = event.decode_tok_s
                if event.prefill_tok_s is not None:
                    state.prefill_tok_s = event.prefill_tok_s
                if event.llm_tpot_ms is not None:
                    state.llm_tpot_ms = event.llm_tpot_ms
            elif isinstance(event, AssistantMessage):
                state.pending.append(
                    RealtimeConversationItemAssistantMessage(type="message", role="assistant", content=event.content)
                )
            elif isinstance(event, ToolCall):
                yield from self._record_tool_call(state, turn, event.item)
            elif isinstance(event, TextDelta):
                # Text-only keeps every character verbatim; audio strips
                # TTS-unfriendly symbols via remove_unspeechable.
                spoken = event.text if not turn.wants_audio else remove_unspeechable(event.text)
                state.clean_text += spoken
                out = spoken if not turn.wants_audio else spoken.strip()
                if (
                    out
                    and not self._generation_is_stale(turn.gen)
                    and self._turn_output_allowed(turn.turn_id, turn.turn_revision)
                ):
                    yield self._chunk(turn, text=out)
        logger.debug(f"Clean text: {state.clean_text}")
        logger.info(f"Tools: {state.tools}")


    # ── orchestration ─────────────────────────────────────────────────────────

    def _generate(
        self,
        active_chat: Chat,
        original_chat: Chat,
        turn: _Turn,
        optional_kwargs: dict[str, Any],
    ) -> Iterator[LLMOut]:
        api_response: Any = None
        state = _GenState()
        error_message: str | None = None
        api_input = self._serialize(active_chat)
        # Images the model actually sees this turn; only these are stripped on
        # write-back, so an image a fast client injects mid-generation for the
        # next turn survives (it is not in this serialized snapshot).
        consumed_image_ids = active_chat.image_message_ids()
        if not api_input:
            # Nothing to send: empty `instructions` and no `input` (in the response,
            # the default conversation, or the out-of-band context). The provider
            # would reject this; fail with a clear message instead of an opaque error.
            error_message = "Cannot generate a response: no instructions and no input were provided."

        try:
            if error_message is None:
                api_response = self._request(api_input, optional_kwargs)
            if api_response is not None:
                events = self._iter_events(api_response)
                if self.stream:
                    yield from self._consume_streaming(events, state, turn)
                else:
                    yield from self._consume_nonstreaming(events, state, turn)
        except httpx.ReadTimeout:
            logger.warning(
                "OpenAI API read timed out after %.1fs; ending the current response",
                self.request_timeout_s,
            )
            if not self._generation_is_stale(turn.gen) and self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                # Canned apology carries no language_code (mirrors the prior handlers).
                yield LLMResponseChunk(
                    text="Wow I'm a bit slow today, could you repeat that?",
                    runtime_config=turn.runtime_config,
                    response=turn.response,
                    turn_id=turn.turn_id,
                    turn_revision=turn.turn_revision,
                    speech_stopped_at_s=turn.speech_stopped_at_s,
                    cancel_generation=turn.gen,
                )
        except Exception as exc:
            # Any other generation failure must still terminate the response: record
            # the error and fall through to the EndOfResponse below. Without this the
            # exception would escape process() and no EndOfResponse would be emitted,
            # leaving st.in_response stuck and locking every subsequent response.
            logger.exception("LLM generation failed; ending the current response")
            if error_message is None:
                error_message = f"Language model generation failed: {exc}"
        finally:
            if api_response is not None and hasattr(api_response, "close"):
                try:
                    api_response.close()
                except Exception:
                    pass

        if (
            error_message is None
            and not self._generation_is_stale(turn.gen)
            and self._turn_output_allowed(turn.turn_id, turn.turn_revision)
        ):
            # Out-of-band responses emit output and usage but never write back to the
            # default conversation (their context was a throwaway chat).
            if not is_out_of_band(turn.response):
                # Tool calls (and any assistant text preceding them) were already
                # written eagerly in _record_tool_call; only trailing items remain.
                for item in state.pending:
                    original_chat.add_item(item)
                original_chat.strip_images(consumed_image_ids)
                original_chat.trim_if_needed(self.compactor)
            if state.input_tokens or state.output_tokens:
                yield TokenUsage(
                    input_tokens=state.input_tokens,
                    output_tokens=state.output_tokens,
                    decode_tok_s=state.decode_tok_s,
                    prefill_tok_s=state.prefill_tok_s,
                    llm_tpot_ms=state.llm_tpot_ms,
                    turn_id=turn.turn_id,
                    turn_revision=turn.turn_revision,
                )
        yield EndOfResponse(
            turn_id=turn.turn_id, turn_revision=turn.turn_revision, cancel_generation=turn.gen, error=error_message
        )

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        """Process a language model request and yield LLMResponseChunks."""
        runtime_config = request.runtime_config
        response = request.response
        turn_id = request.turn_id
        turn_revision = request.turn_revision
        speech_stopped_at_s = request.speech_stopped_at_s
        if not self._turn_is_latest(turn_id, turn_revision):
            logger.info("Skipping stale LLM request for turn=%s rev=%s", turn_id, turn_revision)
            yield EndOfResponse(turn_id=turn_id, turn_revision=turn_revision)
            return

        original_chat = runtime_config.chat
        if is_out_of_band(response):
            try:
                active_chat = build_active_chat(original_chat, response)
            except ChatItemError as exc:
                logger.info("Out-of-band response rejected: %s", exc)
                yield EndOfResponse(turn_id=turn_id, turn_revision=turn_revision, error=str(exc))
                return
        else:
            active_chat = original_chat.copy()
        language_code = request.language_code
        instructions = (
            response.instructions if response and response.instructions else runtime_config.session.instructions
        ) or ""
        req_tools = response.tools if response and response.tools else runtime_config.session.tools
        req_tool_choice = (
            response.tool_choice if response and response.tool_choice else runtime_config.session.tool_choice
        )
        # Server-side Nova route (wins the race against client session.update).
        # Only when the chat tip is a fresh user utterance — skip tool follow-ups.
        user_text = _last_user_text(active_chat)
        routed_tools, routed_choice = _nova_route_override(user_text)
        if routed_tools is not None:
            req_tools = routed_tools
            req_tool_choice = routed_choice
            names = [t.get("name") if isinstance(t, dict) else getattr(t, "name", "?") for t in routed_tools]
            logger.info(
                "Nova route override: tool_choice=%s tools=%s",
                routed_choice,
                names,
            )
        # DriveAuth: payment utterances authorize before the 230M tool agent.
        # STEP_UP / REJECT short-circuit routing with a grounded speak turn.
        session_id = getattr(runtime_config, "session_id", None) or os.environ.get(
            "NOVA_SESSION_ID"
        )
        auth = _nova_auth_precheck(user_text, session_id if isinstance(session_id, str) else None)
        if auth is not None and auth.get("status") in {"step_up_required", "denied"}:
            speak = (auth.get("speak") or auth.get("prompt") or "").strip()
            if speak:
                instructions = (
                    instructions
                    + "\n\n[Authorization] Speak this to the driver exactly; do not call tools:\n"
                    + speak
                ).strip()
                active_chat.add_item(
                    make_user_message(
                        "[Authorization result — speak to the driver now. Do not call tools.]\n"
                        + speak
                    )
                )
            req_tools = []
            req_tool_choice = "none"
            logger.info(
                "Nova DriveAuth short-circuit: status=%s rule=%s",
                auth.get("status"),
                auth.get("rule"),
            )
            agent = None
        else:
            agent = None
        # Model mode: 230M picks+executes tools; 350M only articulates.
        # Speak turn must see ZERO tools — nonempty placeholders still prime
        # LFM to emit <|tool_call_start|>… as spoken text (live failure).
        if agent is None and (auth is None or auth.get("status") in {None, "bypass", "accept"}):
            agent = _nova_tool_agent(
                user_text,
                session_id if isinstance(session_id, str) else None,
            )
        nova_speak_direct = ""
        if agent is not None:
            speak_extra = (agent.get("speak_instructions") or "").strip()
            speak_payload = (agent.get("speak_payload") or "").strip()
            # Factual speak text: TTS verbatim (350M was echoing tool names / ignoring payload).
            if speak_payload:
                nova_speak_direct = speak_payload
            elif speak_extra:
                instructions = (instructions + "\n\n" + speak_extra).strip()
            req_tools = []
            req_tool_choice = "none"
            logger.info(
                "Nova tool agent: needs_tools=%s skipped=%s calls=%s",
                agent.get("needs_tools"),
                agent.get("skipped"),
                [c.get("name") for c in (agent.get("tool_calls") or [])],
            )
        wants_audio = response_wants_audio(response)
        self._apply_config(active_chat, instructions, wants_audio)
        language_code, lang_name = resolve_auto_language(language_code)
        if lang_name and self.enable_lang_prompt:
            active_chat.add_item(make_user_message(f"Please reply to my message in {lang_name}."))

        # CancelScope.is_stale(gen) is checked when the stream iterator advances; a
        # blocked read inside httpx cannot be aborted by cancel_scope.cancel() from
        # the websocket router. Mitigations: request_timeout_s / ReadTimeout.
        gen = self.cancel_scope.generation if self.cancel_scope else None

        turn = _Turn(
            language_code=language_code,
            gen=gen,
            runtime_config=runtime_config,
            response=response,
            turn_id=turn_id,
            turn_revision=turn_revision,
            speech_stopped_at_s=speech_stopped_at_s,
            wants_audio=wants_audio,
        )


        # Speak tool/auth payloads verbatim — do not re-ask the articulator LLM.
        if nova_speak_direct:
            logger.info("Nova speak short-circuit: %d chars", len(nova_speak_direct))
            if (
                not self._generation_is_stale(turn.gen)
                and self._turn_output_allowed(turn.turn_id, turn.turn_revision)
            ):
                yield self._chunk(turn, text=nova_speak_direct)
                if not is_out_of_band(turn.response):
                    original_chat.add_item(
                        RealtimeConversationItemAssistantMessage(
                            type="message",
                            role="assistant",
                            content=[AssistantContent(type="output_text", text=nova_speak_direct)],
                        )
                    )
                    original_chat.trim_if_needed(self.compactor)
            yield EndOfResponse(
                turn_id=turn.turn_id,
                turn_revision=turn.turn_revision,
                cancel_generation=turn.gen,
            )
            return

        # Gemma/llama frequently ignores named tool_choice and narrates. When Nova
        # forces a specific function and we can fill args (empty or simple heuristics),
        # emit the tool call without asking the model.
        # Set NOVA_FORCE_TOOL_SHORTCIRCUIT=0 to disable (pair with NOVA_TOOL_ROUTE_MODE=full).
        shortcircuit_on = os.environ.get("NOVA_FORCE_TOOL_SHORTCIRCUIT", "1").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        if (
            shortcircuit_on
            and isinstance(req_tool_choice, dict)
            and req_tool_choice.get("type") == "function"
            and req_tool_choice.get("name")
            and req_tools
        ):
            fname = str(req_tool_choice["name"])
            schema = next((t for t in req_tools if _tool_name(t) == fname), None)
            if schema is not None:
                args = _args_for_forced_tool(fname, schema, user_text)
                if args is not None:
                    logger.info("Nova forced tool short-circuit: %s %s", fname, args)
                    state = _GenState()
                    yield from self._record_tool_call(state, turn, _forced_tool_call(fname, args))
                    logger.info(f"Tools: {state.tools}")
                    yield EndOfResponse(
                        turn_id=turn.turn_id,
                        turn_revision=turn.turn_revision,
                        cancel_generation=turn.gen,
                    )
                    return

        optional_kwargs = self._build_optional_kwargs(req_tools, req_tool_choice)
        yield from self._generate(active_chat, original_chat, turn, optional_kwargs)

    @property
    def timing_log_level(self) -> int:
        return logging.INFO

    def should_log_timing(self, output: LLMOut) -> bool:
        return isinstance(output, LLMResponseChunk) and self.last_time > self.min_time_to_debug
