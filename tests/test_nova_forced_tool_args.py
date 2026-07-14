"""Nova forced-tool short-circuit arg builders (Gemma often ignores named tool_choice)."""

from __future__ import annotations

import json

from speech_to_speech.LLM.base_openai_compatible_language_model import _args_for_forced_tool


def _schema(name: str, required: list[str] | None = None, props: dict | None = None) -> dict:
    return {
        "type": "function",
        "name": name,
        "parameters": {
            "type": "object",
            "properties": props or {},
            "required": required or [],
        },
    }


def test_empty_required_emits_empty_object():
    assert _args_for_forced_tool("query_vehicle_status", _schema("query_vehicle_status"), "fuel?") == "{}"
    assert json.loads(_args_for_forced_tool("check_email", _schema("check_email"), "Check my email.")) == {
        "mode": "unread"
    }
    assert _args_for_forced_tool("list_reminders", _schema("list_reminders"), "any reminders?") == "{}"
    assert _args_for_forced_tool("get_research_result", _schema("get_research_result"), "tell me") == "{}"


def test_check_calendar_day_from_query():
    tool = _schema("check_calendar", props={"day": {"type": "string"}, "on_date": {"type": "string"}})
    assert json.loads(_args_for_forced_tool("check_calendar", tool, "check my calendar")) == {
        "day": "week"
    }
    assert json.loads(
        _args_for_forced_tool("check_calendar", tool, "what's on my calendar tomorrow")
    ) == {"day": "tomorrow"}
    assert json.loads(
        _args_for_forced_tool(
            "check_calendar", tool, "check my calendar for the day after tomorrow"
        )
    ) == {"day": "day_after_tomorrow"}
    assert json.loads(
        _args_for_forced_tool("check_calendar", tool, "check my calendar for July 16")
    ) == {"on_date": f"{__import__('datetime').datetime.now().year:04d}-07-16"}
    assert json.loads(
        _args_for_forced_tool(
            "check_calendar", tool, "Hey, check for the calendar for the date 16th July."
        )
    ) == {"on_date": f"{__import__('datetime').datetime.now().year:04d}-07-16"}


def test_check_email_latest_mode():
    tool = _schema("check_email", props={"mode": {"type": "string"}})
    assert json.loads(_args_for_forced_tool("check_email", tool, "check my emails")) == {
        "mode": "unread"
    }
    assert json.loads(
        _args_for_forced_tool("check_email", tool, "what's the latest email that I have")
    ) == {"mode": "latest"}
    assert json.loads(
        _args_for_forced_tool("check_email", tool, "summarize that unread email")
    ) == {"mode": "summarize"}


def test_create_drive_folder_name():
    tool = _schema("create_drive_folder", ["name"], {"name": {"type": "string"}})
    assert json.loads(
        _args_for_forced_tool(
            "create_drive_folder", tool, "create a directory in drive named as Nova S"
        )
    ) == {"name": "Nova S"}


def test_windows_open_and_close():
    tool = _schema("set_windows", ["open"], {"open": {"type": "boolean"}})
    assert json.loads(_args_for_forced_tool("set_windows", tool, "Roll down the windows.")) == {"open": True}
    assert json.loads(_args_for_forced_tool("set_windows", tool, "Close the windows.")) == {"open": False}


def test_weather_and_music_and_search():
    assert json.loads(
        _args_for_forced_tool("get_weather", _schema("get_weather", ["place"]), "What's the weather in Bangalore.")
    ) == {"place": "Bangalore"}
    assert json.loads(
        _args_for_forced_tool("play_music", _schema("play_music", ["query"]), "Play some jazz.")
    ) == {"query": "jazz"}
    assert json.loads(
        _args_for_forced_tool("web_search", _schema("web_search", ["query"]), "stock price of Amazon")
    ) == {"query": "stock price of Amazon"}
    news = json.loads(
        _args_for_forced_tool(
            "web_search",
            _schema("web_search", ["query"]),
            "Hey, can you tell me the current news today in Bangalore.",
        )
    )
    assert news["place"] in {"Bangalore", "Bengaluru"}
    assert news["category"] == "general"
    assert "news" in news["query"].lower()
    assert news["query"] != "Hey, can you tell me the current news today in Bangalore."


def test_send_payment_when_payee_and_amount_clear():
    tool = _schema("send_payment", ["payee", "amount"])
    args = json.loads(_args_for_forced_tool("send_payment", tool, "Pay $50 to Starbucks."))
    assert args == {"payee": "Starbucks", "amount": 50.0}
    args2 = json.loads(
        _args_for_forced_tool("send_payment", tool, "The pay is Starbucks. The amount is $50.")
    )
    assert args2 == {"payee": "Starbucks", "amount": 50.0}


def test_send_payment_incomplete_returns_none():
    tool = _schema("send_payment", ["payee", "amount"])
    assert _args_for_forced_tool("send_payment", tool, "Pay Starbucks.") is None


def test_set_hvac_required_without_heuristics_returns_none():
    tool = _schema("set_hvac", ["zone", "on"])
    assert _args_for_forced_tool("set_hvac", tool, "set driver to 20") is None
