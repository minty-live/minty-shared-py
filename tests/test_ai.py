"""minty_shared.ai: the engine every document reader shares. Lifted with the generic cases of
Flask's tests/test_expense_ai.py (2026-11-04); each reader keeps its own tests for its prompt,
context and the check of the reply against the lists it sent. Since 2026-11-20 the route is Vertex
AI's ``models.generate_content`` only, and a reader's settings come from ``ai_policy``.

No network: the client is a stand-in throughout - a test that reached Google would test Google.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("google.genai")

from pydantic import BaseModel  # noqa: E402

from minty_shared import ai  # noqa: E402

CONFIG = ai.ReaderConfig(label="test_ai", enabled=True)
MODELS = ["gemini-3.8-flash"]


class Reply(BaseModel):
    amount_value: float = 0.0
    contact_id: str = ""


SENT_CONTACTS = {"con-sf"}


def _validate(parsed: Reply) -> dict:
    contact = parsed.contact_id if parsed.contact_id in SENT_CONTACTS else ""
    amount = parsed.amount_value if 0 < parsed.amount_value <= 10_000_000 else ""
    return {"amount": ai.checked(amount), "contact": ai.checked(contact)}


def _reply(**over) -> str:
    body = {"amount_value": 120.0, "contact_id": "con-sf"}
    body.update(over)
    return json.dumps(body)


class FakeUsage:
    prompt_token_count = 5000
    candidates_token_count = 300
    thoughts_token_count = 400
    cached_content_token_count = 0


class FakeResponse:
    """A ``GenerateContentResponse`` in the shape the engine reads: one candidate whose parts are
    the model's thinking (if any) then its reply, or a prompt blocked before any candidate."""

    def __init__(self, text=None, finish="STOP", block=None, thought=None, candidates=True):
        self.response_id = "response-1"
        self.usage_metadata = FakeUsage()
        self.prompt_feedback = SimpleNamespace(block_reason=block) if block else None
        parts = [SimpleNamespace(text=thought, thought=True)] if thought else []
        if text is not None:
            parts.append(SimpleNamespace(text=text, thought=None))
        content = SimpleNamespace(parts=parts)
        self.candidates = (
            [SimpleNamespace(finish_reason=finish, content=content)] if candidates else []
        )


class FakeModels:
    def __init__(self, result):
        self._result = result
        self.calls = 0
        self.last_request = None

    def generate_content(self, **kwargs):
        self.calls += 1
        self.last_request = kwargs
        if isinstance(self._result, Exception):
            raise self._result
        if callable(self._result):
            return self._result(self.calls)
        return self._result


class FakeClient:
    def __init__(self, result):
        self.models = FakeModels(result)


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(ai.time, "sleep", lambda _s: None)
    ai.reset_client()
    yield
    ai.reset_client()


def _run(monkeypatch, result, config=CONFIG, document=b"%PDF-1.4 fake", mime="application/pdf"):
    client = FakeClient(result)
    monkeypatch.setattr(ai, "get_client", lambda timeout_s: client)
    extraction = ai.read(
        document,
        mime,
        config=config,
        instruction="Read it.",
        prefix_text="## Lists",
        closing_text="Go.",
        reply_model=Reply,
        validate=_validate,
        models=MODELS,
    )
    return extraction, client


def _api_error(code, message):
    """A real SDK status error: ``generate_content`` raises ``google.genai.errors``' ClientError
    (4xx) or ServerError (5xx), each carrying ``.code``."""
    from google.genai import errors

    body = {"error": {"code": code, "message": message, "status": "X"}}
    return (errors.ClientError if code < 500 else errors.ServerError)(code, body)


# --- the request ---------------------------------------------------------------------------------


def test_a_good_reply_becomes_fields_marked_to_check(monkeypatch):
    extraction, _ = _run(monkeypatch, FakeResponse(_reply()))
    assert extraction.reason is None
    assert extraction.suggestions["amount"] == {
        "value": 120.0,
        "confidence": 0.0,
        "band": "medium",
        "applied": True,
    }
    assert extraction.answer() == {"suggestions": extraction.suggestions, "reason": None}


def test_the_stable_prefix_goes_first_then_the_document_then_the_closing(monkeypatch):
    _, client = _run(monkeypatch, FakeResponse(_reply()))
    request = client.models.last_request
    assert request["model"] == "gemini-3.8-flash"
    (content,) = request["contents"]
    first, document, closing = content.parts
    assert first.text == "## Lists" and closing.text == "Go."
    assert document.inline_data.mime_type == "application/pdf"
    assert document.inline_data.data == b"%PDF-1.4 fake"
    assert request["config"].system_instruction == "Read it."


def test_the_reply_is_json_to_a_schema_that_requires_every_field_and_carries_no_titles(monkeypatch):
    _, client = _run(monkeypatch, FakeResponse(_reply()))
    config = client.models.last_request["config"]
    assert config.response_mime_type == "application/json"
    schema = config.response_json_schema
    assert schema["required"] == sorted(Reply.model_fields)
    assert all("title" not in p and "default" not in p for p in schema["properties"].values())


def test_an_image_is_sent_with_its_own_type(monkeypatch):
    _, client = _run(
        monkeypatch, FakeResponse(_reply()), document=b"\xff\xd8\xff", mime="image/jpeg"
    )
    (content,) = client.models.last_request["contents"]
    assert content.parts[1].inline_data.mime_type == "image/jpeg"


def test_the_policy_s_budget_and_thinking_level_are_sent(monkeypatch):
    config = ai.ReaderConfig(
        label="test_ai", enabled=True, max_output_tokens=6000, thinking_level="medium"
    )
    _, client = _run(monkeypatch, FakeResponse(_reply()), config=config)
    sent = client.models.last_request["config"]
    assert sent.max_output_tokens == 6000
    assert sent.thinking_config.thinking_level.name == "MEDIUM"
    assert sent.automatic_function_calling.disable is True


def test_the_model_s_thinking_is_not_read_as_the_reply(monkeypatch):
    extraction, _ = _run(monkeypatch, FakeResponse(_reply(), thought="Let me think: {not json"))
    assert extraction.reason is None and extraction.suggestions["contact"]["value"] == "con-sf"


def test_nothing_usable_keeps_the_suggestions_and_says_why(monkeypatch):
    extraction, _ = _run(monkeypatch, FakeResponse(_reply(amount_value=0, contact_id="nope")))
    assert extraction.reason == ai.REASON_NO_USABLE_FIELD
    assert extraction.answer() == {
        "suggestions": extraction.suggestions,
        "reason": ai.REASON_NO_USABLE_FIELD,
    }


# --- the failure matrix: every one is "no suggestion" ------------------------------------------


def test_malformed_json_is_discarded_whole(monkeypatch):
    extraction, _ = _run(monkeypatch, FakeResponse("not json at all"))
    assert extraction.suggestions is None and extraction.reason == ai.REASON_INVALID_REPLY
    assert extraction.answer() == {"suggestions": None, "reason": ai.REASON_INVALID_REPLY}


def test_a_reply_failing_the_schema_is_discarded_whole(monkeypatch):
    extraction, _ = _run(monkeypatch, FakeResponse(json.dumps({"amount_value": "lots"})))
    assert extraction.suggestions is None and extraction.reason == ai.REASON_INVALID_REPLY


def test_an_empty_reply_is_invalid(monkeypatch):
    extraction, _ = _run(monkeypatch, FakeResponse(""))
    assert extraction.reason == ai.REASON_INVALID_REPLY


@pytest.mark.parametrize("finish", ["SAFETY", "PROHIBITED_CONTENT", "RECITATION", "SPII"])
def test_a_withheld_reply_is_a_safety_block_and_not_retried(monkeypatch, finish):
    extraction, client = _run(monkeypatch, FakeResponse(None, finish=finish))
    assert extraction.reason == ai.REASON_SAFETY_BLOCKED and client.models.calls == 1


def test_a_blocked_document_is_a_safety_block(monkeypatch):
    extraction, client = _run(
        monkeypatch, FakeResponse(block="PROHIBITED_CONTENT", candidates=False)
    )
    assert extraction.reason == ai.REASON_SAFETY_BLOCKED and client.models.calls == 1


def test_a_truncated_reply_is_discarded(monkeypatch):
    extraction, _ = _run(monkeypatch, FakeResponse('{"amount_value": 12', finish="MAX_TOKENS"))
    assert extraction.reason == ai.REASON_TRUNCATED


def test_the_sdk_s_enum_finish_reason_is_read_by_name(monkeypatch):
    from google.genai import types

    extraction, _ = _run(monkeypatch, FakeResponse(_reply(), finish=types.FinishReason.STOP))
    assert extraction.reason is None
    extraction, _ = _run(monkeypatch, FakeResponse("{", finish=types.FinishReason.MAX_TOKENS))
    assert extraction.reason == ai.REASON_TRUNCATED


@pytest.mark.parametrize(
    "response", [FakeResponse(None, finish="OTHER"), FakeResponse(candidates=False)]
)
def test_any_other_ending_is_a_provider_error(monkeypatch, response):
    extraction, _ = _run(monkeypatch, response)
    assert extraction.reason == ai.REASON_PROVIDER_ERROR


def test_a_timeout_is_not_retried(monkeypatch):
    extraction, client = _run(monkeypatch, TimeoutError("deadline exceeded"))
    assert extraction.reason == ai.REASON_TIMEOUT and client.models.calls == 1


def test_an_sdk_timeout_is_recognised_by_type_and_not_retried(monkeypatch):
    import httpx

    extraction, client = _run(monkeypatch, httpx.ReadTimeout("timed out"))
    assert extraction.reason == ai.REASON_TIMEOUT and client.models.calls == 1


@pytest.mark.parametrize("status", [404, 403, 401, 400])
def test_a_client_error_is_not_retried(monkeypatch, status):
    extraction, client = _run(monkeypatch, _api_error(status, "no"))
    assert extraction.reason == ai.REASON_PROVIDER_ERROR and client.models.calls == 1


def test_a_429_is_rate_limited_not_a_transport_error(monkeypatch):
    extraction, _ = _run(monkeypatch, _api_error(429, "quota exceeded"))
    assert extraction.reason == ai.REASON_RATE_LIMITED


def test_a_429_does_not_retry_when_the_advised_wait_blows_the_deadline(monkeypatch):
    slept = []
    monkeypatch.setattr(ai.time, "sleep", slept.append)
    error = _api_error(429, "Quota exceeded. Please retry in 39.6s.")
    extraction, client = _run(monkeypatch, error)
    assert extraction.reason == ai.REASON_RATE_LIMITED
    assert client.models.calls == 1 and slept == []


def test_a_429_retries_once_when_the_advised_wait_fits(monkeypatch):
    def flaky(call_number):
        if call_number == 1:
            raise _api_error(429, "Slow down. Please retry in 1.5s.")
        return FakeResponse(_reply())

    extraction, client = _run(monkeypatch, flaky)
    assert extraction.reason is None and client.models.calls == 2


def test_a_server_error_retries_once_then_succeeds(monkeypatch):
    def flaky(call_number):
        if call_number == 1:
            raise _api_error(503, "unavailable")
        return FakeResponse(_reply())

    extraction, client = _run(monkeypatch, flaky)
    assert extraction.reason is None and client.models.calls == 2


def test_a_transport_error_twice_is_a_provider_error(monkeypatch):
    extraction, client = _run(monkeypatch, ConnectionError("reset"))
    assert extraction.reason == ai.REASON_PROVIDER_ERROR and client.models.calls == 2


def test_an_unconfigured_client_makes_no_call(monkeypatch):
    def boom(timeout_s):
        raise RuntimeError("Vertex AI is not configured")

    monkeypatch.setattr(ai, "get_client", boom)
    extraction = ai.read(
        b"%PDF",
        "application/pdf",
        config=CONFIG,
        instruction="",
        prefix_text="",
        closing_text="",
        reply_model=Reply,
        validate=_validate,
        models=MODELS,
    )
    assert extraction.suggestions is None and extraction.reason == ai.REASON_NOT_CONFIGURED


# --- usage and cost ---------------------------------------------------------------------------


def test_thinking_tokens_are_recorded_separately(monkeypatch):
    extraction, _ = _run(monkeypatch, FakeResponse(_reply()))
    assert extraction.audit["input_tokens"] == 5000
    assert extraction.audit["output_tokens"] == 300 and extraction.audit["thought_tokens"] == 400
    assert extraction.audit["provider_request_id"] == "response-1"


def test_cost_bills_thinking_at_the_output_rate(monkeypatch):
    extraction, _ = _run(monkeypatch, FakeResponse(_reply()))
    expected = (5000 / 1_000_000) * 0.75 + ((300 + 400) / 1_000_000) * 3.75
    assert extraction.audit["estimated_cost"] == round(expected, 6)


def test_an_unpriced_model_has_no_cost():
    assert ai.estimate_cost("gemini-unknown", 100, 100) is None


def test_the_log_line_carries_numbers_never_values(monkeypatch, caplog):
    caplog.set_level("INFO", logger="minty-shared.ai")
    extraction, _ = _run(monkeypatch, FakeResponse(_reply(amount_value=4321.5)))
    ai.log_read(CONFIG, extraction, entity_id="ent-1", user_id="user-1")
    line = caplog.records[-1].getMessage()
    assert line.startswith(
        "test_ai: extract entity=ent-1 user=user-1 model=gemini-3.8-flash "
        "attempts=gemini-3.8-flash:ok location=global"
    )
    assert "tokens_in=5000" in line and "status=ok" in line
    assert "4321" not in line and "con-sf" not in line


# --- sniffing ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data, expected",
    [
        (b"%PDF-1.7 ...", "application/pdf"),
        (b"\xff\xd8\xff\xe0 JFIF", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n rest", "image/png"),
        (b"MZ\x90\x00 an executable named receipt.pdf", None),
        (b"", None),
    ],
)
def test_the_type_is_sniffed_from_the_bytes(data, expected):
    assert ai.sniff_mime(data) == expected


# --- the route: Vertex AI, always on its global endpoint (owner, 2026-11-23) -------------------


def test_a_reader_is_off_unless_its_settings_say_so():
    assert ai.ReaderConfig(label="x").enabled is False
    assert ai.ReaderConfig(label="x", max_file_mb=3).max_file_bytes == 3 * 1024 * 1024


@pytest.mark.parametrize("project", [None, " "])
def test_no_call_without_a_project(monkeypatch, project):
    if project is None:
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    else:
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", project)
    with pytest.raises(RuntimeError, match="Vertex AI is not configured"):
        ai.get_client(25)


def test_the_client_is_vertex_ai_on_the_global_endpoint(monkeypatch):
    """Owner, 2026-11-23: no region setting - global serves every model (Flash-Lite is not in
    asia-southeast1)."""
    from google import genai

    made = {}

    class Recorder:
        def __init__(self, **kwargs):
            made.update(kwargs)

    monkeypatch.setattr(genai, "Client", Recorder)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "minty-515207")
    monkeypatch.setenv("AI_LOCATION", "asia-southeast1")  # an old setting: ignored
    assert ai.project() == "minty-515207"
    client = ai.get_client(25)
    assert made["vertexai"] is True
    assert (made["project"], made["location"]) == ("minty-515207", "global")
    assert ai.VERTEX_LOCATION == "global"
    assert made["http_options"].timeout == 25_000
    assert ai.get_client(25) is client  # one per process and timeout
    assert ai.get_client(30) is not client


def test_every_read_is_on_the_global_endpoint(monkeypatch):
    asked = []
    client = FakeClient(FakeResponse(_reply()))
    monkeypatch.setattr(ai, "get_client", lambda timeout_s: asked.append(timeout_s) or client)
    config = ai.ReaderConfig(label="test_ai", enabled=True, timeout_seconds=30)
    extraction = ai.read(
        b"%PDF",
        "application/pdf",
        config=config,
        instruction="",
        prefix_text="",
        closing_text="",
        reply_model=Reply,
        validate=_validate,
        models=MODELS,
    )
    assert asked == [30] and extraction.audit["location"] == "global"
    assert not hasattr(ai.ReaderConfig(label="x"), "location")


def test_no_model_named_means_no_call(monkeypatch):
    client = FakeClient(FakeResponse(_reply()))
    monkeypatch.setattr(ai, "get_client", lambda timeout_s: client)
    extraction = ai.read(
        b"%PDF",
        "application/pdf",
        config=CONFIG,
        instruction="",
        prefix_text="",
        closing_text="",
        reply_model=Reply,
        validate=_validate,
        models=[],
    )
    assert extraction.reason == ai.REASON_NOT_CONFIGURED and client.models.calls == 0


# --- models: failover and approved providers ---------------------------------------------------


class FakeProvider:
    """A provider answering per model: a reason string fails that model, a reply text answers."""

    name = "gemini"

    def __init__(self, answers):
        self.answers = answers
        self.calls: list[str] = []

    def call(self, model, parts, config):
        self.calls.append(model)
        answer = self.answers[model]
        usage = {"input_tokens": 1000, "output_tokens": 100, "thought_tokens": 0}
        if answer in ai.FAILOVER_REASONS or answer == ai.REASON_SAFETY_BLOCKED:
            return ai.ProviderReply(reason=answer, usage=usage)
        return ai.ProviderReply(text=answer, usage=usage)


def _read_with(monkeypatch, answers, models):
    provider = FakeProvider(answers)
    monkeypatch.setitem(ai.PROVIDERS, "gemini", provider)
    extraction = ai.read(
        b"%PDF-1.4 fake",
        "application/pdf",
        config=CONFIG,
        instruction="Read it.",
        prefix_text="## Lists",
        closing_text="Go.",
        reply_model=Reply,
        validate=_validate,
        models=models,
    )
    return extraction, provider


@pytest.mark.parametrize(
    "failure",
    [
        ai.REASON_TIMEOUT,
        ai.REASON_PROVIDER_ERROR,
        ai.REASON_RATE_LIMITED,
        ai.REASON_TRUNCATED,
        ai.REASON_INVALID_REPLY,
    ],
)
def test_the_backup_answers_when_the_primary_fails(monkeypatch, failure):
    extraction, provider = _read_with(
        monkeypatch,
        {"gemini-3.5-flash-lite": failure, "gemini-3.8-flash": _reply()},
        ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
    )
    assert provider.calls == ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    assert extraction.suggestions is not None and extraction.reason is None
    assert extraction.answered and extraction.audit["model_answered"] == "gemini-3.8-flash"
    assert extraction.attempts_text() == f"gemini-3.5-flash-lite:{failure}>gemini-3.8-flash:ok"


def test_a_reply_that_is_not_json_moves_to_the_backup(monkeypatch):
    extraction, provider = _read_with(
        monkeypatch,
        {"gemini-3.5-flash-lite": "{not json", "gemini-3.8-flash": _reply()},
        ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
    )
    assert extraction.audit["model_answered"] == "gemini-3.8-flash"
    assert extraction.attempts_text().startswith("gemini-3.5-flash-lite:invalid_reply>")


def test_the_backup_is_not_called_when_the_primary_answers(monkeypatch):
    extraction, provider = _read_with(
        monkeypatch,
        {"gemini-3.5-flash-lite": _reply(), "gemini-3.8-flash": _reply()},
        ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
    )
    assert provider.calls == ["gemini-3.5-flash-lite"]
    assert extraction.audit["model_answered"] == "gemini-3.5-flash-lite"


def test_a_safety_block_does_not_fail_over(monkeypatch):
    extraction, provider = _read_with(
        monkeypatch,
        {"gemini-3.5-flash-lite": ai.REASON_SAFETY_BLOCKED, "gemini-3.8-flash": _reply()},
        ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
    )
    assert provider.calls == ["gemini-3.5-flash-lite"]
    assert extraction.reason == ai.REASON_SAFETY_BLOCKED and not extraction.answered


def test_nothing_usable_is_an_answer_and_does_not_fail_over(monkeypatch):
    extraction, provider = _read_with(
        monkeypatch,
        {
            "gemini-3.5-flash-lite": _reply(amount_value=0, contact_id="nope"),
            "gemini-3.8-flash": _reply(),
        },
        ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
    )
    assert provider.calls == ["gemini-3.5-flash-lite"]
    assert extraction.reason == ai.REASON_NO_USABLE_FIELD and extraction.answered


def test_both_failing_is_a_failed_read_with_every_token_counted(monkeypatch):
    extraction, provider = _read_with(
        monkeypatch,
        {"gemini-3.5-flash-lite": ai.REASON_TIMEOUT, "gemini-3.8-flash": ai.REASON_PROVIDER_ERROR},
        ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
    )
    assert extraction.suggestions is None and extraction.reason == ai.REASON_PROVIDER_ERROR
    assert not extraction.answered
    assert extraction.tokens() == 2 * 1100
    assert extraction.attempts_text() == (
        "gemini-3.5-flash-lite:timeout>gemini-3.8-flash:provider_error"
    )


def test_cost_sums_each_attempt_at_its_own_model_s_price(monkeypatch):
    extraction, _ = _read_with(
        monkeypatch,
        {"gemini-3.5-flash-lite": ai.REASON_TIMEOUT, "gemini-3.8-flash": _reply()},
        ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
    )
    lite = ai.estimate_cost("gemini-3.5-flash-lite", 1000, 100, 0)
    flash = ai.estimate_cost("gemini-3.8-flash", 1000, 100, 0)
    assert extraction.audit["estimated_cost"] == round(lite + flash, 6)


@pytest.mark.parametrize("model", ["openai:gpt-6.1-terra", "claude-sonnet-5-5", "gpt-6-luna"])
def test_a_model_of_an_unapproved_provider_is_never_called(monkeypatch, model):
    extraction, provider = _read_with(
        monkeypatch, {"gemini-3.8-flash": _reply()}, [model, "gemini-3.8-flash"]
    )
    assert provider.calls == ["gemini-3.8-flash"]
    assert extraction.attempts_text() == f"{model}:not_configured>gemini-3.8-flash:ok"


def test_a_named_gemini_model_reaches_gemini_without_its_prefix(monkeypatch):
    extraction, provider = _read_with(
        monkeypatch, {"gemini-3.8-flash": _reply()}, ["gemini:gemini-3.8-flash"]
    )
    assert provider.calls == ["gemini-3.8-flash"] and extraction.answered


def test_the_log_line_names_the_model_that_answered_after_a_failover(monkeypatch, caplog):
    caplog.set_level("INFO", logger="minty-shared.ai")
    extraction, _ = _read_with(
        monkeypatch,
        {"gemini-3.5-flash-lite": ai.REASON_TIMEOUT, "gemini-3.8-flash": _reply()},
        ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
    )
    ai.log_read(CONFIG, extraction, entity_id="ent-1", user_id="user-1")
    line = caplog.records[-1].getMessage()
    assert (
        "model=gemini-3.8-flash attempts=gemini-3.5-flash-lite:timeout>gemini-3.8-flash:ok" in line
    )
