"""Reading a document with Gemini - the engine every reader shares (owner, 2026-11-04).

Lifted from Flask's ``blueprints/report/services/expense_ai.py`` (Stage 1, docs/expense_ai/ in
Minty), whose rules it keeps. A *reader* (minty-petty-cash-api's receipts, minty-payment-request-
api's invoices) brings only what is its own - the company's lists, through ``minty_shared.ai_prompt``
(the one prompt both send), and the check of the reply against those lists - and calls ``read``.
Everything else is here:

* the reader's settings, a ``ReaderConfig`` - its switch, the call's timeout, the reply's token
  budget, the thinking level, the largest document. They are ``ai_policy``'s, given by
  ``minty_shared.django.ai_allowance.Policy.reader`` and read on every call, so they turn with an
  UPDATE, never a deploy (owner, 2026-11-20; they were each service's environment);
* the models: a read tries the first model it is given and, when that one errors, times out, is
  rate-limited or replies with something unusable, the next (the failover; owner, 2026-11-18). A
  model id names its provider (``gemini-3.5-flash``, or ``<provider>:<model>``), and only a provider
  in ``APPROVED_PROVIDERS`` is ever called - paid terms, customer content not used for training.
  Each provider is a ``Provider`` in ``PROVIDERS``; Gemini is the only one today;
* the client (one per process and timeout), the call with at most ONE retry and only for the
  retryable classes, usage and estimated cost, and which model answered;
* the reply's shape: a field is ``{value, confidence, band, applied}`` and ``applied`` is the
  page's instruction; ``checked`` makes one from a reply with no confidence - the owner's simple
  prompt, ``minty_shared.ai_prompt`` - always in the ``medium`` band.

FAILURE IS ALWAYS SILENT to the person: every error returns "no suggestion" with a ``REASON_*``
code, and the form works by hand as before. The reply is a suggestion - nothing here writes any
record. How many reads a company may make is not this module's: the readers ask
``minty_shared.django.ai_allowance`` (the shared allowance in ``ai_policy``/``ai_read``), which also
supplies the models to try. Extra ``ai``.

THE ROUTE. Vertex AI only (owner, 2026-11-20; the direct Gemini API is gone): Google Cloud's paid
terms and IAM instead of an API key, on Vertex's ``global`` endpoint (``VERTEX_LOCATION``; owner,
2026-11-23: no region setting - global serves every model, and Flash-Lite is not served in
``asia-southeast1``, the region used from 2026-11-20; a request may be processed in any Google
region). Each service's environment names only the project (``GOOGLE_CLOUD_PROJECT``, without it no
call is made), and Google's client finds the service account's key through
``GOOGLE_APPLICATION_CREDENTIALS``. The call is ``models.generate_content``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("minty-shared.ai")

# One of these accompanies every "no suggestion" answer. For the logs; the page decides what,
# if anything, the person sees.
REASON_DISABLED = "disabled"
REASON_LIMIT_REACHED = "limit_reached"  # the company's AI allowance is used up (ai_allowance)
REASON_RATE_LIMITED = "rate_limited"
REASON_UNSUPPORTED_FILE = "unsupported_file"
REASON_FILE_TOO_LARGE = "file_too_large"
REASON_NO_CONTEXT = "no_context"
REASON_NOT_CONFIGURED = "not_configured"
REASON_PROVIDER_ERROR = "provider_error"
REASON_TIMEOUT = "timeout"
REASON_SAFETY_BLOCKED = "safety_blocked"
REASON_TRUNCATED = "truncated"
REASON_INVALID_REPLY = "invalid_reply"
REASON_NO_USABLE_FIELD = "no_usable_field"
# The model answered, but the document is not one record of a payment (owner, 2026-11-23, IRD "A
# Guide to Keeping Business Records"). The page says why; the read counts (it was answered).
REASON_SEVERAL_DOCUMENTS = "several_documents"  # more than one receipt or invoice in the file
REASON_NO_DESCRIPTION = "no_description"  # what was paid for can't be described or inferred
#: What the file may be instead of a receipt or invoice (the user, 2026-11-24): the answer's
#: `document_warning`, a warning beside the suggestions - never a refusal.
DOCUMENT_WARNINGS = ("bank_payment_confirmation", "other", "unreadable")

# Content sniffing: the declared extension and the browser's type are not evidence.
_MAGIC = (
    (b"%PDF", "application/pdf"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
)

# Model list prices, USD per 1M tokens (input, output), checked 3 Sep 2026 (the Gemini API's; Vertex
# AI lists the same models at its own prices). An indication for the log's cost field, not an
# invoice; re-verify at each stage gate.
PRICING = {
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
}


# --------------------------------------------------------------------------- configuration

#: ``ai_policy.thinking_level``'s values (its CHECK), as Gemini 3.x names them.
THINKING_LEVELS = ("minimal", "low", "medium", "high")


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return None if value is None or value.strip() == "" else value.strip()


@dataclass(frozen=True)
class ReaderConfig:
    """One reader's settings - ``ai_policy``'s, from ``ai_allowance.Policy.reader`` (owner,
    2026-11-20; they were the service's environment). ``label`` tags its log lines."""

    label: str
    enabled: bool = False  # the reader's switch: a reader ships dark
    timeout_seconds: int = 25  # each model call's limit
    # Thinking tokens come out of this same budget on Gemini 3.x: a long think truncates the JSON
    # mid-object at a budget that looks ample for the reply alone.
    max_output_tokens: int = 4096
    thinking_level: str = "low"
    max_file_mb: int = 10

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024


def project() -> str | None:
    """The Google Cloud project Vertex AI bills (``GOOGLE_CLOUD_PROJECT``)."""
    return _env("GOOGLE_CLOUD_PROJECT")


# --------------------------------------------------------------------------- files


def sniff_mime(data: bytes) -> str | None:
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    return None


# --------------------------------------------------------------------------- the client

#: Where every read is sent: Vertex AI's global endpoint (owner, 2026-11-23). Recorded on every
#: read's log line.
VERTEX_LOCATION = "global"

_clients: dict[int, Any] = {}
_client_lock = threading.Lock()


def get_client(timeout_s: int) -> Any:
    """One Vertex AI client per process and timeout, on the global endpoint. Raises when the
    project is not named - no call is ever made unbilled. The credentials are Google's own lookup
    (``GOOGLE_APPLICATION_CREDENTIALS``, the service account's key file)."""
    with _client_lock:
        if timeout_s in _clients:
            return _clients[timeout_s]
        name = project()
        if not name:
            raise RuntimeError(
                "Vertex AI is not configured: set GOOGLE_CLOUD_PROJECT (and "
                "GOOGLE_APPLICATION_CREDENTIALS, the service account's key file)."
            )
        from google import genai
        from google.genai import types

        http_options = types.HttpOptions(
            timeout=timeout_s * 1000,  # milliseconds
            # The SDK retries up to four times by default - far too many for a person waiting on
            # a form. One attempt; the single retry allowed is ours, in call_model.
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        client = genai.Client(
            vertexai=True, project=name, location=VERTEX_LOCATION, http_options=http_options
        )
        logger.info("ai: Vertex AI client ready, project=%s location=%s", name, VERTEX_LOCATION)
        _clients[timeout_s] = client
        return client


def reset_client() -> None:
    """Drop the cached clients so a configuration change is picked up (tests, route flips)."""
    with _client_lock:
        _clients.clear()


# --------------------------------------------------------------------------- the reply's shape


def response_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A flat Pydantic model as a schema Gemini accepts: no ``title``/``default`` on properties,
    and every field required so the model answers each one instead of omitting the hard ones."""
    schema = model.model_json_schema()
    properties = schema.get("properties", {})
    for prop in properties.values():
        prop.pop("title", None)
        prop.pop("default", None)
    return {"type": "object", "properties": properties, "required": sorted(properties)}


def checked(value: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """One suggestion from a reply that carries no confidence (``minty_shared.ai_prompt``, the
    owner's simple prompt, 2026-11-20): filled when it has a value, and always in the ``medium``
    band - marked for the person to check. ``confidence`` stays 0.0 (none was asked)."""
    out: dict[str, Any] = {
        "value": value,
        "confidence": 0.0,
        "band": "medium",
        "applied": value not in (None, ""),
    }
    if extra:
        out.update(extra)
    return out


def document_warning(document_type: object) -> str | None:
    """The reply's ``document_type`` as a warning for the page, or None: a receipt or an invoice
    is fine in either reader, and a value the prompt didn't offer is no warning."""
    value = str(document_type or "").strip().lower()
    return value if value in DOCUMENT_WARNINGS else None


def evidence_refusal(suggestions: dict[str, Any]) -> str | None:
    """Why an answered read is not one record of a payment, or None (owner, 2026-11-23): several
    receipts or invoices in the file first (``document_count``), then nothing described or
    inferred (``description``). The readers answer ``{suggestions: None, reason}`` for either."""
    try:
        count = int(suggestions.get("document_count") or 1)
    except (TypeError, ValueError):
        count = 1
    if count > 1:
        return REASON_SEVERAL_DOCUMENTS
    description = (suggestions.get("description") or {}).get("value")
    if not str(description or "").strip():
        return REASON_NO_DESCRIPTION
    return None


# --------------------------------------------------------------------------- a supplier to add

#: How long a reader's new-supplier token holds: a local draft's life (``@minty/shared`` drafts,
#: 7 days), so a restored draft still adds its contact.
NEW_SUPPLIER_MAX_AGE_S = 7 * 86400
NEW_SUPPLIER_MAX_NAME = 100
_NEW_SUPPLIER_SALT = b"minty-shared.ai.new-supplier"


def new_supplier_token(key: str, entity_id: object, name: str, *, now: float | None = None) -> str:
    """A supplier name the reader read and the company does not have, signed for that company
    (the user, 2026-11-23: a missing contact is added automatically, when the record is saved -
    even by someone who may not add contacts by hand). The save endpoint turns it back into the
    name with ``new_supplier_name``; a name typed by hand carries no token, so the server never
    takes the page's word that the reader read it. ``key`` is the service's ``SECRET_KEY``."""
    payload = {
        "e": str(entity_id),
        "n": " ".join(str(name).split())[:NEW_SUPPLIER_MAX_NAME],
        "t": int(time.time() if now is None else now),
    }
    body = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    return f"{body}.{_b64(_new_supplier_mac(key, body))}"


def new_supplier_name(
    key: str,
    token: object,
    entity_id: object,
    *,
    max_age_s: int = NEW_SUPPLIER_MAX_AGE_S,
    now: float | None = None,
) -> str | None:
    """The name ``new_supplier_token`` signed, or None when the token is malformed, altered, for
    another company or older than ``max_age_s``."""
    if not isinstance(token, str) or token.count(".") != 1:
        return None
    body, mac = token.split(".")
    try:
        given = base64.urlsafe_b64decode(mac + "=" * (-len(mac) % 4))
    except (ValueError, binascii.Error):
        return None
    if not hmac.compare_digest(given, _new_supplier_mac(key, body)):
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        entity, name, stamp = str(payload["e"]), str(payload["n"]), int(payload["t"])
    except (ValueError, binascii.Error, KeyError, TypeError):
        return None
    age = (time.time() if now is None else now) - stamp
    if entity != str(entity_id) or not name.strip() or age < -60 or age > max_age_s:
        return None
    return name


def _new_supplier_mac(key: str, body: str) -> bytes:
    if not key:
        raise ValueError("new_supplier_token needs the service's SECRET_KEY")
    return hmac.new(key.encode() + _NEW_SUPPLIER_SALT, body.encode(), hashlib.sha256).digest()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


# --------------------------------------------------------------------------- providers


@dataclass(frozen=True)
class DocumentParts:
    """What a reader asks of a model, in a form every provider can send: the cacheable prefix
    (the company's lists) FIRST, then the document, then the closing; the instruction; the reply's
    JSON schema."""

    prefix_text: str
    document: bytes
    mime: str
    closing_text: str
    instruction: str
    schema: dict[str, Any]


@dataclass
class ProviderReply:
    """One model call in the engine's terms. ``reason`` set = nothing to read. ``usage`` holds
    ``input_tokens``, ``output_tokens``, ``thought_tokens`` and ``cached_tokens`` when known -
    tokens are billed even when the reply is unusable."""

    text: str | None = None
    reason: str | None = None
    usage: dict[str, int | None] = dataclass_field(default_factory=dict)
    request_id: str | None = None


class Provider(Protocol):
    """A model provider. ``call`` never raises: every failure is a ``REASON_*``."""

    name: str

    def call(self, model: str, parts: DocumentParts, config: ReaderConfig) -> ProviderReply: ...


class GeminiProvider:
    """Gemini on Vertex AI through ``google-genai``'s ``models.generate_content``, one client per
    process and timeout, at most one retry (``call_model``)."""

    name = "gemini"

    def call(self, model: str, parts: DocumentParts, config: ReaderConfig) -> ProviderReply:
        try:
            client = get_client(config.timeout_seconds)
            from google.genai import types
        except Exception as exc:  # noqa: BLE001 - every failure is a reason, never a raise
            logger.error("%s: client unavailable: %s", config.label, exc)
            return ProviderReply(reason=REASON_NOT_CONFIGURED)
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=parts.prefix_text),
                    types.Part.from_bytes(data=parts.document, mime_type=parts.mime),
                    types.Part.from_text(text=parts.closing_text),
                ],
            )
        ]
        settings = types.GenerateContentConfig(
            system_instruction=parts.instruction,
            response_mime_type="application/json",
            response_json_schema=parts.schema,
            max_output_tokens=config.max_output_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel[config.thinking_level.upper()]
            ),
            # no tools are offered, so nothing for the SDK to call on our behalf
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response, reason = call_model(
            lambda: client.models.generate_content(model=model, contents=contents, config=settings),
            model,
            config,
        )
        if response is None:
            return ProviderReply(reason=reason)
        reply = ProviderReply(
            usage=_gemini_usage(response), request_id=getattr(response, "response_id", None)
        )
        blocked = _name(getattr(getattr(response, "prompt_feedback", None), "block_reason", None))
        if blocked:
            logger.warning(
                "%s: the document was blocked by safety filters (%s)", config.label, blocked
            )
            reply.reason = REASON_SAFETY_BLOCKED
            return reply
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            logger.warning("%s: the model returned no candidate", config.label)
            reply.reason = REASON_PROVIDER_ERROR
            return reply
        finish = _name(getattr(candidates[0], "finish_reason", None))
        if finish == "MAX_TOKENS":
            # neither a truncation nor a block is retried
            logger.warning(
                "%s: reply truncated - output=%s thought=%s against max_output_tokens=%s. "
                "Raise ai_policy.max_output_tokens if this recurs.",
                config.label,
                reply.usage.get("output_tokens"),
                reply.usage.get("thought_tokens"),
                config.max_output_tokens,
            )
            reply.reason = REASON_TRUNCATED
            return reply
        if finish in _BLOCKED:
            logger.warning("%s: blocked by safety filters (%s)", config.label, finish)
            reply.reason = REASON_SAFETY_BLOCKED
            return reply
        if finish != "STOP":
            logger.warning("%s: the model stopped with %s", config.label, finish or "no reason")
            reply.reason = REASON_PROVIDER_ERROR
            return reply
        reply.text = _reply_text(candidates[0]) or None
        if reply.text is None:
            reply.reason = REASON_INVALID_REPLY
        return reply


#: A finish reason that means the reply was withheld - not retried, not failed over.
_BLOCKED = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
    }
)


def _name(value: Any) -> str:
    """An SDK enum's name (``FinishReason.STOP`` -> ``"STOP"``), a plain string as it is, or ""."""
    if value is None:
        return ""
    return str(getattr(value, "name", None) or value)


def _reply_text(candidate: Any) -> str:
    """The reply's text: every text part that is not the model's thinking, joined."""
    content = getattr(candidate, "content", None)
    pieces = [
        part.text
        for part in (getattr(content, "parts", None) or [])
        if getattr(part, "text", None) and not getattr(part, "thought", False)
    ]
    return "".join(pieces)


#: Every provider the engine can call, by name.
PROVIDERS: dict[str, Provider] = {"gemini": GeminiProvider()}
#: Providers approved for customer documents (paid terms, content not used for training). A model
#: of any other provider is refused before a call, whatever a setting says.
APPROVED_PROVIDERS = frozenset({"gemini"})

#: A model ending in one of these is passed over for the next model in the list. A safety block is
#: not (another model would be asked the same thing), nor a valid reply with nothing usable in it.
FAILOVER_REASONS = frozenset(
    {
        REASON_TIMEOUT,
        REASON_PROVIDER_ERROR,
        REASON_RATE_LIMITED,
        REASON_TRUNCATED,
        REASON_INVALID_REPLY,
        REASON_NOT_CONFIGURED,
    }
)


def provider_for(model: str) -> tuple[Provider | None, str]:
    """``(provider, model id to send)``. ``gemini-...`` is Gemini's; ``<provider>:<model>`` names
    any other. None when the provider is unknown or not approved."""
    name, sep, bare = model.partition(":")
    if not sep:
        name, bare = ("gemini" if model.startswith("gemini-") else ""), model
    if name not in APPROVED_PROVIDERS:
        return None, bare
    return PROVIDERS.get(name), bare


# --------------------------------------------------------------------------- the call


class Extraction:
    """What a read got back. ``suggestions`` is None on every failure."""

    def __init__(
        self,
        suggestions: dict[str, Any] | None = None,
        reason: str | None = None,
        audit: dict[str, Any] | None = None,
    ) -> None:
        self.suggestions = suggestions
        self.reason = reason
        self.audit: dict[str, Any] = audit or {}

    @property
    def answered(self) -> bool:
        """A model replied with a valid answer - the read happened, even when it found nothing
        usable. This is what the allowance counts; a failed read is never counted."""
        return bool(self.audit.get("model_answered"))

    def tokens(self) -> int:
        """Every token billed for this read, every attempt's (input, output and thinking)."""
        a = self.audit
        return sum(int(a.get(k) or 0) for k in ("input_tokens", "output_tokens", "thought_tokens"))

    def attempts_text(self) -> str:
        """``gemini-3.5-flash-lite:timeout>gemini-3.8-flash:ok`` - the models tried, in order."""
        return ">".join(self.audit.get("attempts") or [])

    def answer(self) -> dict[str, Any]:
        """The endpoint's body: ``{suggestions, reason}``, ``suggestions`` null on failure."""
        if not self.suggestions:
            return {"suggestions": None, "reason": self.reason or REASON_NO_USABLE_FIELD}
        return {"suggestions": self.suggestions, "reason": self.reason}


def read(
    document: bytes,
    mime: str,
    *,
    config: ReaderConfig,
    instruction: str,
    prefix_text: str,
    closing_text: str,
    reply_model: type[BaseModel],
    validate: Callable[[Any], dict[str, Any]],
    models: Sequence[str] | None = None,
) -> Extraction:
    """Read one prepared document. Never raises.

    ``models`` is the list to try in order (the primary, then the backup - ``ai_policy``'s); with
    none, nothing is called (``REASON_NOT_CONFIGURED``). The next model is tried only when one ends
    in a ``FAILOVER_REASONS`` reason, so a read makes at most one call per model (each with its own
    ``timeout_seconds``).

    ``prefix_text`` (the entity's lists) goes FIRST and must be byte-stable for the same entity:
    Gemini's caching is implicit prefix matching, and unstable ordering does not lower the hit
    rate, it removes it. ``validate`` turns the parsed reply into the suggestions, accepting an
    id only if it was in the lists sent - the reply is untrusted input, and a value that fails a
    check is dropped, never repaired."""
    started = time.monotonic()
    chain: list[str] = []
    for model in models or []:
        if model and model not in chain:
            chain.append(model)
    audit: dict[str, Any] = {
        "model_id": chain[0] if chain else None,
        "model_answered": None,
        "attempts": [],
        "location": VERTEX_LOCATION,
        "thinking_level": config.thinking_level,
    }
    if not chain:
        logger.error("%s: no model to read with - ai_policy names none", config.label)
    parts = DocumentParts(
        prefix_text=prefix_text,
        document=document,
        mime=mime,
        closing_text=closing_text,
        instruction=instruction,
        schema=response_schema(reply_model),
    )
    result = Extraction(reason=REASON_NOT_CONFIGURED, audit=audit)
    for index, model in enumerate(chain):
        result = _attempt(model, parts, config, reply_model, validate, audit)
        if result.reason not in FAILOVER_REASONS:
            break
        if index + 1 < len(chain):
            logger.warning(
                "%s: %s ended in %s; trying the backup %s",
                config.label,
                model,
                result.reason,
                chain[index + 1],
            )
    audit["latency_ms"] = int((time.monotonic() - started) * 1000)
    return result


def _attempt(
    model: str,
    parts: DocumentParts,
    config: ReaderConfig,
    reply_model: type[BaseModel],
    validate: Callable[[Any], dict[str, Any]],
    audit: dict[str, Any],
) -> Extraction:
    """One model's try, recorded in ``audit`` (tokens summed over attempts, the chain grown)."""
    provider, bare = provider_for(model)
    audit["model_id"] = model
    if provider is None:
        logger.error(
            "%s: %s is not a model of an approved provider (%s) - not called. Fix ai_policy.",
            config.label,
            model,
            ", ".join(sorted(APPROVED_PROVIDERS)),
        )
        return _attempted(audit, model, Extraction(reason=REASON_NOT_CONFIGURED, audit=audit))
    reply = provider.call(bare, parts, config)
    _add_usage(audit, bare, reply)
    if reply.reason or reply.text is None:
        return _attempted(
            audit, model, Extraction(reason=reply.reason or REASON_INVALID_REPLY, audit=audit)
        )
    try:
        parsed = reply_model.model_validate_json(reply.text)
    except ValidationError as exc:
        # Discarded whole - a suggestion is never repaired. The shape of the failure is logged,
        # never the reply.
        logger.warning(
            "%s: reply failed schema validation at %s",
            config.label,
            [e.get("loc") for e in exc.errors()][:6],
        )
        return _attempted(audit, model, Extraction(reason=REASON_INVALID_REPLY, audit=audit))
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s: reply was not JSON: %s", config.label, type(exc).__name__)
        return _attempted(audit, model, Extraction(reason=REASON_INVALID_REPLY, audit=audit))

    audit["model_answered"] = model
    suggestions = validate(parsed)
    audit["confidence_by_field"] = {
        key: value.get("confidence")
        for key, value in suggestions.items()
        if isinstance(value, dict)
    }
    if not any(isinstance(v, dict) and v.get("applied") for v in suggestions.values()):
        result = Extraction(suggestions=suggestions, reason=REASON_NO_USABLE_FIELD, audit=audit)
    else:
        result = Extraction(suggestions=suggestions, audit=audit)
    return _attempted(audit, model, result)


def _attempted(audit: dict[str, Any], model: str, result: Extraction) -> Extraction:
    outcome = "ok" if audit.get("model_answered") == model else (result.reason or "failed")
    audit["attempts"].append(f"{model}:{outcome}")
    return result


def _add_usage(audit: dict[str, Any], model: str, reply: ProviderReply) -> None:
    """Sum this attempt's tokens and cost into the read's - a failed attempt is billed too."""
    if reply.request_id:
        audit["provider_request_id"] = reply.request_id
    usage = reply.usage
    if not usage:
        return
    for key in ("input_tokens", "output_tokens", "thought_tokens", "cached_tokens"):
        if usage.get(key) is not None:
            audit[key] = int(audit.get(key) or 0) + int(usage[key] or 0)
    cost = estimate_cost(
        model, usage.get("input_tokens"), usage.get("output_tokens"), usage.get("thought_tokens")
    )
    if cost is not None:
        audit["estimated_cost"] = round(float(audit.get("estimated_cost") or 0) + cost, 6)


def log_read(
    config: ReaderConfig, result: Extraction, *, entity_id: object, user_id: object
) -> None:
    """One log line per read. Numbers only: confidences, tokens, timings, never a value read off
    the document (a client's financial details do not belong in logs). ``model`` is the model
    that answered (else the last one tried) and ``attempts`` the chain, so a failover shows; the
    allowance's ``ai_read`` row keeps the same facts."""
    a = result.audit
    status = "ok" if result.suggestions and not result.reason else "no_suggestion"
    logger.info(
        "%s: extract entity=%s user=%s model=%s attempts=%s location=%s latency_ms=%s "
        "tokens_in=%s tokens_out=%s thought=%s cached=%s cost=%s status=%s reason=%s "
        "confidence=%s",
        config.label,
        entity_id,
        user_id,
        a.get("model_answered") or a.get("model_id"),
        result.attempts_text() or None,
        a.get("location"),
        a.get("latency_ms"),
        a.get("input_tokens"),
        a.get("output_tokens"),
        a.get("thought_tokens"),
        a.get("cached_tokens"),
        a.get("estimated_cost"),
        status,
        result.reason,
        a.get("confidence_by_field", {}),
    )


# ``models.generate_content`` raises ``google.genai.errors.ClientError``/``ServerError`` carrying
# ``.code``. Classifying on the ``status_code``/``code`` ATTRIBUTE, not class identity, keeps a 429
# from being reported as a transport error whichever exception family the SDK uses.
_RETRY_DELAY = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)


def _error_status(exc: BaseException) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _is_timeout(exc: BaseException) -> bool:
    return any("timeout" in klass.__name__.lower() for klass in type(exc).__mro__)


def _advised_delay(exc: BaseException) -> float | None:
    """Seconds the provider asked us to wait, if it said ("Please retry in 39.6s")."""
    match = _RETRY_DELAY.search(str(exc))
    return float(match.group(1)) if match else None


def call_model(send: Callable[[], Any], model: str, config: ReaderConfig) -> tuple[Any, str | None]:
    """One call and at most one retry - only for the retryable classes, most specific first:
    a timeout has spent the budget (no retry); a 429 retries once only if the provider's advised
    wait fits the deadline; 401/403/404 and other 4xx never; 5xx and transport errors once."""
    deadline = time.monotonic() + config.timeout_seconds
    for attempt in (1, 2):
        try:
            return send(), None
        except Exception as exc:  # noqa: BLE001 - classified below by status
            status = _error_status(exc)
            if _is_timeout(exc):
                logger.warning("%s: timed out after %ss", config.label, config.timeout_seconds)
                return None, REASON_TIMEOUT
            if status == 429:
                advised = _advised_delay(exc)
                remaining = deadline - time.monotonic()
                delay = advised if advised is not None else 0.5 + random.random()
                if attempt == 1 and delay < remaining:
                    logger.warning(
                        "%s: rate limited (429), retrying in %ss", config.label, round(delay, 1)
                    )
                    time.sleep(delay)
                    continue
                logger.warning(
                    "%s: rate limited (429); advised wait %ss does not fit the %ss deadline, "
                    "giving up without retrying",
                    config.label,
                    advised,
                    config.timeout_seconds,
                )
                return None, REASON_RATE_LIMITED
            if status in (401, 403):
                logger.error(
                    "%s: PERMISSION DENIED (%s) - bad credential or IAM "
                    "misconfiguration. Page on-call.",
                    config.label,
                    status,
                )
                return None, REASON_PROVIDER_ERROR
            if status == 404:
                logger.error(
                    "%s: MODEL NOT FOUND (404) - %s is not served where we are calling "
                    "(%s). Page on-call.",
                    config.label,
                    model,
                    VERTEX_LOCATION,
                )
                return None, REASON_PROVIDER_ERROR
            if status is not None and 400 <= status < 500:
                logger.error("%s: bad request (%s): %s", config.label, status, exc)
                return None, REASON_PROVIDER_ERROR
            if status is not None and status >= 500:
                logger.warning("%s: provider %s: %s", config.label, status, exc)
            else:
                logger.warning(
                    "%s: transport error (%s): %s", config.label, type(exc).__name__, exc
                )
            if attempt == 1 and time.monotonic() < deadline:
                time.sleep(0.5 + random.random())
                continue
            return None, REASON_PROVIDER_ERROR
    return None, REASON_PROVIDER_ERROR


def _gemini_usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        # Thinking tokens: invisible in the reply, billed at the OUTPUT rate, drawn from
        # max_output_tokens. Without them a truncation by thinking looks like a verbose answer.
        "thought_tokens": getattr(usage, "thoughts_token_count", None),
        # Zero across repeated same-entity calls: the prefix is under the cache minimum or unstable.
        "cached_tokens": getattr(usage, "cached_content_token_count", None),
    }


def estimate_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    thought_tokens: int | None = None,
) -> float | None:
    """Thinking is charged at the output rate; leaving it out understated a measured call ~60%."""
    price = PRICING.get(model)
    if not price or input_tokens is None or output_tokens is None:
        return None
    billable_output = output_tokens + (thought_tokens or 0)
    return round(
        (input_tokens / 1_000_000) * price[0] + (billable_output / 1_000_000) * price[1], 6
    )
