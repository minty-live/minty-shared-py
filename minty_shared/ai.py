"""Reading a document with Gemini - the engine every reader shares (owner, 2026-11-04).

Lifted from Flask's ``blueprints/report/services/expense_ai.py`` (Stage 1, docs/expense_ai/ in
Minty), whose rules it keeps. A *reader* (minty-petty-cash-api's receipts, minty-payment-request-
api's invoices) brings only what is its own - the company's lists, through ``minty_shared.ai_prompt``
(the one prompt both send), and the check of the reply against those lists - and calls ``read``.
Everything else is here:

* configuration, per reader by prefix (``ReaderConfig("EXPENSE_AI")`` reads ``EXPENSE_AI_ENABLED``,
  ``_MODEL``, ...), read on every call so the kill switch and cut-offs turn without a deploy. The
  route to the model (Vertex or the direct API, its tier) is one setting for every reader;
* the models: a read tries the first model it is given and, when that one errors, times out, is
  rate-limited or replies with something unusable, the next (the failover; owner, 2026-11-18). A
  model id names its provider (``gemini-3.8-flash``, or ``<provider>:<model>``), and only a provider
  in ``APPROVED_PROVIDERS`` is ever called - paid terms, customer content not used for training.
  Each provider is a ``Provider`` in ``PROVIDERS``; Gemini is the only one today;
* the client (one per process and timeout), the call with at most ONE retry and only for the
  retryable classes, usage and estimated cost, and which model answered;
* the reply's shape: a field is ``{value, confidence, band, applied}``, and ``applied`` is the
  page's instruction (a low band is never prefilled; ``checked`` is a field from a reply with no
  confidence - the owner's simple prompt, ``minty_shared.ai_prompt`` - always ``medium``).

FAILURE IS ALWAYS SILENT to the person: every error returns "no suggestion" with a ``REASON_*``
code, and the form works by hand as before. The reply is a suggestion - nothing here writes any
record. How many reads a company may make is not this module's: the readers ask
``minty_shared.django.ai_allowance`` (the shared allowance in ``ai_policy``/``ai_read``), which also
supplies the models to try. Extra ``ai``.

ROUTES. Vertex AI (``GOOGLE_CLOUD_PROJECT``; IAM, a regional residency commitment) is chosen
whenever a project is set - the owner's choice (project ``minty-515207``, ``asia-southeast1``),
deferred on 2026-11-20. The route in use in both environments is the direct Gemini API on the FREE
tier, which does not meet "approved providers only" (on it Google may use submitted content and
human reviewers may read it): decide before customers' documents are read. The tier is stated in
configuration (``EXPENSE_AI_DIRECT_TIER``) and ``_check_tier_matches_reality`` shouts when Google's
own quota errors contradict it.
"""

from __future__ import annotations

import base64
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

from minty_shared.config import flag

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

# Content sniffing: the declared extension and the browser's type are not evidence.
_MAGIC = (
    (b"%PDF", "application/pdf"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
)

# Model list prices, USD per 1M tokens (input, output), checked 3 Sep 2026. An indication for
# the log's cost field, not an invoice; re-verify at each stage gate.
PRICING = {
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
}


# --------------------------------------------------------------------------- configuration


def _env(name: str, default: Any = None) -> Any:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(_env(name, default)).strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(_env(name, default)).strip())
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ReaderConfig:
    """One reader's settings, ``<prefix>_*``, read from the environment on each call."""

    prefix: str

    @property
    def label(self) -> str:
        """The log tag: ``EXPENSE_AI`` -> ``expense_ai``."""
        return self.prefix.lower()

    def enabled(self) -> bool:
        """The kill switch. Default off - a reader ships dark."""
        return flag(f"{self.prefix}_ENABLED", False, blank_is_unset=True)

    def model_id(self) -> str:
        """The model when no list is given to ``read`` - the APIs pass ``ai_policy``'s, so this is
        only the fallback. A Gemini id MUST be 3.x: ``interactions.create`` (the only API with
        implicit caching) serves 3.x only, and the 2.5 family answers 404 here though
        ``models.list`` shows it."""
        return _env(f"{self.prefix}_MODEL", "gemini-3.5-flash")

    def thinking_level(self) -> str:
        level = str(_env(f"{self.prefix}_THINKING_LEVEL", "low")).strip().lower()
        return level if level in ("low", "medium", "high") else "low"

    def max_file_bytes(self) -> int:
        return _env_int(f"{self.prefix}_MAX_FILE_MB", 10) * 1024 * 1024

    def timeout_seconds(self) -> int:
        return _env_int(f"{self.prefix}_TIMEOUT_S", 25)

    def max_output_tokens(self) -> int:
        """Thinking tokens come out of this same allowance on Gemini 3.x: a long think truncates
        the JSON mid-object at a budget that looks ample for the reply alone."""
        return _env_int(f"{self.prefix}_MAX_OUTPUT_TOKENS", 4096)

    def confidence_cutoffs(self) -> tuple[float, float]:
        """(high, medium). Set from measured data before a pilot, never guessed."""
        return (
            _env_float(f"{self.prefix}_CONF_HIGH", 0.85),
            _env_float(f"{self.prefix}_CONF_MEDIUM", 0.60),
        )


def uses_vertex() -> bool:
    """Vertex is the route unless someone deliberately opted out of it."""
    if _env("GOOGLE_CLOUD_PROJECT"):
        return flag("EXPENSE_AI_USE_VERTEX", True, blank_is_unset=True)
    return False


def direct_tier() -> str:
    """The Gemini API terms we run under, "paid" or "free". Nothing at the call site can tell them
    apart, so the safe default for an unconfigured key is the restrictive one: claiming "paid" is
    a deliberate act."""
    tier = str(_env("EXPENSE_AI_DIRECT_TIER", "free")).strip().lower()
    return tier if tier in ("free", "paid") else "free"


def location() -> str:
    """Recorded on every log line. On the direct API there is no region, and that absence is the
    fact worth recording: the value names the route and the tier instead."""
    if uses_vertex():
        return _env("EXPENSE_AI_LOCATION", "asia-southeast1")
    return "gemini-api-direct-" + direct_tier()


# --------------------------------------------------------------------------- files


def sniff_mime(data: bytes) -> str | None:
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    return None


# --------------------------------------------------------------------------- the client

_clients: dict[int, Any] = {}
_client_lock = threading.Lock()


def get_client(timeout_s: int) -> Any:
    """One google-genai client per process and timeout. Raises when no route is configured."""
    with _client_lock:
        if timeout_s in _clients:
            return _clients[timeout_s]
        from google import genai
        from google.genai import types

        http_options = types.HttpOptions(
            timeout=timeout_s * 1000,  # milliseconds
            # The SDK retries up to four times by default - far too many for a person waiting on
            # a form. One attempt; the single retry allowed is ours, in call_model.
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        if uses_vertex():
            client = genai.Client(
                vertexai=True,
                project=_env("GOOGLE_CLOUD_PROJECT"),
                location=location(),
                http_options=http_options,
            )
            logger.info(
                "ai: Vertex AI client ready, project=%s location=%s",
                _env("GOOGLE_CLOUD_PROJECT"),
                location(),
            )
        else:
            if not flag("EXPENSE_AI_ALLOW_DIRECT_API", False, blank_is_unset=True):
                raise RuntimeError(
                    "No GOOGLE_CLOUD_PROJECT configured and the direct Gemini API is not opted "
                    "in. Set GOOGLE_CLOUD_PROJECT for the Vertex route, or "
                    "EXPENSE_AI_ALLOW_DIRECT_API=true to use the direct API."
                )
            api_key = _env("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            client = genai.Client(api_key=api_key, http_options=http_options)
            if direct_tier() == "paid":
                logger.info(
                    "ai: direct Gemini API, PAID tier. Content is not used to improve Google "
                    "products; prompts and responses are retained briefly for abuse detection. "
                    "NOTE: this route pins no region - documents may be processed outside "
                    "Singapore."
                )
            else:
                logger.warning(
                    "ai: direct Gemini API, FREE tier. Google may use submitted content to "
                    "develop its products and HUMAN REVIEWERS may read input and output. "
                    "Non-customer documents ONLY. Set EXPENSE_AI_DIRECT_TIER=paid once billing "
                    "is enabled on the key."
                )
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


def clamp_confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def band(confidence: float, config: ReaderConfig) -> str:
    high, medium = config.confidence_cutoffs()
    if confidence >= high:
        return "high"
    if confidence >= medium:
        return "medium"
    return "low"


def field(
    value: Any, confidence: object, config: ReaderConfig, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """One suggestion. A low-band or dropped field keeps its confidence for the log, but
    ``applied`` is false: a wrong suggestion costs the person more than none."""
    conf = clamp_confidence(confidence)
    level = band(conf, config)
    out: dict[str, Any] = {
        "value": value,
        "confidence": round(conf, 3),
        "band": level,
        "applied": bool(value not in (None, "")) and level != "low",
    }
    if extra:
        out.update(extra)
    return out


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
    """Gemini through ``google-genai``'s ``interactions.create``: Vertex AI or the direct API
    (``uses_vertex``), one client per process and timeout, at most one retry (``call_model``)."""

    name = "gemini"

    def call(self, model: str, parts: DocumentParts, config: ReaderConfig) -> ProviderReply:
        try:
            client = get_client(config.timeout_seconds())
        except Exception as exc:  # noqa: BLE001 - every failure is a reason, never a raise
            logger.error("%s: client unavailable: %s", config.label, exc)
            return ProviderReply(reason=REASON_NOT_CONFIGURED)
        request = {
            "model": model,
            "input": [
                {"type": "text", "text": parts.prefix_text},
                {
                    "type": "document" if parts.mime == "application/pdf" else "image",
                    "data": base64.b64encode(parts.document).decode("ascii"),
                    "mime_type": parts.mime,
                },
                {"type": "text", "text": parts.closing_text},
            ],
            "system_instruction": parts.instruction,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": parts.schema,
            },
            "generation_config": {
                "thinking_level": config.thinking_level(),
                "max_output_tokens": config.max_output_tokens(),
            },
        }
        interaction, reason = call_model(client, request, config)
        if interaction is None:
            return ProviderReply(reason=reason)
        reply = ProviderReply(
            usage=_gemini_usage(interaction), request_id=getattr(interaction, "id", None)
        )
        status = str(getattr(interaction, "status", "") or "")
        if status == "incomplete":
            # Truncation and a safety block both land here; neither is retried.
            detail = _error_detail(interaction)
            blocked = "safety" in detail.lower() or "block" in detail.lower()
            if blocked:
                logger.warning("%s: blocked by safety filters (%s)", config.label, detail)
            else:
                logger.warning(
                    "%s: reply truncated - output=%s thought=%s against max_output_tokens=%s. "
                    "Raise %s_MAX_OUTPUT_TOKENS if this recurs. (%s)",
                    config.label,
                    reply.usage.get("output_tokens"),
                    reply.usage.get("thought_tokens"),
                    config.max_output_tokens(),
                    config.prefix,
                    detail,
                )
            reply.reason = REASON_SAFETY_BLOCKED if blocked else REASON_TRUNCATED
            return reply
        if status != "completed":
            logger.warning(
                "%s: interaction status=%s (%s)", config.label, status, _error_detail(interaction)
            )
            reply.reason = REASON_PROVIDER_ERROR
            return reply
        reply.text = getattr(interaction, "output_text", None) or None
        if reply.text is None:
            reply.reason = REASON_INVALID_REPLY
        return reply


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

    ``models`` is the list to try in order (the primary, then the backup - ``ai_policy``'s); none
    given = ``config.model_id()`` alone. The next model is tried only when one ends in a
    ``FAILOVER_REASONS`` reason, so a read makes at most one call per model (each with its own
    ``timeout_seconds``).

    ``prefix_text`` (the entity's lists) goes FIRST and must be byte-stable for the same entity:
    Gemini's caching is implicit prefix matching, and unstable ordering does not lower the hit
    rate, it removes it. ``validate`` turns the parsed reply into the suggestions, accepting an
    id only if it was in the lists sent - the reply is untrusted input, and a value that fails a
    check is dropped, never repaired."""
    started = time.monotonic()
    chain: list[str] = []
    for model in models or [config.model_id()]:
        if model and model not in chain:
            chain.append(model)
    audit: dict[str, Any] = {
        "model_id": chain[0] if chain else None,
        "model_answered": None,
        "attempts": [],
        "location": location(),
        "thinking_level": config.thinking_level(),
    }
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


# The SDK's ``interactions.create`` raises a different exception family from the legacy
# ``models.generate_content`` (``google.genai._gaos...compat_errors``, all carrying
# ``.status_code``). Classifying on the ``status_code``/``code`` ATTRIBUTE, not class identity,
# keeps a 429 from being reported as a transport error and survives the private path moving.
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


def _check_tier_matches_reality(exc: BaseException, config: ReaderConfig) -> bool:
    """Shout if we believe we are on the paid tier and Google says otherwise: its free-tier quota
    errors name the metric ("..._free_tier_requests"). Deliberately does NOT switch the reader
    off - a transient message must not end the feature for everyone; that is a human decision."""
    if uses_vertex() or direct_tier() != "paid":
        return False
    if "free_tier" not in str(exc).lower():
        return False
    logger.error(
        "%s: CONFIGURATION MISMATCH - EXPENSE_AI_DIRECT_TIER=paid, but Google rejected this "
        "request against a FREE TIER quota. If billing is not active on this key, customer "
        "documents are being processed under free-tier terms, where Google may use them and "
        "human reviewers may read them. Set %s_ENABLED=false and confirm billing first.",
        config.label,
        config.prefix,
    )
    return True


def _advised_delay(exc: BaseException) -> float | None:
    """Seconds the provider asked us to wait, if it said ("Please retry in 39.6s")."""
    match = _RETRY_DELAY.search(str(exc))
    return float(match.group(1)) if match else None


def call_model(
    client: Any, request: dict[str, Any], config: ReaderConfig
) -> tuple[Any, str | None]:
    """One call and at most one retry - only for the retryable classes, most specific first:
    a timeout has spent the budget (no retry); a 429 retries once only if the provider's advised
    wait fits the deadline; 401/403/404 and other 4xx never; 5xx and transport errors once."""
    deadline = time.monotonic() + config.timeout_seconds()
    for attempt in (1, 2):
        try:
            return client.interactions.create(**request), None
        except Exception as exc:  # noqa: BLE001 - classified below by status
            status = _error_status(exc)
            if _is_timeout(exc):
                logger.warning("%s: timed out after %ss", config.label, config.timeout_seconds())
                return None, REASON_TIMEOUT
            if status == 429:
                _check_tier_matches_reality(exc, config)
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
                    config.timeout_seconds(),
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
                    request.get("model"),
                    location(),
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


def _error_detail(interaction: Any) -> str:
    try:
        entries = getattr(interaction, "errors", None) or []
        return "; ".join(str(getattr(e, "message", e)) for e in entries)[:300]
    except Exception:  # noqa: BLE001
        return ""


def _gemini_usage(interaction: Any) -> dict[str, int | None]:
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "total_input_tokens", None),
        "output_tokens": getattr(usage, "total_output_tokens", None),
        # Thinking tokens: invisible in the reply, billed at the OUTPUT rate, drawn from
        # max_output_tokens. Without them a truncation by thinking looks like a verbose answer.
        "thought_tokens": getattr(usage, "total_thought_tokens", None),
        # Zero across repeated same-entity calls: the prefix is under the cache minimum or unstable.
        "cached_tokens": getattr(usage, "total_cached_tokens", None),
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
