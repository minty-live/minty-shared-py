"""The AI readers' allowance and settings (owner, 2026-11-18): each company reads against its own
GRANTS, Petty Cash and Payment Request together, all held in the database so they change with an
UPDATE (or a command), never a release.

``ai_policy`` (one row) holds the models every read tries - ``primary_model``, then
``backup_model`` - and two allowance settings:

* ``default_grant`` - the one-time grant every company is given on its first read: ``{"tokens": n,
  "reads": n, "days": n}``, each key optional (at least ``tokens`` or ``reads``; ``days`` = it
  expires that many days after it is given). NULL = no automatic grant: only grants given by hand.
  It is given ONCE per company, ever (``uq_ai_grant_one_default``), so a used-up company stays used
  up until someone gives it another grant. The preview's is ``{"tokens": 20000}``.
* ``limit_rules`` - extra caps over every company that ALL apply: ``{"reads": n, "tokens": n,
  "period_days": n}``, each key optional; no ``period_days`` = the company's whole life, otherwise
  a rolling window (a daily throttle, say). The preview has none.

``ai_grant`` is one allowance given to one company (``tokens_total`` and/or ``reads_total``, an
optional ``expires_at``); a top-up is another grant. ``ai_read`` has one row per read a person asks
for, CHARGED TO ONE GRANT - the oldest live one with room. ``reserve`` writes it ``pending`` before
the model is called, under a per-company lock (two clicks with room for one: one passes), and
``finish`` marks it ``ok`` (a model answered with a valid reply) or ``failed``. What counts against
a grant (and a rule):

* a READ is an ``ok`` row, or a ``pending`` one younger than ``PENDING_COUNTS_FOR`` (a read in
  flight; a worker that died mid-read stops holding one after that). A failed read never counts;
* TOKENS are input + output + thinking tokens. A GRANT is charged only the tokens of reads a model
  answered - a failed read uses none of the company's allowance (owner's rule) - while the extra
  caps (``limit_rules``, our cost guard) count every billed token, failed reads' too. A grant is
  used up once its tokens reach its total; the read that crosses it is allowed to finish.

The page shows ``percent_left`` - how much of the company's live grants is left - and, used up, a
disabled button with the reason.

FAIL CLOSED: no policy row, or one that does not parse, logs an error and turns reading off; it
never falls back to a guessed limit. The models come through ``django.apps`` by name
(``minty_db.AiPolicy``, ``.AiGrant``, ``.AiRead``), so this module needs ``minty_db`` installed in
the service but does not import it. Extra ``django``.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from django.apps import apps
from django.db import connection, transaction
from django.db.models import Count, F, Min, Q, Sum
from django.utils import timezone

logger = logging.getLogger("minty-shared.ai")

MODULE_PETTY_CASH = "petty_cash"
MODULE_PAYMENT_REQUEST = "payment_request"
MODULES = (MODULE_PETTY_CASH, MODULE_PAYMENT_REQUEST)

#: A reserved read still counts while it may be in flight: two model calls of at most ~25 s each.
PENDING_COUNTS_FOR = timedelta(minutes=5)

_RULE_KEYS = frozenset({"reads", "tokens", "period_days"})
_GRANT_KEYS = frozenset({"tokens", "reads", "days"})


class PolicyError(ValueError):
    """``ai_policy`` (or a grant being given) holds something the readers cannot use."""


@dataclass(frozen=True)
class Rule:
    reads: int | None = None
    tokens: int | None = None
    period_days: int | None = None


@dataclass(frozen=True)
class GrantSpec:
    """A grant's size before it is given: tokens and/or reads, and days until it expires."""

    tokens: int | None = None
    reads: int | None = None
    days: int | None = None


@dataclass(frozen=True)
class Policy:
    primary_model: str
    backup_model: str | None
    rules: tuple[Rule, ...]
    default_grant: GrantSpec | None = None

    @property
    def models(self) -> list[str]:
        """The models a read tries, in order."""
        return [self.primary_model] + ([self.backup_model] if self.backup_model else [])


@dataclass(frozen=True)
class RuleUsage:
    """What a company has used inside one rule's window."""

    reads: int = 0
    tokens: int = 0
    oldest: datetime | None = None  # the oldest row in the window: when the window first moves


@dataclass(frozen=True)
class GrantUsage:
    """One grant and what has been charged to it. ``id`` None = the default grant not given yet."""

    id: Any
    tokens_total: int | None
    reads_total: int | None
    expires_at: datetime | None = None
    tokens_used: int = 0
    reads_used: int = 0

    def live(self, now: datetime) -> bool:
        return self.expires_at is None or self.expires_at > now

    def has_room(self, now: datetime) -> bool:
        return (
            self.live(now)
            and (self.tokens_total is None or self.tokens_used < self.tokens_total)
            and (self.reads_total is None or self.reads_used < self.reads_total)
        )


@dataclass(frozen=True)
class Allowance:
    """Where a company stands. ``percent_left`` is how much of its open grants (live, with room) is
    left (by tokens when they have token caps, else by reads); ``reads_limit``/``reads_left`` come
    from read caps (the tightest rule, else the open grants), None when nothing counts reads. ``blocked`` = no
    live grant has room, or a rule is used up; ``resumes_at`` is when the rules blocking it next
    have room (None = not blocked, or blocked until a new grant is given)."""

    reads_used: int | None
    reads_limit: int | None
    reads_left: int | None
    blocked: bool
    resumes_at: datetime | None
    percent_left: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reads_used": self.reads_used,
            "reads_limit": self.reads_limit,
            "reads_left": self.reads_left,
            "percent_left": self.percent_left,
            "blocked": self.blocked,
            "resumes_at": self.resumes_at.isoformat() if self.resumes_at else None,
        }


# --------------------------------------------------------------------------- the policy


def _whole(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PolicyError(f"{name} must be a whole number of at least {minimum}, not {value!r}")
    return value


def parse_rules(raw: Any) -> tuple[Rule, ...]:
    """``limit_rules`` as rules, or ``PolicyError``. An empty list is a deliberate "no extra cap"."""
    if not isinstance(raw, list):
        raise PolicyError(f"limit_rules must be a list of rules, not {type(raw).__name__}")
    rules = []
    for index, item in enumerate(raw):
        where = f"limit_rules[{index}]"
        if not isinstance(item, dict):
            raise PolicyError(f'{where} must be an object like {{"reads": 10}}')
        unknown = set(item) - _RULE_KEYS
        if unknown:
            raise PolicyError(f"{where} has unknown keys {sorted(unknown)}")
        if item.get("reads") is None and item.get("tokens") is None:
            raise PolicyError(f"{where} limits nothing: give it reads, tokens or both")
        rules.append(
            Rule(
                reads=None if item.get("reads") is None else _whole(item["reads"], "reads", 0),
                tokens=None if item.get("tokens") is None else _whole(item["tokens"], "tokens", 0),
                period_days=None
                if item.get("period_days") is None
                else _whole(item["period_days"], "period_days", 1),
            )
        )
    return tuple(rules)


def parse_grant(raw: Any, where: str = "default_grant") -> GrantSpec | None:
    """A grant's size (``ai_policy.default_grant``, or one being given), or ``PolicyError``."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise PolicyError(f'{where} must be an object like {{"tokens": 10000}}')
    unknown = set(raw) - _GRANT_KEYS
    if unknown:
        raise PolicyError(f"{where} has unknown keys {sorted(unknown)}")
    if raw.get("tokens") is None and raw.get("reads") is None:
        raise PolicyError(f"{where} gives nothing: give it tokens, reads or both")
    return GrantSpec(
        tokens=None if raw.get("tokens") is None else _whole(raw["tokens"], "tokens", 1),
        reads=None if raw.get("reads") is None else _whole(raw["reads"], "reads", 1),
        days=None if raw.get("days") is None else _whole(raw["days"], "days", 1),
    )


def parse_policy(
    primary_model: Any, backup_model: Any, limit_rules: Any, default_grant: Any = None
) -> Policy:
    if not isinstance(primary_model, str) or not primary_model.strip():
        raise PolicyError("primary_model is empty")
    backup = (
        backup_model.strip() if isinstance(backup_model, str) and backup_model.strip() else None
    )
    return Policy(
        primary_model.strip(), backup, parse_rules(limit_rules), parse_grant(default_grant)
    )


def load_policy() -> Policy | None:
    """The policy row, read on every call (an UPDATE takes effect on the next read). None - and an
    error in the log - when there is no usable row: reading is then off."""
    row = (
        apps.get_model("minty_db", "AiPolicy")
        .objects.filter(id=1)
        .values("primary_model", "backup_model", "limit_rules", "default_grant")
        .first()
    )
    if row is None:
        logger.error(
            "ai: ai_policy has no row - AI reading is OFF until it has one "
            "(seed_catalogue.sql, or minty-db's 2026-11-18_ai_policy_and_reads.sql)"
        )
        return None
    try:
        return parse_policy(
            row["primary_model"], row["backup_model"], row["limit_rules"], row["default_grant"]
        )
    except PolicyError as exc:
        logger.error("ai: ai_policy is unusable (%s) - AI reading is OFF until it is fixed", exc)
        return None


# --------------------------------------------------------------------------- where a company stands


def _percent(remaining: int, total: int) -> int:
    """Rounded UP, so a grant with anything left never reads 0%."""
    if total <= 0 or remaining <= 0:
        return 0
    return min(100, math.ceil(100 * remaining / total))


def summarise(
    rules: tuple[Rule, ...],
    usages: list[RuleUsage],
    grants: list[GrantUsage],
    now: datetime,
) -> Allowance:
    """The allowance from each rule's usage (``usages[i]`` is ``rules[i]``'s) and the company's
    grants (the default one not given yet included, as a grant with id None). Pure."""
    blocking: list[tuple[Rule, RuleUsage]] = []
    tightest: tuple[int, int, int] | None = None  # (left, limit, used)
    for rule, used in zip(rules, usages, strict=True):
        if rule.reads is not None:
            left = max(0, rule.reads - used.reads)
            if tightest is None or left < tightest[0]:
                tightest = (left, rule.reads, used.reads)
        out_of_reads = rule.reads is not None and used.reads >= rule.reads
        out_of_tokens = rule.tokens is not None and used.tokens >= rule.tokens
        if out_of_reads or out_of_tokens:
            blocking.append((rule, used))

    # what is left is measured over the grants that still have room: a used-up grant is spent, so a
    # top-up after it reads 100%, not a share of everything ever given
    open_grants = [g for g in grants if g.has_room(now)]
    no_grant_room = not open_grants
    by_tokens = [g for g in open_grants if g.tokens_total is not None]
    by_reads = [g for g in open_grants if g.reads_total is not None]
    percent = 0
    if by_tokens:
        total = sum(g.tokens_total or 0 for g in by_tokens)
        remaining = sum(max(0, (g.tokens_total or 0) - g.tokens_used) for g in by_tokens)
        percent = _percent(remaining, total)
    elif by_reads:
        total = sum(g.reads_total or 0 for g in by_reads)
        remaining = sum(max(0, (g.reads_total or 0) - g.reads_used) for g in by_reads)
        percent = _percent(remaining, total)
    if no_grant_room:
        percent = 0
    if tightest is None and by_reads:
        limit = sum(g.reads_total or 0 for g in by_reads)
        used = sum(g.reads_used for g in by_reads)
        tightest = (max(0, limit - used), limit, used)

    resumes_at = None
    if blocking and not no_grant_room and all(r.period_days for r, _ in blocking):
        moments = [
            u.oldest + timedelta(days=r.period_days)  # type: ignore[arg-type]
            for r, u in blocking
            if u.oldest is not None
        ]
        resumes_at = max(moments) if moments else None
    return Allowance(
        reads_used=tightest[2] if tightest else None,
        reads_limit=tightest[1] if tightest else None,
        reads_left=tightest[0] if tightest else None,
        blocked=bool(blocking) or no_grant_room,
        resumes_at=resumes_at,
        percent_left=percent,
    )


def _counted(now: datetime) -> Q:
    counted: Q = Q(status="ok") | Q(status="pending", created_at__gt=now - PENDING_COUNTS_FOR)  # pyright: ignore[reportAssignmentType] - Django has no stubs
    return counted


def _usage(entity_id: Any, period_days: int | None, now: datetime) -> RuleUsage:
    rows = apps.get_model("minty_db", "AiRead").objects.filter(entity_id=entity_id)
    if period_days is not None:
        rows = rows.filter(created_at__gte=now - timedelta(days=period_days))
    totals = rows.aggregate(
        reads=Count("id", filter=_counted(now)),
        tokens=Sum(F("tokens_in") + F("tokens_out") + F("tokens_thought")),
        oldest=Min("created_at"),
    )
    return RuleUsage(
        reads=int(totals["reads"] or 0), tokens=int(totals["tokens"] or 0), oldest=totals["oldest"]
    )


def grants_of(entity_id: Any, now: datetime | None = None) -> list[GrantUsage]:
    """The company's grants, oldest first, with what has been charged to each."""
    return grants_of_many([entity_id], now).get(str(entity_id), [])


def grants_of_many(
    entity_ids: list[Any], now: datetime | None = None
) -> dict[str, list[GrantUsage]]:
    """Several companies' grants at once - two queries, whatever the count (a superuser's list).
    Keyed by the company id as text; a company with no grant is absent."""
    now = now or timezone.now()
    rows = list(
        apps.get_model("minty_db", "AiGrant")
        .objects.filter(entity_id__in=list(entity_ids))
        .order_by("created_at", "id")
        .values("id", "entity_id", "tokens_total", "reads_total", "expires_at")
    )
    if not rows:
        return {}
    charged = {
        row["grant_id"]: row
        for row in apps.get_model("minty_db", "AiRead")
        .objects.filter(grant_id__in=[r["id"] for r in rows])
        .values("grant_id")
        .annotate(
            reads=Count("id", filter=_counted(now)),
            # only reads a model answered are charged to the company's grant
            tokens=Sum(
                F("tokens_in") + F("tokens_out") + F("tokens_thought"), filter=Q(status="ok")
            ),
        )
    }
    out: dict[str, list[GrantUsage]] = {}
    for r in rows:
        used = charged.get(r["id"]) or {}
        out.setdefault(str(r["entity_id"]), []).append(
            GrantUsage(
                id=r["id"],
                tokens_total=r["tokens_total"],
                reads_total=r["reads_total"],
                expires_at=r["expires_at"],
                tokens_used=int(used.get("tokens") or 0),
                reads_used=int(used.get("reads") or 0),
            )
        )
    return out


def grant_allowance(
    grants: list[GrantUsage], policy: Policy, now: datetime | None = None
) -> Allowance:
    """Where a company stands on its grants alone (no extra caps) - the superuser list's figure,
    from ``grants_of_many``. A company never given anything shows the default it will get."""
    now = now or timezone.now()
    return summarise((), [], with_default(grants, policy, now), now)


def with_default(grants: list[GrantUsage], policy: Policy, now: datetime) -> list[GrantUsage]:
    """A company never given anything counts the default grant it WILL get on its first read."""
    spec = policy.default_grant
    if grants or spec is None:
        return grants
    expires = now + timedelta(days=spec.days) if spec.days else None
    return [
        GrantUsage(id=None, tokens_total=spec.tokens, reads_total=spec.reads, expires_at=expires)
    ]


def status(entity_id: Any, policy: Policy) -> Allowance:
    """Where ``entity_id`` stands now under ``policy``: its grants, and one query per distinct
    rule window."""
    now = timezone.now()
    by_period: dict[int | None, RuleUsage] = {}
    for rule in policy.rules:
        if rule.period_days not in by_period:
            by_period[rule.period_days] = _usage(entity_id, rule.period_days, now)
    grants = with_default(grants_of(entity_id, now), policy, now)
    return summarise(policy.rules, [by_period[r.period_days] for r in policy.rules], grants, now)


def reader_status(enabled: bool, entity_id: Any) -> dict[str, Any]:
    """What a page needs to draw the read button: ``enabled`` (the reader's switch AND a usable
    policy) and the allowance. A switched-off reader sends no numbers."""
    if not enabled:
        return {"enabled": False}
    policy = load_policy()
    if policy is None:
        return {"enabled": False}
    return {"enabled": True, **status(entity_id, policy).as_dict()}


# --------------------------------------------------------------------------- grants and reads


def give_grant(
    entity_id: Any,
    spec: GrantSpec,
    *,
    granted_by: str | None = None,
    note: str | None = None,
    source: str = "manual",
) -> Any:
    """Give ``entity_id`` an allowance (a top-up, or a size other than the default): the new
    ``ai_grant`` id. Its tokens/reads add to whatever the company still has."""
    expires = timezone.now() + timedelta(days=spec.days) if spec.days else None
    row = apps.get_model("minty_db", "AiGrant").objects.create(
        entity_id=entity_id,
        tokens_total=spec.tokens,
        reads_total=spec.reads,
        expires_at=expires,
        source=source,
        granted_by=granted_by,
        note=note,
    )
    return row.id


def _lock(entity_id: Any) -> None:
    if connection.vendor == "postgresql":
        with connection.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", [f"ai_read:{entity_id}"]
            )


def reserve(
    entity_id: Any, user_id: Any, module: str, policy: Policy
) -> tuple[uuid.UUID | None, Allowance]:
    """Hold one read for ``entity_id`` before the model is called: ``(read id, allowance after
    it)``, or ``(None, allowance)`` when no grant has room or a rule is used up. A company's first
    read gives it the default grant. Serialised per company (a Postgres advisory lock, held to the
    end of the transaction), across both services."""
    if module not in MODULES:
        raise ValueError(f"unknown module {module!r}")
    read_model = apps.get_model("minty_db", "AiRead")
    with transaction.atomic():  # pyright: ignore[reportGeneralTypeIssues] - Django has no stubs
        _lock(entity_id)
        now = timezone.now()
        if policy.default_grant and not grants_of(entity_id, now):
            give_grant(entity_id, policy.default_grant, granted_by="ai_policy", source="default")
        before = status(entity_id, policy)
        if before.blocked:
            return None, before
        charge = next(g for g in grants_of(entity_id, now) if g.has_room(now))
        row = read_model.objects.create(
            entity_id=entity_id,
            grant_id=charge.id,
            user_id=user_id,
            module=module,
            status="pending",
        )
        return row.id, status(entity_id, policy)


def finish(read_id: uuid.UUID, extraction: Any) -> None:
    """Record how a reserved read ended, from ``minty_shared.ai.Extraction``: ``ok`` when a model
    answered (it then counts), else ``failed`` (it never does); tokens of every attempt."""
    audit = extraction.audit
    cost = audit.get("estimated_cost")
    apps.get_model("minty_db", "AiRead").objects.filter(id=read_id).update(
        status="ok" if extraction.answered else "failed",
        reason=(extraction.reason or None) and str(extraction.reason)[:40],
        model_answered=audit.get("model_answered"),
        attempts=extraction.attempts_text()[:400] or None,
        tokens_in=int(audit.get("input_tokens") or 0),
        tokens_out=int(audit.get("output_tokens") or 0),
        tokens_thought=int(audit.get("thought_tokens") or 0),
        tokens_cached=int(audit.get("cached_tokens") or 0),
        estimated_cost=None if cost is None else Decimal(str(cost)),
        latency_ms=audit.get("latency_ms"),
        finished_at=timezone.now(),
    )
