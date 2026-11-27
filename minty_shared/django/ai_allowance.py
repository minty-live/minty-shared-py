"""The AI readers' allowance and settings (owner, 2026-11-18): each company reads against its own
GRANTS, Petty Cash and Payment Request together, all held in the database so they change with an
UPDATE (or a command), never a release.

``ai_policy`` (one row) holds the models every read tries - ``primary_model``, then
``backup_model`` - the readers' settings (owner, 2026-11-20; once each service's environment):
``receipt_reader_on`` and ``invoice_reader_on`` (Petty Cash's and Payment Request's switches),
``timeout_s``, ``max_output_tokens``, ``thinking_level`` and ``max_file_mb`` (no region: every read
is on Vertex's global endpoint, owner 2026-11-23), which ``Policy.reader`` turns into the ``minty_shared.ai.ReaderConfig`` a read uses - each model's price
(``model_prices``, the user 2026-11-24: ``{"<model>": {"input": n, "output": n, "cached_input": n}}``,
USD per 1M tokens; every model the policy names must have one, else reading is OFF) - and the
allowance settings. AI is its own product on the subscription (the user, 2026-11-27):

* ``default_grant`` - the company's AI TRIAL (``source='trial'``, once per company, ever:
  ``uq_ai_grant_one_trial``): ``{"tokens": n, "reads": n, "days": n}``, each key optional (at least
  ``tokens`` or ``reads``; ``days`` = how long the trial runs). It starts with the company's first
  module subscription (its trial, or a purchase without one) - every company that existed on
  2026-11-27 was given one from that day instead (minty-db's ``2026-11-27_ai_trial_reset.sql``) -
  and is written at the first read inside that window. NULL = no trial. Live: 280,000 tokens,
  30 days.
* ``cycle_tokens`` - after the trial, one grant per billing period of the company's subscription
  (``source='cycle'``): the period from the payer's anchor (``minty_shared.periods``, billing's own
  arithmetic), expiring at the period's end, no rollover. The first one is prorated like billing
  (elapsed seconds, half away from zero) from the trial's end or the company's first charge when
  either falls inside the period. Only while some module is paid for and its access stands
  (subscription-api's ``access_end``: ``app_access_until``, else ``paid_through``, plus the past-due
  window when past due). NULL = no period grants.
* ``limit_rules`` - extra caps over every company that ALL apply: ``{"reads": n, "tokens": n,
  "period_days": n}``, each key optional; no ``period_days`` = the company's whole life, otherwise
  a rolling window (a daily throttle, say). Live: none.

``ai_grant`` is one allowance given to one company (``tokens_total`` and/or ``reads_total``, an
optional ``expires_at``): the trial, a period's grant, a top-up given by hand (``manual``), or the
one-time ``default`` grant every company got before 2026-11-27. ``ai_read`` has one row per read a
person asks for, CHARGED TO ONE GRANT - the live one with room that expires soonest (no expiry
last, then the oldest), so a top-up carries over. The trial and period grants are given at the
read, under the lock, never by a scheduler; until then ``status`` counts them as due, so a page
is right before the first read. ``reserve`` writes the read ``pending`` before
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
(``minty_db.AiPolicy``, ``.AiGrant``, ``.AiRead``, and the subscription's
``.EntityModuleSubscription``, ``.UserStripeCustomer``, ``.EntityBillingGroup``,
``.SubscriptionTransfer``, ``.BillingPolicy`` - read only, the same rows subscription-api decides
access from), so this module needs ``minty_db`` installed in the service but does not import it.
Extra ``django``.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from django.apps import apps
from django.db import connection, transaction
from django.db.models import Count, F, Min, Q, Sum
from django.utils import timezone

from minty_shared import ai
from minty_shared.periods import period_containing, prorate

logger = logging.getLogger("minty-shared.ai")

#: ``ai_grant.source``: the AI trial, a billing period's grant, a top-up by hand, and the one-time
#: grant before 2026-11-27.
SOURCE_TRIAL = "trial"
SOURCE_CYCLE = "cycle"
SOURCE_MANUAL = "manual"
SOURCE_DEFAULT = "default"

#: Who the automatic grants are given by (``ai_grant.granted_by``).
GRANTED_BY_POLICY = "ai_policy"

MODULE_PETTY_CASH = "petty_cash"
MODULE_PAYMENT_REQUEST = "payment_request"
MODULES = (MODULE_PETTY_CASH, MODULE_PAYMENT_REQUEST)

#: The two readers, by their ``ai_policy`` switch: Petty Cash's receipts, Payment Request's invoices.
READER_RECEIPT = "receipt"
READER_INVOICE = "invoice"
READERS = (READER_RECEIPT, READER_INVOICE)

#: A reserved read still counts while it may be in flight: two model calls of at most ~25 s each.
PENDING_COUNTS_FOR = timedelta(minutes=5)

_RULE_KEYS = frozenset({"reads", "tokens", "period_days"})
_GRANT_KEYS = frozenset({"tokens", "reads", "days"})
_PRICE_KEYS = frozenset({"input", "output", "cached_input"})


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
class ReaderSettings:
    """``ai_policy``'s reader columns. The defaults are the columns' own."""

    receipt_reader_on: bool = False
    invoice_reader_on: bool = False
    timeout_s: int = 25
    max_output_tokens: int = 4096
    thinking_level: str = "low"
    max_file_mb: int = 10


#: The reader columns, in order - what ``load_policy`` reads and ``parse_settings`` accepts.
SETTING_FIELDS = tuple(ReaderSettings.__dataclass_fields__)

#: Each number's allowed range - the columns' CHECKs, so a bad value is named before the database
#: refuses it.
SETTING_RANGES = {"timeout_s": (5, 120), "max_output_tokens": (256, 65536), "max_file_mb": (1, 20)}


@dataclass(frozen=True)
class Policy:
    primary_model: str
    backup_model: str | None
    rules: tuple[Rule, ...]
    default_grant: GrantSpec | None = None
    settings: ReaderSettings = ReaderSettings()
    prices: tuple[tuple[str, ai.ModelPrice], ...] = ()
    cycle_tokens: int | None = None

    @property
    def models(self) -> list[str]:
        """The models a read tries, in order."""
        return [self.primary_model] + ([self.backup_model] if self.backup_model else [])

    def reader(self, name: str) -> ai.ReaderConfig:
        """``name``'s (``READER_RECEIPT`` / ``READER_INVOICE``) settings for ``minty_shared.ai``."""
        if name not in READERS:
            raise ValueError(f"unknown reader {name!r}")
        s = self.settings
        return ai.ReaderConfig(
            label=f"{name}_ai",
            enabled=s.receipt_reader_on if name == READER_RECEIPT else s.invoice_reader_on,
            timeout_seconds=s.timeout_s,
            max_output_tokens=s.max_output_tokens,
            thinking_level=s.thinking_level,
            max_file_mb=s.max_file_mb,
            prices=dict(self.prices),
        )


@dataclass(frozen=True)
class RuleUsage:
    """What a company has used inside one rule's window."""

    reads: int = 0
    tokens: int = 0
    oldest: datetime | None = None  # the oldest row in the window: when the window first moves


@dataclass(frozen=True)
class GrantUsage:
    """One grant and what has been charged to it. ``id`` None = a grant due but not given yet (the
    trial or this period's, given at the next read)."""

    id: Any
    tokens_total: int | None
    reads_total: int | None
    expires_at: datetime | None = None
    tokens_used: int = 0
    reads_used: int = 0
    source: str = SOURCE_MANUAL
    created_at: datetime | None = None

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
    live grant has room, or a rule is used up; ``resumes_at`` is when reading comes back by itself
    - the rules' windows moving, the next billing period's grant, or the trial handing over to the
    period grants - (None = not blocked, or blocked until someone gives a grant)."""

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


# --------------------------------------------------------------------------- the subscription

#: ``entity_module_subscription.phase`` values the allowance reads (subscription-api's).
PHASE_ACTIVE = "active"
PHASE_PAST_DUE = "past_due"
PHASE_SCHEDULED_CANCEL = "scheduled_cancel"


@dataclass(frozen=True)
class ModuleTerm:
    """One module's subscription row (``entity_module_subscription``), the parts that matter here."""

    phase: str
    created_at: datetime
    trial_end: datetime | None = None
    app_access_until: datetime | None = None
    first_billed_at: datetime | None = None

    @property
    def paying(self) -> bool:
        """Billed - not a trial, not finished: subscription-api's ``is_paid_module``."""
        return self.phase in (PHASE_ACTIVE, PHASE_PAST_DUE) or (
            self.phase == PHASE_SCHEDULED_CANCEL and self.first_billed_at is not None
        )


@dataclass(frozen=True)
class Subscription:
    """What a company's subscription says about its periods: its module rows, the payer's anchor
    (every period is derived from it), and what the company is paid through (its billing
    account's ``paid_through``, or a handover's parked ``collect_at`` when later)."""

    modules: tuple[ModuleTerm, ...] = ()
    anchor_at: datetime | None = None
    paid_through: datetime | None = None
    past_due_days: int = 0

    def access_until(self, module: ModuleTerm) -> datetime | None:
        """When a paid module's access ends: subscription-api's ``access.access_end`` for the
        paying phases."""
        if module.app_access_until is not None:
            return module.app_access_until
        if self.paid_through is None:
            return None
        if module.phase == PHASE_PAST_DUE:
            return self.paid_through + timedelta(days=self.past_due_days)
        return self.paid_through

    def paid_access(self) -> tuple[datetime | None, bool]:
        """``(when the last paid module's access ends, whether one renews by itself)`` - renews =
        active with no end of its own, so its access moves on at each renewal."""
        paying = [m for m in self.modules if m.paying]
        ends = [end for end in (self.access_until(m) for m in paying) if end is not None]
        renews = any(m.phase == PHASE_ACTIVE and m.app_access_until is None for m in paying)
        return (max(ends) if ends else None), renews

    @property
    def first_billed_at(self) -> datetime | None:
        """The company's first charge, over the modules it pays for."""
        billed = [m.first_billed_at for m in self.modules if m.paying and m.first_billed_at]
        return min(billed) if billed else None


@dataclass(frozen=True)
class DueGrant:
    """A grant the company is due now but has not been given: the trial, or this period's."""

    source: str
    tokens: int | None
    reads: int | None
    expires_at: datetime | None

    def usage(self) -> GrantUsage:
        return GrantUsage(
            id=None,
            tokens_total=self.tokens,
            reads_total=self.reads,
            expires_at=self.expires_at,
            source=self.source,
        )


@dataclass(frozen=True)
class GrantPlan:
    """The grants due now, and when the next one comes by itself (None = none will)."""

    due: tuple[DueGrant, ...] = ()
    next_grant_at: datetime | None = None


def plan_grants(
    grants: list[GrantUsage], subscription: Subscription, policy: Policy, now: datetime
) -> GrantPlan:
    """What ``grants`` (the company's, as given) lack now under ``policy``, and when the next
    automatic grant comes. Pure.

    * The TRIAL: none given yet, ``default_grant`` set and the company has a module row - it runs
      ``days`` from the first row's ``created_at``; due only inside that window.
    * A PERIOD'S GRANT: after the trial (or with no trial), ``cycle_tokens`` set, an anchor, and a
      paid module whose access is still running. One per period - one created inside it counts,
      ended early or not - prorated from the latest of the period's start, the trial's end and
      the company's first charge.
    """
    trial = next((g for g in grants if g.source == SOURCE_TRIAL), None)
    due: list[DueGrant] = []
    has_trial = trial is not None
    trial_end = trial.expires_at if trial is not None else None
    spec = policy.default_grant
    if trial is None and spec is not None and subscription.modules:
        start = min(m.created_at for m in subscription.modules)
        end = start + timedelta(days=spec.days) if spec.days else None
        if end is None or now < end:
            due.append(DueGrant(SOURCE_TRIAL, spec.tokens, spec.reads, end))
        has_trial, trial_end = True, end

    next_at = None
    tokens = policy.cycle_tokens
    until, renews = subscription.paid_access()
    if tokens and subscription.anchor_at is not None and until is not None and until > now:
        if has_trial and (trial_end is None or now < trial_end):
            # the AI trial runs its full length; the period grants start when it ends
            if trial_end is not None and (renews or until > trial_end):
                next_at = trial_end
        else:
            period = period_containing(subscription.anchor_at, now)
            starts = [period.start]
            if trial_end is not None:
                starts.append(trial_end)
            if subscription.first_billed_at is not None:
                starts.append(subscription.first_billed_at)
            size = prorate(tokens, period, max(starts))
            given = any(
                g.source == SOURCE_CYCLE
                and g.created_at is not None
                and period.contains(g.created_at)
                for g in grants
            )
            if size > 0 and not given:
                due.append(DueGrant(SOURCE_CYCLE, size, None, period.end))
            if renews or until > period.end:
                next_at = period.end
    return GrantPlan(tuple(due), next_at)


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


def parse_settings(raw: dict[str, Any] | None) -> ReaderSettings:
    """``ai_policy``'s reader columns as ``ReaderSettings``, or ``PolicyError``. A key not given
    keeps the column's default."""
    raw = dict(raw or {})
    unknown = set(raw) - set(SETTING_FIELDS)
    if unknown:
        raise PolicyError(f"unknown reader settings {sorted(unknown)}")
    values: dict[str, Any] = {}
    for key in ("receipt_reader_on", "invoice_reader_on"):
        if key in raw:
            if not isinstance(raw[key], bool):
                raise PolicyError(f"{key} must be true or false, not {raw[key]!r}")
            values[key] = raw[key]
    for key, (low, high) in SETTING_RANGES.items():
        if key in raw:
            value = _whole(raw[key], key, low)
            if value > high:
                raise PolicyError(f"{key} must be at most {high}, not {value}")
            values[key] = value
    if "thinking_level" in raw:
        level = raw["thinking_level"]
        if level not in ai.THINKING_LEVELS:
            raise PolicyError(
                f"thinking_level must be one of {', '.join(ai.THINKING_LEVELS)}, not {level!r}"
            )
        values["thinking_level"] = level
    return ReaderSettings(**values)


def _price(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal) or value < 0:
        raise PolicyError(
            f"{where} must be a price of at least 0 (USD per 1M tokens), not {value!r}"
        )
    return float(value)


def parse_prices(raw: Any, models: list[str]) -> dict[str, ai.ModelPrice]:
    """``model_prices`` as each model's ``ai.ModelPrice``, or ``PolicyError`` - also when one of
    ``models`` (the policy's) has no price, so a read's cost is never left out silently."""
    if not isinstance(raw, dict):
        raise PolicyError(
            'model_prices must be an object like {"<model>": {"input": 0.3, "output": 2.5, '
            '"cached_input": 0.03}}'
        )
    prices = {}
    for model, entry in raw.items():
        where = f"model_prices[{model!r}]"
        if not isinstance(entry, dict) or set(entry) != _PRICE_KEYS:
            raise PolicyError(f"{where} must have exactly {sorted(_PRICE_KEYS)}")
        prices[model] = ai.ModelPrice(
            input=_price(entry["input"], f"{where}.input"),
            output=_price(entry["output"], f"{where}.output"),
            cached_input=_price(entry["cached_input"], f"{where}.cached_input"),
        )
    missing = [model for model in models if model not in prices]
    if missing:
        raise PolicyError(f"model_prices has no price for {', '.join(missing)}")
    return prices


def parse_policy(
    primary_model: Any,
    backup_model: Any,
    limit_rules: Any,
    default_grant: Any = None,
    settings: dict[str, Any] | None = None,
    model_prices: Any = None,
    cycle_tokens: Any = None,
) -> Policy:
    if not isinstance(primary_model, str) or not primary_model.strip():
        raise PolicyError("primary_model is empty")
    backup = (
        backup_model.strip() if isinstance(backup_model, str) and backup_model.strip() else None
    )
    models = [primary_model.strip()] + ([backup] if backup else [])
    return Policy(
        primary_model.strip(),
        backup,
        parse_rules(limit_rules),
        parse_grant(default_grant),
        parse_settings(settings),
        tuple(parse_prices(model_prices, models).items()),
        None if cycle_tokens is None else _whole(cycle_tokens, "cycle_tokens", 1),
    )


def load_policy() -> Policy | None:
    """The policy row, read on every call (an UPDATE takes effect on the next read). None - and an
    error in the log - when there is no usable row: reading is then off."""
    row = (
        apps.get_model("minty_db", "AiPolicy")
        .objects.filter(id=1)
        .values(
            "primary_model",
            "backup_model",
            "limit_rules",
            "default_grant",
            "model_prices",
            "cycle_tokens",
            *SETTING_FIELDS,
        )
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
            row["primary_model"],
            row["backup_model"],
            row["limit_rules"],
            row["default_grant"],
            {key: row[key] for key in SETTING_FIELDS},
            row["model_prices"],
            row["cycle_tokens"],
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
    next_grant_at: datetime | None = None,
) -> Allowance:
    """The allowance from each rule's usage (``usages[i]`` is ``rules[i]``'s) and the company's
    grants (the ones due but not given yet included, with id None). ``next_grant_at`` is when the
    next automatic grant comes (``GrantPlan``): a company with no grant room reads again then.
    Pure."""
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

    # reading comes back by itself only when everything stopping it does: each blocking rule's
    # window moving, and - with no grant room - the next automatic grant
    waits: list[datetime | None] = []
    if blocking:
        if all(r.period_days for r, _ in blocking):
            moments = [
                u.oldest + timedelta(days=r.period_days)  # type: ignore[arg-type]
                for r, u in blocking
                if u.oldest is not None
            ]
            waits.append(max(moments) if moments else None)
        else:
            waits.append(None)
    if no_grant_room:
        waits.append(next_grant_at)
    resumes_at = None
    if waits and all(w is not None for w in waits):
        resumes_at = max(w for w in waits if w is not None)
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
    """The company's grants, oldest first, with what has been charged to each (``charge_order``
    picks the one a read uses)."""
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
        .values(
            "id",
            "entity_id",
            "tokens_total",
            "reads_total",
            "expires_at",
            "source",
            "created_at",
        )
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
                source=r["source"],
                created_at=r["created_at"],
            )
        )
    return out


def subscription_of(entity_id: Any) -> Subscription:
    """``entity_id``'s ``Subscription`` (an empty one when it has no module row)."""
    return subscriptions_of_many([entity_id]).get(str(entity_id), Subscription())


def subscriptions_of_many(entity_ids: list[Any]) -> dict[str, Subscription]:
    """Several companies' subscriptions at once - at most five queries, whatever the count. Keyed by the
    company id as text; a company with no module row is absent. Read the way subscription-api
    reads them: one payer per company (its rows' ``payer_user_id``), the anchor on the payer's
    ``user_stripe_customer``, ``paid_through`` on the company's billing account for that payer -
    or a handover's parked ``collect_at`` when later (``store.paid_through_for_entity``)."""
    ids = [str(e) for e in entity_ids]
    rows = list(
        apps.get_model("minty_db", "EntityModuleSubscription")
        .objects.filter(entity_id__in=ids)
        .values(
            "entity_id",
            "phase",
            "created_at",
            "trial_end",
            "app_access_until",
            "first_billed_at",
            "payer_user_id",
        )
    )
    if not rows:
        return {}
    modules: dict[str, list[ModuleTerm]] = {}
    payer: dict[str, Any] = {}
    for r in rows:
        key = str(r["entity_id"])
        modules.setdefault(key, []).append(
            ModuleTerm(
                phase=str(r["phase"]),
                created_at=r["created_at"],
                trial_end=r["trial_end"],
                app_access_until=r["app_access_until"],
                first_billed_at=r["first_billed_at"],
            )
        )
        if r["payer_user_id"] is not None:
            payer[key] = r["payer_user_id"]
    payers = set(payer.values())
    anchors: dict[str, datetime | None] = {}
    paid: dict[str, datetime | None] = {}
    if payers:
        anchors = {
            str(user): anchor
            for user, anchor in apps.get_model("minty_db", "UserStripeCustomer")
            .objects.filter(user_id__in=payers)
            .values_list("user_id", "anchor_at")
        }
        for entity, user, through in (
            apps.get_model("minty_db", "EntityBillingGroup")
            .objects.filter(entity_id__in=list(payer), payer_user_id__in=payers)
            .values_list("entity_id", "payer_user_id", "billing_group__paid_through")
        ):
            if payer.get(str(entity)) == user:
                paid[str(entity)] = through
        for entity, user, collect in (
            apps.get_model("minty_db", "SubscriptionTransfer")
            .objects.filter(
                entity_id__in=list(payer), to_user_id__in=payers, collect_at__isnull=False
            )
            .values_list("entity_id", "to_user_id", "collect_at")
        ):
            key = str(entity)
            if payer.get(key) == user and (paid.get(key) is None or collect > paid[key]):
                paid[key] = collect
    window = 0
    if any(t.phase == PHASE_PAST_DUE for terms in modules.values() for t in terms):
        # only a past-due module's access needs the window (subscription-api's past_due_window_days)
        found = (
            apps.get_model("minty_db", "BillingPolicy")
            .objects.filter(id=1)
            .values_list("past_due_window_days", flat=True)
            .first()
        )
        if found is None:
            logger.error(
                "ai: billing_policy has no row - a past-due company gets no AI grace "
                "(seed_catalogue.sql)"
            )
        window = int(found or 0)
    return {
        key: Subscription(
            modules=tuple(terms),
            anchor_at=anchors.get(str(payer[key])) if key in payer else None,
            paid_through=paid.get(key),
            past_due_days=int(window),
        )
        for key, terms in modules.items()
    }


def charge_order(grant: GrantUsage) -> tuple[bool, datetime]:
    """The order a read picks among grants with room: the soonest to expire first, no expiry last
    (``min`` keeps the oldest among equals: ``grants_of`` lists oldest first)."""
    return grant.expires_at is None, grant.expires_at or datetime.max.replace(tzinfo=UTC)


def with_due(
    grants: list[GrantUsage], policy: Policy, subscription: Subscription, now: datetime
) -> tuple[list[GrantUsage], GrantPlan]:
    """``grants`` and the grants due now but not given yet (counted as if given: the next read
    gives them), with the plan they came from."""
    plan = plan_grants(grants, subscription, policy, now)
    return grants + [d.usage() for d in plan.due], plan


def grant_allowance(
    grants: list[GrantUsage],
    policy: Policy,
    subscription: Subscription,
    now: datetime | None = None,
) -> Allowance:
    """Where a company stands on its grants alone (no extra caps) - the superuser list's figure,
    from ``grants_of_many`` and ``subscriptions_of_many``. A grant due but not given yet counts."""
    now = now or timezone.now()
    counted, plan = with_due(grants, policy, subscription, now)
    return summarise((), [], counted, now, plan.next_grant_at)


def status(entity_id: Any, policy: Policy) -> Allowance:
    """Where ``entity_id`` stands now under ``policy``: its grants (the ones due counted), its
    subscription, and one query per distinct rule window."""
    now = timezone.now()
    by_period: dict[int | None, RuleUsage] = {}
    for rule in policy.rules:
        if rule.period_days not in by_period:
            by_period[rule.period_days] = _usage(entity_id, rule.period_days, now)
    counted, plan = with_due(grants_of(entity_id, now), policy, subscription_of(entity_id), now)
    return summarise(
        policy.rules,
        [by_period[r.period_days] for r in policy.rules],
        counted,
        now,
        plan.next_grant_at,
    )


def reader_status(reader: str, entity_id: Any, user_id: Any = None) -> dict[str, Any]:
    """What a page needs to draw the read button: ``enabled`` (a usable policy AND ``reader``'s
    switch in it) and the allowance. A switched-off reader sends no numbers. With ``user_id`` (the
    signed-in person, from their verified token - never the client's word) also ``first_read_at``,
    which sizes the reading cat (the user, 2026-11-26)."""
    policy = load_policy()
    if policy is None or not policy.reader(reader).enabled:
        return {"enabled": False}
    answer = {"enabled": True, **status(entity_id, policy).as_dict()}
    if user_id is not None:
        answer["first_read_at"] = first_read_at(user_id)
    return answer


def first_read_at(user_id: Any) -> str | None:
    """When this person's first read was answered - any company, either reader - as ISO, or None
    (never read, or nobody signed in). The page shows the large reading cat for 24 hours from it,
    the small one after (the user, 2026-11-26: worked out from ``ai_read``, no column of its own).
    ``ai_read`` has no ``user_id`` index; a few hundred rows today, so the plain query is fine -
    an index is a schema change, asked for when the table grows."""
    if user_id is None:
        return None
    first = (
        apps.get_model("minty_db", "AiRead")
        .objects.filter(user_id=user_id, status="ok")
        .aggregate(first=Min("created_at"))["first"]
    )
    return first.isoformat() if first else None


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
    it)``, or ``(None, allowance)`` when no grant has room or a rule is used up. The grants due now
    (the AI trial, this period's) are given first. Serialised per company (a Postgres advisory
    lock, held to the end of the transaction), across both services."""
    if module not in MODULES:
        raise ValueError(f"unknown module {module!r}")
    read_model = apps.get_model("minty_db", "AiRead")
    grant_model = apps.get_model("minty_db", "AiGrant")
    with transaction.atomic():  # pyright: ignore[reportGeneralTypeIssues] - Django has no stubs
        _lock(entity_id)
        now = timezone.now()
        plan = plan_grants(grants_of(entity_id, now), subscription_of(entity_id), policy, now)
        for due in plan.due:
            grant_model.objects.create(
                entity_id=entity_id,
                tokens_total=due.tokens,
                reads_total=due.reads,
                expires_at=due.expires_at,
                source=due.source,
                granted_by=GRANTED_BY_POLICY,
            )
        before = status(entity_id, policy)
        if before.blocked:
            return None, before
        charge = min((g for g in grants_of(entity_id, now) if g.has_room(now)), key=charge_order)
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
