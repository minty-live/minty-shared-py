"""Billing periods and proration - minty-subscription-api's arithmetic, for code that must agree
with billing without importing it (the AI allowance's period grants, the user 2026-11-27).

A COPY of ``minty-subscription-api/billing/services/billing.py``'s ``Period``, ``add_months``,
``months_between``, ``period_containing`` and ``prorate`` (that service pins an older
minty-shared, so it cannot import this yet); the tests are its oracle cases. Change both
together, or move billing onto this module first.

* A period is half-open, ``[start, end)``, derived from the payer's ANCHOR every time: the day of
  month clamps without sticking (31 Jan -> 28 Feb -> 31 Mar).
* A proration is by ELAPSED SECONDS, rounded half away from zero, once.

Standard library only.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class Period:
    """One billing period, half-open: ``start`` inclusive, ``end`` exclusive."""

    start: datetime
    end: datetime

    @property
    def seconds(self) -> int:
        return int((self.end - self.start).total_seconds())

    def contains(self, at: datetime) -> bool:
        return self.start <= at < self.end

    def remaining_seconds(self, at: datetime) -> int:
        """Seconds from ``at`` to the end of the period; the whole period before it, 0 after."""
        if at <= self.start:
            return self.seconds
        if at >= self.end:
            return 0
        return int((self.end - at).total_seconds())


def _aware(name: str, value: datetime) -> datetime:
    """A naive datetime is refused rather than guessed: it would move a period boundary."""
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def add_months(moment: datetime, count: int) -> datetime:
    """``moment`` shifted by ``count`` whole months, the day clamped to the month's length - and
    re-derived from ``moment`` each time, so the clamp never sticks."""
    moment = _aware("moment", moment)
    total = moment.month - 1 + count
    year = moment.year + total // 12
    month = total % 12 + 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def months_between(anchor: datetime, at: datetime) -> int:
    """Whole billing months from ``anchor`` to ``at`` (negative before it)."""
    months = (at.year - anchor.year) * 12 + (at.month - anchor.month)
    if at < add_months(anchor, months):
        months -= 1
    return months


def period_containing(anchor: datetime, at: datetime, *, interval_months: int = 1) -> Period:
    """The billing period ``at`` falls in, for a subscription anchored at ``anchor``. Monthly is
    the only interval billing sells (and has checked)."""
    if interval_months < 1:
        raise ValueError("interval_months must be at least 1")
    anchor = _aware("anchor", anchor)
    at = _aware("at", at)
    index = (months_between(anchor, at) // interval_months) * interval_months
    return Period(add_months(anchor, index), add_months(anchor, index + interval_months))


def round_half_away(value: float) -> int:
    """Whole units, half away from zero (``round`` is banker's, ``int`` truncates)."""
    return int(value + 0.5) if value >= 0 else -int(-value + 0.5)


def prorate(amount: int, period: Period, at: datetime) -> int:
    """``amount`` scaled to the part of ``period`` still to come at ``at``: all of it at or before
    the start, none at or after the end."""
    at = _aware("at", at)
    total = period.seconds
    if total <= 0:
        return 0
    return round_half_away(amount * period.remaining_seconds(at) / total)
