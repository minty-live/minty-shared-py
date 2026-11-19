"""The exchange rate for an AI-read amount in another currency (owner, 2026-11-19).

When the reader finds a receipt or invoice in a currency other than the company's, the amount it
suggests is converted at Frankfurter's rate (https://frankfurter.dev, v1: the European Central
Bank's reference rates, no key, 30 currencies). One call per read, when the person presses the read
button; no rate is stored anywhere but on the saved line.

* The day: the document's date (basis ``receipt``); the report's day instead (``report_date``)
  when the document's is empty, unreadable, in the future or more than a year from the report.
  A day after the company's today is asked as today: v1 answers a future day with its latest rate.
* v1 gives a weekend or holiday the previous business day's rate, and its answer's ``date`` is
  the day the rate is for - that is what is kept.
* No usable answer (any error, a currency v1 doesn't have, a timeout) is no rate: the amount is
  left empty for the person to type, and nothing is ever blocked.

Standard library only (``urllib``), like ``tokens``. ``FX_API_URL`` moves the address (tests, a
self-hosted Frankfurter).
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from minty_shared.time import parse_day

logger = logging.getLogger("minty_shared.fx")

SOURCE = "Frankfurter"
DEFAULT_URL = "https://api.frankfurter.dev/v1"
TIMEOUT_SECONDS = 5.0
BASIS_RECEIPT = "receipt"
BASIS_REPORT = "report_date"
MAX_DAYS_FROM_REPORT = 365

_CODE = re.compile(r"[A-Z]{3}")


@dataclass(frozen=True)
class Rate:
    """One unit of the document's currency in the company's, for ``date`` (the day Frankfurter
    says the rate is for - the previous business day when asked for a weekend or holiday)."""

    rate: Decimal
    date: date
    source: str = SOURCE


def rate_date(document_date: object, report_date: date, today: date) -> tuple[date, str]:
    """The day to ask a rate for, and why: ``(day, "receipt" | "report_date")``."""
    doc = parse_day(document_date)
    if doc is not None and doc <= today and abs((doc - report_date).days) <= MAX_DAYS_FROM_REPORT:
        return doc, BASIS_RECEIPT
    return min(report_date, today), BASIS_REPORT


def _base_url() -> str:
    return (os.environ.get("FX_API_URL") or DEFAULT_URL).rstrip("/")


def frankfurter_rate(
    base: str,
    quote: str,
    on: date,
    *,
    timeout: float = TIMEOUT_SECONDS,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Rate | None:
    """``base`` -> ``quote`` for ``on``, or None when there is no usable answer. One try; one log
    line either way (the pair and the days, never an amount)."""
    if not (_CODE.fullmatch(base or "") and _CODE.fullmatch(quote or "")):
        return None
    url = f"{_base_url()}/{on.isoformat()}?base={base}&symbols={quote}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with opener(request, timeout=timeout) as answer:  # noqa: S310 - a fixed https address
            body = json.loads(answer.read())
        rate = Decimal(str(body["rates"][quote]))
        day = date.fromisoformat(body["date"])
        if not rate.is_finite() or rate <= 0 or day > on:
            raise ValueError("not a usable rate")
    except urllib.error.HTTPError as exc:
        logger.warning("exchange rate %s->%s on %s: none (HTTP %s)", base, quote, on, exc.code)
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("exchange rate %s->%s on %s: none (%s)", base, quote, on, type(exc).__name__)
        return None
    except (ValueError, KeyError, TypeError, InvalidOperation) as exc:
        logger.warning(
            "exchange rate %s->%s on %s: none (bad answer: %s)", base, quote, on, type(exc).__name__
        )
        return None
    logger.info("exchange rate %s->%s on %s: %s for %s", base, quote, on, rate, day)
    return Rate(rate=rate, date=day)


def convert(original: Decimal, rate: Decimal, places: int) -> Decimal:
    """``original`` x ``rate`` in the company currency's places, half up."""
    return (original * rate).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def _emptied(field: dict[str, Any]) -> dict[str, Any]:
    return {**field, "value": "", "applied": False}


def exchange_for(
    suggestions: dict[str, Any],
    *,
    report_date: date,
    today: date,
    places: int,
    max_amount: Decimal,
    fetch: Callable[[str, str, date], Rate | None] = frankfurter_rate,
    date_field: str = "document_date",
) -> dict[str, Any]:
    """The readers' one step after a read: a document in another currency gets its amount
    converted, with ``exchange`` saying how (``original_currency``, ``original_amount``, ``rate``,
    ``rate_date``, ``date_basis``, ``source``, and ``asked_date``: the day asked for, which a weekend
    or holiday moves ``rate_date`` back from). No rate: the amount is emptied and
    ``exchange_failed: {currency}`` says why. A converted amount over ``max_amount`` is emptied too.
    The company's own currency, no currency or no amount: unchanged, and nothing is asked.
    ``date_field`` names the suggestion holding the document's date (an invoice's ``invoice_date``)."""
    currency = str(suggestions.get("currency") or "")
    company = str(suggestions.get("entity_currency") or "")
    amount = suggestions.get("amount")
    if not currency or not company or currency == company or not isinstance(amount, dict):
        return suggestions
    try:
        original = Decimal(str(amount.get("value") or ""))
    except InvalidOperation:
        return suggestions
    if not original.is_finite() or original <= 0:
        return suggestions

    document = suggestions.get(date_field)
    day, basis = rate_date(
        document.get("value") if isinstance(document, dict) else document, report_date, today
    )
    rate = fetch(currency, company, day)
    out = dict(suggestions)
    if rate is None:
        out["amount"] = _emptied(amount)
        out["exchange_failed"] = {"currency": currency}
        return out
    converted = convert(original, rate.rate, places)
    if not Decimal(0) < converted <= max_amount:
        out["amount"] = _emptied(amount)
        return out
    out["amount"] = {**amount, "value": format(converted, "f")}
    out["exchange"] = {
        "original_currency": currency,
        "original_amount": format(original.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f"),
        "rate": format(rate.rate, "f"),
        "rate_date": rate.date.isoformat(),
        "date_basis": basis,
        "source": rate.source,
        # the day asked for: the page says why the rate's day differs (a weekend or holiday)
        "asked_date": day.isoformat(),
    }
    return out


# --------------------------------------------------------------------------- what a save sends back

COLUMNS = (
    "original_currency",
    "original_amount",
    "exchange_rate",
    "exchange_rate_date",
    "exchange_rate_basis",
    "exchange_rate_source",
)


def exchange_refused(document: str = "receipt") -> str:
    """The refusal of six details that can't be right, naming what was read (a receipt, an invoice)."""
    return f"The exchange-rate details don't match this amount. Read the {document} again or enter the amount yourself."


EXCHANGE_REFUSED = exchange_refused("receipt")


class ExchangeRefusedError(ValueError):
    """The six details a save sent can't be right: the save is refused (422, ``EXCHANGE_REFUSED``)."""

    def __init__(self) -> None:
        super().__init__(EXCHANGE_REFUSED)


def saved_exchange(
    sent: Any, *, amount: Decimal, company_currency: str, places: int, today: date
) -> dict[str, Any] | None:
    """The six columns a saved line keeps, checked: None when the save sent none of them; all six
    when they hold together - another currency than the company's, a rate above 0 for a day not
    after today, a known basis, Frankfurter - and give ``amount`` (original x rate in the company
    currency's places). Anything else raises ``ExchangeRefusedError``: the web sends them only while
    the amount is still the converted one, so a mismatch is never stored."""
    given = {name: sent.get(name) for name in COLUMNS}
    present = {k: v for k, v in given.items() if v not in (None, "")}
    if not present:
        return None
    if len(present) != len(COLUMNS):
        raise ExchangeRefusedError()
    currency = str(present["original_currency"])
    try:
        original = Decimal(str(present["original_amount"]))
        rate = Decimal(str(present["exchange_rate"]))
    except InvalidOperation:
        raise ExchangeRefusedError() from None
    day = parse_day(present["exchange_rate_date"])
    basis = present["exchange_rate_basis"]
    if (
        not _CODE.fullmatch(currency)
        or currency == company_currency
        or not original.is_finite()
        or original <= 0
        or original != original.quantize(Decimal("0.01"))
        or not rate.is_finite()
        or rate <= 0
        or day is None
        or day > today
        or basis not in (BASIS_RECEIPT, BASIS_REPORT)
        or present["exchange_rate_source"] != SOURCE
        or convert(original, rate, places) != amount
    ):
        raise ExchangeRefusedError()
    return {
        "original_currency": currency,
        "original_amount": original,
        "exchange_rate": rate,
        "exchange_rate_date": day,
        "exchange_rate_basis": basis,
        "exchange_rate_source": SOURCE,
    }
