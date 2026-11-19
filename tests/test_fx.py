"""The exchange rate for an AI-read amount in another currency (owner, 2026-11-19;
``minty_shared.fx``): which day's rate, the one Frankfurter call, the conversion, and what the
readers hand back. No network: every call goes through a fake opener."""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import date
from decimal import Decimal

import pytest

from minty_shared import fx

REPORT = date(2026, 11, 19)
TODAY = date(2026, 11, 19)


# ---- which day -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        (
            "2026-11-15",
            (date(2026, 11, 15), "receipt"),
        ),  # the receipt's own day (a Sunday: Frankfurter steps back)
        ("", (REPORT, "report_date")),  # none read
        ("15/11/2026", (REPORT, "report_date")),  # not a date
        ("2026-02-30", (REPORT, "report_date")),
        ("2026-11-20", (REPORT, "report_date")),  # the future
        ("2025-11-19", (date(2025, 11, 19), "receipt")),  # exactly a year before the report
        ("2025-11-18", (REPORT, "report_date")),  # a day more
    ],
)
def test_the_rate_is_for_the_receipt_s_day_unless_it_is_empty_future_or_a_year_off(
    document, expected
):
    assert fx.rate_date(document, REPORT, TODAY) == expected


def test_a_report_day_after_today_asks_for_today():
    # Frankfurter v1 answers a future day with its latest rate - never ask for one
    assert fx.rate_date("", date(2026, 12, 1), TODAY) == (TODAY, "report_date")
    assert fx.rate_date("2026-11-30", date(2026, 12, 1), TODAY) == (TODAY, "report_date")


def test_a_receipt_more_than_a_year_after_the_report_is_not_used():
    assert fx.rate_date("2026-11-19", date(2025, 11, 1), TODAY) == (
        date(2025, 11, 1),
        "report_date",
    )


# ---- the call ------------------------------------------------------------------------------------


class Opener:
    """A fake ``urlopen``: records the address and answers what it was given."""

    def __init__(self, answer):
        self.answer = answer
        self.urls: list[str] = []
        self.timeouts: list[float] = []

    def __call__(self, request, timeout):
        self.urls.append(request.full_url)
        self.timeouts.append(timeout)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return io.BytesIO(
            self.answer if isinstance(self.answer, bytes) else json.dumps(self.answer).encode()
        )


def test_a_weekend_day_gets_the_previous_business_day_s_rate_and_says_which(monkeypatch):
    monkeypatch.delenv("FX_API_URL", raising=False)
    opener = Opener({"amount": 1.0, "base": "USD", "date": "2026-11-13", "rates": {"HKD": 7.846}})

    rate = fx.frankfurter_rate("USD", "HKD", date(2026, 11, 15), opener=opener)

    assert rate == fx.Rate(rate=Decimal("7.846"), date=date(2026, 11, 13), source="Frankfurter")
    assert opener.urls == ["https://api.frankfurter.dev/v1/2026-11-15?base=USD&symbols=HKD"]
    assert opener.timeouts == [fx.TIMEOUT_SECONDS]


def test_the_address_can_be_set(monkeypatch):
    monkeypatch.setenv("FX_API_URL", "http://rates.test/v1/")
    opener = Opener({"date": "2026-11-13", "rates": {"HKD": 7.846}})
    fx.frankfurter_rate("USD", "HKD", date(2026, 11, 13), opener=opener)
    assert opener.urls == ["http://rates.test/v1/2026-11-13?base=USD&symbols=HKD"]


@pytest.mark.parametrize(
    "answer",
    [
        urllib.error.HTTPError(
            "u", 404, "not found", None, None
        ),  # a currency v1 doesn't have (TWD)
        urllib.error.HTTPError("u", 422, "invalid", None, None),
        urllib.error.URLError("no route"),
        TimeoutError("slow"),
        b"<html>not json</html>",
        {"date": "2026-11-13", "rates": {}},  # not the asked currency
        {"date": "2026-11-13", "rates": {"HKD": 0}},  # no rate
        {"date": "2026-11-13", "rates": {"HKD": "x"}},
        {"date": "2026-11-20", "rates": {"HKD": 7.8}},  # a day after the one asked
        {"rates": {"HKD": 7.8}},  # no day
    ],
)
def test_no_usable_answer_is_no_rate(answer, caplog):
    assert fx.frankfurter_rate("USD", "HKD", date(2026, 11, 15), opener=Opener(answer)) is None
    assert "exchange rate USD->HKD on 2026-11-15: none" in caplog.text


def test_a_code_that_is_not_three_letters_is_never_asked():
    opener = Opener({"date": "2026-11-13", "rates": {"HKD": 7.8}})
    assert fx.frankfurter_rate("US$", "HKD", date(2026, 11, 15), opener=opener) is None
    assert fx.frankfurter_rate("USD", "hk", date(2026, 11, 15), opener=opener) is None
    assert opener.urls == []


# ---- the conversion ------------------------------------------------------------------------------


def test_the_conversion_rounds_half_up_to_the_company_currency_s_places():
    assert fx.convert(Decimal("12.34"), Decimal("7.846"), 2) == Decimal("96.82")  # 96.81964
    assert fx.convert(Decimal("10.00"), Decimal("0.0005"), 2) == Decimal("0.01")  # 0.005 -> up
    assert fx.convert(Decimal("12.34"), Decimal("156.5"), 0) == Decimal("1931")  # JPY: 1931.21


# ---- what the readers hand back ------------------------------------------------------------------


def _read(amount="12.34", currency="USD", document_date="2026-11-15"):
    return {
        "amount": {"value": amount, "confidence": 0.95, "band": "high", "applied": True},
        "description": {"value": "Taxi", "confidence": 0.9, "band": "high", "applied": True},
        "document_date": {
            "value": document_date,
            "confidence": 0.9,
            "band": "high",
            "applied": True,
        },
        "currency": currency,
        "entity_currency": "HKD",
    }


def _rate(on: date):
    return fx.Rate(rate=Decimal("7.846"), date=date(2026, 11, 13))


def test_another_currency_fills_the_converted_amount_and_says_how():
    asked = []

    def fetch(base, quote, on):
        asked.append((base, quote, on))
        return _rate(on)

    out = fx.exchange_for(
        _read(),
        report_date=REPORT,
        today=TODAY,
        places=2,
        max_amount=Decimal("10000000"),
        fetch=fetch,
    )

    assert asked == [("USD", "HKD", date(2026, 11, 15))]
    assert out["amount"] == {"value": "96.82", "confidence": 0.95, "band": "high", "applied": True}
    assert out["exchange"] == {
        "original_currency": "USD",
        "original_amount": "12.34",
        "rate": "7.846",
        "rate_date": "2026-11-13",
        "date_basis": "receipt",
        "source": "Frankfurter",
        "asked_date": "2026-11-15",  # the Sunday asked for; the rate is Friday's
    }
    assert "exchange_failed" not in out
    assert out["description"]["value"] == "Taxi"  # nothing else changes


def test_no_rate_leaves_the_amount_empty_and_says_so():
    out = fx.exchange_for(
        _read(),
        report_date=REPORT,
        today=TODAY,
        places=2,
        max_amount=Decimal("10000000"),
        fetch=lambda *a: None,
    )
    assert out["amount"]["value"] == "" and out["amount"]["applied"] is False
    assert out["exchange_failed"] == {"currency": "USD"}
    assert "exchange" not in out


def test_the_company_s_currency_or_no_currency_or_no_amount_asks_nothing():
    def fetch(*a):
        raise AssertionError("asked for a rate")

    for read in (_read(currency="HKD"), _read(currency=""), _read(amount="")):
        assert (
            fx.exchange_for(
                read,
                report_date=REPORT,
                today=TODAY,
                places=2,
                max_amount=Decimal("10000000"),
                fetch=fetch,
            )
            == read
        )


def test_the_cap_is_on_the_converted_amount():
    # 2,000,000 JPY is about HKD 100,000: far under the cap; the same in HKD at 6 would be over it
    small = fx.exchange_for(
        _read(amount="2000000", currency="JPY"),
        report_date=REPORT,
        today=TODAY,
        places=2,
        max_amount=Decimal("10000000"),
        fetch=lambda *a: fx.Rate(Decimal("0.05"), date(2026, 11, 13)),
    )
    assert small["amount"]["value"] == "100000.00"
    over = fx.exchange_for(
        _read(amount="2000000"),
        report_date=REPORT,
        today=TODAY,
        places=2,
        max_amount=Decimal("10000000"),
        fetch=lambda *a: fx.Rate(Decimal("6"), date(2026, 11, 13)),
    )
    assert (
        over["amount"]["value"] == "" and "exchange" not in over and "exchange_failed" not in over
    )


def test_a_receipt_dated_long_ago_uses_the_report_day():
    asked = []
    fx.exchange_for(
        _read(document_date="2024-01-01"),
        report_date=date(2026, 11, 10),
        today=TODAY,
        places=2,
        max_amount=Decimal("10000000"),
        fetch=lambda b, q, on: asked.append(on) or _rate(on),
    )
    assert asked == [date(2026, 11, 10)]


# ---- what a save sends back ----------------------------------------------------------------------

SAVED = {
    "original_currency": "USD",
    "original_amount": "12.34",
    "exchange_rate": "7.846",
    "exchange_rate_date": "2026-11-13",
    "exchange_rate_basis": "receipt",
    "exchange_rate_source": "Frankfurter",
}


def _check(sent, amount="96.82", places=2):
    return fx.saved_exchange(
        sent, amount=Decimal(amount), company_currency="HKD", places=places, today=TODAY
    )


def test_a_save_without_the_details_keeps_none():
    assert _check({}) is None
    assert _check({"description": "Taxi"}) is None


def test_a_save_with_all_six_matching_its_amount_keeps_them_as_columns():
    assert _check(SAVED) == {
        "original_currency": "USD",
        "original_amount": Decimal("12.34"),
        "exchange_rate": Decimal("7.846"),
        "exchange_rate_date": date(2026, 11, 13),
        "exchange_rate_basis": "receipt",
        "exchange_rate_source": "Frankfurter",
    }
    assert (
        _check({**SAVED, "exchange_rate_basis": "report_date"})["exchange_rate_basis"]
        == "report_date"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"exchange_rate": None},  # not all six
        {"original_currency": "HKD"},  # the company's own
        {"original_currency": "usd"},
        {"original_amount": "-1"},
        {"original_amount": "12.345"},
        {"exchange_rate": "0"},
        {"exchange_rate": "abc"},
        {"exchange_rate_date": "2026-11-20"},  # after today
        {"exchange_rate_date": "13/11/2026"},
        {"exchange_rate_basis": "invoice"},
        {"exchange_rate_source": "ECB"},
    ],
)
def test_details_that_cannot_be_right_are_refused(change):
    sent = {k: v for k, v in {**SAVED, **change}.items() if v is not None}
    with pytest.raises(fx.ExchangeRefusedError) as refused:
        _check(sent)
    assert str(refused.value) == fx.EXCHANGE_REFUSED


def test_the_refusal_names_the_document():
    assert fx.EXCHANGE_REFUSED == fx.exchange_refused("receipt")
    assert fx.exchange_refused("invoice") == (
        "The exchange-rate details don't match this amount. Read the invoice again or enter the amount yourself."
    )


def test_details_that_do_not_give_the_amount_are_refused():
    with pytest.raises(fx.ExchangeRefusedError):
        _check(SAVED, amount="100.00")
    assert _check(SAVED, amount="96.82") is not None


def test_the_company_s_places_decide_the_match():
    jpy = {**SAVED, "exchange_rate": "156.5"}
    assert fx.saved_exchange(
        jpy, amount=Decimal("1931"), company_currency="JPY", places=0, today=TODAY
    )


def test_an_invoice_s_date_is_read_from_its_own_field():
    asked = []
    read = _read(document_date="")
    read["invoice_date"] = {
        "value": "2026-11-10",
        "confidence": 0.9,
        "band": "high",
        "applied": True,
    }
    fx.exchange_for(
        read,
        report_date=REPORT,
        today=TODAY,
        places=2,
        max_amount=Decimal("10000000"),
        fetch=lambda b, q, on: asked.append(on) or _rate(on),
        date_field="invoice_date",
    )
    assert asked == [date(2026, 11, 10)]
