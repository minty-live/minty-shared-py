"""The allowance's rules - parsing, grants and where a company stands. Pure: the database half
(reserve, the default grant, charging a grant, the per-company lock) is tested in the APIs against
a real Postgres."""

from datetime import UTC, datetime, timedelta

import pytest

from minty_shared import ai
from minty_shared.django.ai_allowance import (
    READER_INVOICE,
    READER_RECEIPT,
    GrantSpec,
    GrantUsage,
    PolicyError,
    Rule,
    RuleUsage,
    parse_grant,
    parse_policy,
    parse_rules,
    summarise,
)

NOW = datetime(2026, 11, 18, 9, 0, tzinfo=UTC)
#: ai_policy.model_prices as the database holds it (the user, 2026-11-24).
PRICES = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50, "cached_input": 0.03},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00, "cached_input": 0.15},
}


def grant(tokens=None, reads=None, *, used_tokens=0, used_reads=0, expires=None, gid="g1"):
    return GrantUsage(
        id=gid,
        tokens_total=tokens,
        reads_total=reads,
        expires_at=expires,
        tokens_used=used_tokens,
        reads_used=used_reads,
    )


# --- parsing ---------------------------------------------------------------------------------


def test_the_three_controls_combine_in_one_rule_or_several():
    assert parse_rules(
        [{"reads": 10, "tokens": 3000, "period_days": 1}, {"reads": 100, "period_days": 30}]
    ) == (Rule(10, 3000, 1), Rule(100, None, 30))


def test_no_rules_is_no_extra_cap():
    assert parse_rules([]) == ()


@pytest.mark.parametrize(
    "raw, says",
    [
        ({"reads": 10}, "must be a list"),
        (["reads"], "must be an object"),
        ([{"reads": 10, "per": "day"}], "unknown keys"),
        ([{"period_days": 30}], "limits nothing"),
        ([{"reads": -1}], "at least 0"),
        ([{"reads": "10"}], "whole number"),
        ([{"reads": True}], "whole number"),
        ([{"reads": 10, "period_days": 0}], "period_days must be"),
    ],
)
def test_a_rule_that_cannot_be_used_is_refused_with_its_reason(raw, says):
    with pytest.raises(PolicyError, match=says):
        parse_rules(raw)


def test_the_preview_grant_parses_and_none_means_no_automatic_grant():
    assert parse_grant({"tokens": 10000}) == GrantSpec(tokens=10000)
    assert parse_grant({"tokens": 50000, "reads": 10, "days": 30}) == GrantSpec(50000, 10, 30)
    assert parse_grant(None) is None


@pytest.mark.parametrize(
    "raw, says",
    [
        ([10000], "must be an object"),
        ({"tokens": 10000, "credits": 5}, "unknown keys"),
        ({"days": 30}, "gives nothing"),
        ({"tokens": 0}, "at least 1"),
        ({"tokens": 10000, "days": 0}, "days must be"),
    ],
)
def test_a_grant_that_cannot_be_used_is_refused_with_its_reason(raw, says):
    with pytest.raises(PolicyError, match=says):
        parse_grant(raw)


def test_the_models_are_primary_then_backup():
    policy = parse_policy(
        "gemini-3.5-flash-lite", " gemini-3.5-flash ", [], {"tokens": 10000}, None, PRICES
    )
    assert policy.models == ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
    assert policy.default_grant == GrantSpec(tokens=10000)
    assert parse_policy("gemini-3.5-flash", "", [], None, None, PRICES).models == [
        "gemini-3.5-flash"
    ]


def test_a_policy_without_a_primary_model_is_refused():
    with pytest.raises(PolicyError, match="primary_model"):
        parse_policy("  ", None, [])


# --- each model's price (the user, 2026-11-24: ai_policy.model_prices) -------------------------


def test_every_reader_carries_each_model_s_price():
    reader = parse_policy("gemini-3.5-flash-lite", None, [], None, None, PRICES).reader(
        READER_RECEIPT
    )
    assert reader.prices["gemini-3.5-flash-lite"] == ai.ModelPrice(0.30, 2.50, 0.03)
    assert reader.prices["gemini-3.5-flash"] == ai.ModelPrice(1.50, 9.00, 0.15)


@pytest.mark.parametrize(
    "primary, backup, prices, says",
    [
        ("gemini-3.8-flash", None, PRICES, "no price for gemini-3.8-flash"),
        ("gemini-3.5-flash", "gemini-3.8-flash", PRICES, "no price for gemini-3.8-flash"),
        ("gemini-3.5-flash", None, None, "must be an object"),
        ("gemini-3.5-flash", None, [], "must be an object"),
        ("gemini-3.5-flash", None, {"gemini-3.5-flash": {"input": 1.5, "output": 9}}, "exactly"),
        (
            "gemini-3.5-flash",
            None,
            {"gemini-3.5-flash": {"input": -1, "output": 9, "cached_input": 0.15}},
            "at least 0",
        ),
        (
            "gemini-3.5-flash",
            None,
            {"gemini-3.5-flash": {"input": "1.5", "output": 9, "cached_input": 0.15}},
            "at least 0",
        ),
    ],
)
def test_a_policy_whose_cost_cannot_be_worked_out_is_refused(primary, backup, prices, says):
    with pytest.raises(PolicyError, match=says):
        parse_policy(primary, backup, [], None, None, prices)


# --- the readers' settings (owner, 2026-11-20: ai_policy, not the environment) ----------------


def test_each_reader_gets_its_own_switch_and_the_shared_settings():
    settings = {
        "receipt_reader_on": True,
        "invoice_reader_on": False,
        "timeout_s": 40,
        "max_output_tokens": 6000,
        "thinking_level": "medium",
        "max_file_mb": 8,
    }
    policy = parse_policy("gemini-3.5-flash", None, [], None, settings, PRICES)
    receipt, invoice = policy.reader(READER_RECEIPT), policy.reader(READER_INVOICE)
    assert receipt.enabled is True and invoice.enabled is False
    assert (receipt.label, invoice.label) == ("receipt_ai", "invoice_ai")
    assert receipt.timeout_seconds == 40 and receipt.max_output_tokens == 6000
    assert receipt.thinking_level == "medium" and receipt.max_file_bytes == 8 * 1024 * 1024
    assert invoice.timeout_seconds == 40


def test_a_reader_ships_dark_and_keeps_the_columns_defaults():
    reader = parse_policy("gemini-3.5-flash", None, [], None, None, PRICES).reader(READER_INVOICE)
    assert reader.enabled is False
    assert (reader.timeout_seconds, reader.max_output_tokens, reader.thinking_level) == (
        25,
        4096,
        "low",
    )
    assert reader.max_file_mb == 10


def test_an_unknown_reader_is_refused():
    with pytest.raises(ValueError, match="unknown reader"):
        parse_policy("gemini-3.5-flash", None, [], None, None, PRICES).reader("sales")


@pytest.mark.parametrize(
    "raw, says",
    [
        ({"timeout_s": 4}, "timeout_s"),
        ({"timeout_s": 121}, "at most 120"),
        ({"max_output_tokens": 100}, "max_output_tokens"),
        ({"max_file_mb": 21}, "at most 20"),
        ({"max_file_mb": "10"}, "whole number"),
        ({"thinking_level": "extreme"}, "thinking_level"),
        ({"receipt_reader_on": "yes"}, "true or false"),
        ({"temperature": 1}, "unknown reader settings"),
        # owner, 2026-11-23: no region setting - every read is on Vertex's global endpoint
        ({"location": "global"}, "unknown reader settings"),
    ],
)
def test_a_setting_out_of_its_range_is_refused_with_its_reason(raw, says):
    with pytest.raises(PolicyError, match=says):
        parse_policy("gemini-3.5-flash", None, [], None, raw, PRICES)


@pytest.mark.parametrize("on, expected", [(True, True), (False, False)])
def test_the_page_s_reader_follows_the_reader_s_switch(monkeypatch, on, expected):
    from minty_shared.django import ai_allowance

    policy = parse_policy("gemini-3.5-flash", None, [], None, {"receipt_reader_on": on}, PRICES)
    monkeypatch.setattr(ai_allowance, "load_policy", lambda: policy)
    monkeypatch.setattr(
        ai_allowance, "status", lambda entity_id, p: summarise((), [], [grant(tokens=100)], NOW)
    )
    status = ai_allowance.reader_status(READER_RECEIPT, "ent-1")
    assert status["enabled"] is expected
    assert ("percent_left" in status) is expected
    assert ai_allowance.reader_status(READER_INVOICE, "ent-1") == {"enabled": False}


def test_no_usable_policy_means_no_reader(monkeypatch):
    from minty_shared.django import ai_allowance

    monkeypatch.setattr(ai_allowance, "load_policy", lambda: None)
    assert ai_allowance.reader_status(READER_RECEIPT, "ent-1") == {"enabled": False}


# --- a company's grants ----------------------------------------------------------------------


def test_a_fresh_grant_is_all_left():
    allowance = summarise((), [], [grant(tokens=10000)], NOW)
    assert (allowance.percent_left, allowance.blocked) == (100, False)
    assert allowance.reads_left is None  # nothing counts reads


def test_the_percent_left_is_of_the_tokens():
    allowance = summarise((), [], [grant(tokens=10000, used_tokens=4482)], NOW)
    assert allowance.percent_left == 56 and not allowance.blocked


def test_anything_left_never_reads_zero_percent():
    assert summarise((), [], [grant(tokens=10000, used_tokens=9990)], NOW).percent_left == 1


def test_a_used_up_grant_stops_reading_for_good():
    allowance = summarise((), [], [grant(tokens=10000, used_tokens=10450)], NOW)
    assert allowance.blocked and allowance.percent_left == 0 and allowance.resumes_at is None


def test_a_top_up_adds_to_what_is_left():
    used_up = grant(tokens=10000, used_tokens=10450, gid="first")
    top_up = grant(tokens=40000, gid="second")
    allowance = summarise((), [], [used_up, top_up], NOW)
    assert not allowance.blocked and allowance.percent_left == 100  # the spent grant is not counted


def test_an_expired_grant_no_longer_counts():
    expired = grant(tokens=10000, expires=NOW - timedelta(seconds=1))
    assert summarise((), [], [expired], NOW).blocked
    later = grant(tokens=10000, expires=NOW + timedelta(days=1))
    assert not summarise((), [], [later], NOW).blocked


def test_no_grant_and_no_default_cannot_read():
    allowance = summarise((), [], [], NOW)
    assert allowance.blocked and allowance.percent_left == 0


def test_a_reads_only_grant_counts_reads():
    allowance = summarise((), [], [grant(reads=10, used_reads=3)], NOW)
    assert (allowance.percent_left, allowance.reads_left, allowance.reads_limit) == (70, 7, 10)


def test_a_grant_with_both_caps_stops_at_the_first_used_up():
    both = grant(tokens=50000, reads=10, used_tokens=20000, used_reads=10)
    allowance = summarise((), [], [both], NOW)
    assert allowance.blocked and allowance.percent_left == 0


# --- the extra caps over every company -------------------------------------------------------


def test_a_daily_cap_pauses_reading_until_the_window_moves():
    rule = Rule(reads=10, tokens=3000, period_days=1)
    oldest = NOW - timedelta(hours=5)
    allowance = summarise(
        (rule,), [RuleUsage(reads=1, tokens=5300, oldest=oldest)], [grant(tokens=50000)], NOW
    )
    assert allowance.blocked and allowance.reads_left == 9
    assert allowance.resumes_at == oldest + timedelta(days=1)
    assert allowance.percent_left == 100  # the grant itself is untouched here


def test_a_used_up_grant_has_no_resume_time_whatever_the_caps():
    rule = Rule(reads=1, period_days=1)
    allowance = summarise(
        (rule,),
        [RuleUsage(reads=1, oldest=NOW)],
        [grant(tokens=10000, used_tokens=12000)],
        NOW,
    )
    assert allowance.blocked and allowance.resumes_at is None


def test_the_tightest_reads_cap_is_the_one_shown():
    rules = (Rule(reads=10, period_days=1), Rule(reads=100, period_days=30))
    usage = [RuleUsage(reads=2), RuleUsage(reads=95)]
    allowance = summarise(rules, usage, [grant(tokens=10000)], NOW)
    assert (allowance.reads_left, allowance.reads_limit) == (5, 100) and not allowance.blocked


def test_the_page_gets_iso_times_and_the_percent():
    rule = Rule(reads=1, period_days=1)
    allowance = summarise((rule,), [RuleUsage(reads=1, oldest=NOW)], [grant(tokens=100)], NOW)
    shown = allowance.as_dict()
    assert shown["resumes_at"] == "2026-11-19T09:00:00+00:00" and shown["percent_left"] == 100
