"""minty_shared.periods = minty-subscription-api's billing arithmetic. These are its oracle cases
(billing/tests/engine/test_billing_engine.py: figures Stripe produced), so the AI allowance's
period grants fall on the same days and prorate by the same fraction as the invoices."""

from datetime import UTC, datetime, timedelta

import pytest

from minty_shared.periods import (
    Period,
    add_months,
    period_containing,
    prorate,
    round_half_away,
)


def dt(y, m, d, hour=13):
    return datetime(y, m, d, hour, tzinfo=UTC)


def test_period_containing_is_derived_from_the_anchor():
    period = period_containing(dt(2026, 9, 8), dt(2026, 11, 10))
    assert (period.start, period.end) == (dt(2026, 11, 8), dt(2026, 12, 8))


def test_a_moment_exactly_on_the_anchor_starts_the_new_period():
    anchor = dt(2026, 9, 8)
    assert period_containing(anchor, dt(2026, 10, 8)).start == dt(2026, 10, 8)
    assert period_containing(anchor, dt(2026, 10, 8) - timedelta(seconds=1)).start == dt(2026, 9, 8)


def test_month_end_anchors_clamp_without_sticking():
    anchor = dt(2027, 1, 31)
    assert add_months(anchor, 1) == dt(2027, 2, 28)
    assert add_months(anchor, 2) == dt(2027, 3, 31)  # not 28 Mar
    assert period_containing(anchor, dt(2027, 3, 15)).start == dt(2027, 2, 28)


def test_a_leap_day_anchor_keeps_day_29_rather_than_the_month_end():
    anchor = dt(2028, 2, 29, hour=12)
    assert period_containing(anchor, dt(2028, 3, 1, hour=12)) == Period(
        dt(2028, 2, 29, hour=12), dt(2028, 3, 29, hour=12)
    )
    assert period_containing(anchor, dt(2029, 2, 1, hour=12)) == Period(
        dt(2029, 1, 29, hour=12), dt(2029, 2, 28, hour=12)
    )
    assert period_containing(anchor, dt(2029, 3, 1, hour=12)) == Period(
        dt(2029, 2, 28, hour=12), dt(2029, 3, 29, hour=12)
    )


def test_periods_tile_without_gaps_or_overlap():
    anchor = at = dt(2026, 1, 31)
    for _ in range(14):
        period = period_containing(anchor, at)
        assert period.contains(at)
        assert period_containing(anchor, period.end).start == period.end
        at = period.end + timedelta(days=1)


def test_naive_datetimes_are_refused():
    with pytest.raises(ValueError):
        period_containing(datetime(2026, 9, 8), dt(2026, 10, 1))


def test_prorate_is_by_elapsed_time_rounded_half_away_from_zero():
    period = Period(dt(2026, 9, 8), dt(2026, 10, 8))  # 30 days
    assert prorate(40000, period, dt(2026, 10, 1)) == 9333  # 7/30
    assert prorate(40000, period, period.start) == 40000
    assert prorate(40000, period, period.start - timedelta(days=3)) == 40000
    assert prorate(40000, period, period.end) == 0
    assert prorate(40000, period, dt(2026, 10, 7, hour=1)) == 2000  # 36 hours: 1.5/30
    assert prorate(40000, Period(dt(2026, 11, 8), dt(2026, 12, 8)), dt(2026, 11, 10)) == 37333
    assert prorate(12000, Period(dt(2026, 8, 20), dt(2026, 9, 20)), dt(2026, 8, 28)) == 8903


def test_the_user_s_example_18_of_30_days_left_is_168000_tokens():
    period = Period(dt(2026, 12, 1), dt(2026, 12, 31))
    assert prorate(280000, period, dt(2026, 12, 13)) == 168000


def test_halves_round_away_from_zero():
    assert (round_half_away(2.5), round_half_away(-2.5), round_half_away(2.49)) == (3, -3, 2)
