"""minty_shared.accounts: a company's account rules (owner, 2026-11-12), lifted from
minty-onboarding-api's accounting service so the hub's Settings checks the same."""

import pytest

from minty_shared import accounts


def test_the_next_code_follows_the_types_highest_and_skips_a_taken_one():
    rows = [("400", "expense"), ("401", "expense"), ("200", "sales"), ("402", "sales")]
    assert accounts.next_code(rows, "expense") == "403"  # 402 is taken by a sales account
    assert accounts.next_code(rows, "overheads") == "403"  # starts at 400, 400-402 taken
    assert accounts.next_code([], "bank") == "090"
    assert accounts.next_code([("ABC", "equity")], "equity") == "960"


def test_a_clean_account_passes():
    clean, fields = accounts.check_fields(
        {"code": " 469 ", "name": "Rent  and   rates", "type": "expense"}
    )
    assert fields == {}
    assert clean == {
        "code": "469",
        "name": "Rent and rates",
        "type": "expense",
        "description": None,
    }


@pytest.mark.parametrize(
    "payload, field, sentence",
    [
        ({"code": "1", "name": "X"}, "type", "Choose the account type to continue."),
        (
            {"name": "X", "type": "expense"},
            "code",
            "The code is still empty. Add a code to continue.",
        ),
        (
            {"code": "1" * 11, "name": "X", "type": "expense"},
            "code",
            "The code can be up to 10 characters. Shorten it to continue.",
        ),
        (
            {"code": "4 0", "name": "X", "type": "expense"},
            "code",
            "The code can use letters, numbers, dots and dashes. Remove the other characters.",
        ),
        (
            {"code": "1", "type": "expense"},
            "name",
            "The name is still empty. Add a name to continue.",
        ),
        (
            {"code": "1", "name": "n" * 81, "type": "expense"},
            "name",
            "The name can be up to 80 characters. Shorten it to continue.",
        ),
        (
            {"code": "1", "name": "X", "type": "expense", "description": "d" * 256},
            "description",
            "The description can be up to 255 characters. Shorten it to continue.",
        ),
    ],
)
def test_each_field_says_its_own_rule(payload, field, sentence):
    assert accounts.check_fields(payload)[1][field] == sentence


def test_a_bank_account_may_go_without_a_code():
    assert accounts.check_fields({"name": "HSBC", "type": "bank"})[1] == {}


def test_a_taken_code_or_name_names_its_holder():
    assert accounts.clash_fields("469", "Rent", "Rent") == {
        "code": "The code 469 is already used by Rent. Choose another code.",
        "name": "Another account is already called Rent. Choose another name.",
    }
    assert accounts.clash_fields("469", None, None) == {}
