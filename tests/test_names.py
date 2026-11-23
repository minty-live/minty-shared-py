"""minty_shared.names: one name for a contact however it is typed (the user, 2026-11-23)."""

from __future__ import annotations

import pytest

from minty_shared import names


@pytest.mark.parametrize(
    "a, b",
    [
        ("7 Eleven", "7-Eleven"),
        ("7-ELEVEN", "7eleven"),
        ("  Wellcome   Supermarket ", "wellcome supermarket"),
        ("A&B Trading", "A and B Trading"),
        ("McDonald's", "McDonalds"),
        ("Café de Coral", "CAFÉ DE CORAL"),
        ("ＡＢＣ　Ｌｔｄ．", "ABC Ltd"),  # full-width letters and space (NFKC)
        ("滬江", " 滬江 "),
    ],
)
def test_same_name_however_it_is_typed(a, b):
    assert names.name_key(a) == names.name_key(b)
    assert names.name_key(a) != ""


@pytest.mark.parametrize(
    "a, b",
    [
        ("7-Eleven", "7-Eleven Central"),
        ("Wu Kong", "滬江"),
        ("ABC Ltd", "ABD Ltd"),
    ],
)
def test_different_names_stay_different(a, b):
    assert names.name_key(a) != names.name_key(b)


@pytest.mark.parametrize("blank", [None, "", "   ", "-", "&", "...", 0])
def test_a_name_with_no_letters_or_digits_matches_nothing(blank):
    assert names.name_key(blank) == ""


def test_same_name_never_matches_two_blanks():
    assert not names.same_name("-", "...")
    assert not names.same_name(None, "")
    assert names.same_name("7 Eleven", "7-Eleven")


@pytest.mark.parametrize(
    "name, term",
    [
        ("7-Eleven", "Eleven"),
        ("7 Eleven", "Eleven"),
        ("Wellcome Supermarket", "Supermarket"),
        ("McDonald's", "McDonald"),
        ("滬江", "滬江"),
        ("  ABC  ", "ABC"),
        ("-", ""),
    ],
)
def test_search_term_is_the_longest_word(name, term):
    # Xero's searchTerm is a substring search, so "7 Eleven" never finds "7-Eleven" - the
    # longest run of letters and digits finds both, and name_key then decides
    assert names.search_term(name) == term
