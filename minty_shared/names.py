"""One name for a contact however it is typed (the user, 2026-11-23). Standard library only.

Before Minty adds a contact - in Minty, and in Xero when the company is live there - it looks for
the one the company already has under the same name, so "7 Eleven", "7-ELEVEN" and "7-Eleven" are
one supplier, never three. ``name_key`` is that comparison: Unicode-normalised (NFKC: full-width
letters and spaces are the plain ones), case ignored, "&" read as "and", and everything that is
not a letter or a digit dropped (spaces, hyphens, apostrophes, full stops). A name with no letter
or digit has the empty key and matches nothing.

``search_term`` is what to ask Xero's ``GET /Contacts?searchTerm=`` for: a substring search, so
"7 Eleven" never finds "7-Eleven". Its longest word finds both, and ``name_key`` then decides which
of the answers (one page, up to 100) is the same name.
"""

from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"[^\W_]+")


def name_key(name: object) -> str:
    """``name`` as Minty compares contact names: "7 Eleven" == "7-Eleven" == "7eleven"."""
    text = unicodedata.normalize("NFKC", str(name or "")).casefold()
    if not any(ch.isalnum() for ch in text):
        return ""
    return "".join(ch for ch in text.replace("&", " and ") if ch.isalnum())


def same_name(a: object, b: object) -> bool:
    """True when ``a`` and ``b`` are the same contact name; two names with no letters never are."""
    key = name_key(a)
    return bool(key) and key == name_key(b)


def search_term(name: object) -> str:
    """The longest run of letters and digits in ``name`` (the first of equals), "" when none."""
    words = _WORD.findall(unicodedata.normalize("NFKC", str(name or "")))
    return max(words, key=len, default="")
