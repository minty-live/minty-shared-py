"""A company's chart of accounts: the rules every backend checks an account against (owner,
2026-11-12: onboarding's rules, shared so the hub's Settings and onboarding agree). Pure: the
types, where a type's codes start, the field checks and their Part 11 sentences, the next free
code. Who already holds a code or a name is the caller's database read (``clash_fields``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: Minty's type -> Xero's spelling (the old ``account_info.type`` column the apps still read).
XERO_TYPE: dict[str, str] = {
    "bank": "BANK",
    "current": "CURRENT",
    "fixed": "FIXED",
    "inventory": "INVENTORY",
    "noncurrent": "NONCURRENT",
    "prepayment": "PREPAYMENT",
    "current_liability": "CURRLIAB",
    "liability": "LIABILITY",
    "term_liability": "TERMLIAB",
    "equity": "EQUITY",
    "sales": "SALES",
    "revenue": "REVENUE",
    "other_income": "OTHERINCOME",
    "direct_costs": "DIRECTCOSTS",
    "expense": "EXPENSE",
    "overheads": "OVERHEADS",
    "depreciation": "DEPRECIATN",
}

#: Where a type's codes start when the company has none of that type yet (Xero's demo chart).
CODE_START: dict[str, int] = {
    "bank": 90,
    "current": 600,
    "fixed": 700,
    "inventory": 630,
    "noncurrent": 700,
    "prepayment": 620,
    "current_liability": 800,
    "liability": 800,
    "term_liability": 900,
    "equity": 960,
    "sales": 200,
    "revenue": 200,
    "other_income": 260,
    "direct_costs": 300,
    "expense": 400,
    "overheads": 400,
    "depreciation": 416,
}

#: The type as a person reads it.
TYPE_WORDS: dict[str, str] = {
    "bank": "Bank",
    "current": "Current asset",
    "fixed": "Fixed asset",
    "inventory": "Inventory",
    "noncurrent": "Non-current asset",
    "prepayment": "Prepayment",
    "current_liability": "Current liability",
    "liability": "Liability",
    "term_liability": "Non-current liability",
    "equity": "Equity",
    "sales": "Sales",
    "revenue": "Revenue",
    "other_income": "Other income",
    "direct_costs": "Direct costs",
    "expense": "Expense",
    "overheads": "Overhead",
    "depreciation": "Depreciation",
}

CODE_MAX = 10
NAME_MAX = 80  # account_info.name is 80 until CONTRACT widens it
DESCRIPTION_MAX = 255
CODE_SHAPE = re.compile(r"^[A-Za-z0-9.\-]+$")


def _number(code: str | None) -> int | None:
    return int(code) if code and code.isdigit() else None


def next_code(rows: Iterable[tuple[str | None, str | None]], account_type: str) -> str:
    """The next free code for a type, from the company's active (code, type) rows: one past the
    type's highest number, else where the type starts, moved on past any code already used."""
    rows = list(rows)
    taken = {c for c, _ in rows if c}
    numbers = [n for c, t in rows if t == account_type and (n := _number(c)) is not None]
    number = max(numbers) + 1 if numbers else CODE_START.get(account_type, 100)
    while f"{number:03d}" in taken:
        number += 1
    return f"{number:03d}"


def check_fields(payload: dict) -> tuple[dict, dict]:
    """The cleaned code, name, type and description, and a Part 11 sentence for each field that
    breaks a rule of its own. A bank account may go without a code (Xero allows it)."""
    code = str(payload.get("code") or "").strip()
    name = re.sub(r"\s+", " ", str(payload.get("name") or "")).strip()
    account_type = payload.get("type")
    description = str(payload.get("description") or "").strip()
    fields: dict[str, str] = {}
    if account_type not in XERO_TYPE:
        fields["type"] = "Choose the account type to continue."
    if not code and account_type != "bank":
        fields["code"] = "The code is still empty. Add a code to continue."
    elif len(code) > CODE_MAX:
        fields["code"] = f"The code can be up to {CODE_MAX} characters. Shorten it to continue."
    elif code and not CODE_SHAPE.match(code):
        fields["code"] = (
            "The code can use letters, numbers, dots and dashes. Remove the other characters."
        )
    if not name:
        fields["name"] = "The name is still empty. Add a name to continue."
    elif len(name) > NAME_MAX:
        fields["name"] = f"The name can be up to {NAME_MAX} characters. Shorten it to continue."
    if len(description) > DESCRIPTION_MAX:
        fields["description"] = (
            f"The description can be up to {DESCRIPTION_MAX} characters. Shorten it to continue."
        )
    clean = {
        "code": code or None,
        "name": name,
        "type": account_type,
        "description": description or None,
    }
    return clean, fields


def clash_fields(code: str | None, code_holder: str | None, name_holder: str | None) -> dict:
    """The sentences for a code or a name another active account already has; the caller passes
    the holders' names (None = free)."""
    fields = {}
    if code and code_holder:
        fields["code"] = f"The code {code} is already used by {code_holder}. Choose another code."
    if name_holder:
        fields["name"] = f"Another account is already called {name_holder}. Choose another name."
    return fields
