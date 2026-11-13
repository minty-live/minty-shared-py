"""minty_shared.policy: the rank ladder, the rules table and the strict role-management rule
(owner, 2026-11-12), lifted from the five services' copies."""

import pytest

from minty_shared.policy import (
    PERMISSION_RULES,
    READONLY_ALLOWED_PERMISSIONS,
    Permission,
    Role,
    assignable_roles,
    can_manage_role_assignment,
    normalize_role,
    role_allows,
    role_at_least,
)


@pytest.mark.parametrize(
    "stored, role",
    [
        (None, "entity_base"),
        ("Cashier", "cashier"),
        (" client ", "cashier"),
        ("shop manager", "shop_manager"),
        ("super-admin", "super_admin"),
        ("owner", "entity_base"),  # unknown: never a guess
    ],
)
def test_a_stored_role_normalises_and_an_unknown_one_gets_nothing(stored, role):
    assert normalize_role(stored) == role


def test_the_ladder():
    assert role_at_least("admin", "accountant") and not role_at_least("cashier", "shop_manager")
    assert role_at_least("super_admin", "super_admin")


def test_every_permission_has_a_rule_and_an_unknown_one_is_refused():
    assert set(PERMISSION_RULES) == set(Permission)
    assert role_allows("cashier", Permission.COA_VIEW) and not role_allows(
        "cashier", Permission.COA_CREATE
    )
    assert not role_allows("super_admin", "no_such_permission")  # type: ignore[arg-type]


def test_a_superusers_visit_reads_only():
    assert all(
        PERMISSION_RULES[p].min_role in (Role.CASHIER, Role.SHOP_MANAGER)
        for p in READONLY_ALLOWED_PERMISSIONS
    )


@pytest.mark.parametrize(
    "actor, target, allowed",
    [
        ("super_admin", "admin", True),
        ("admin", "admin", True),  # owner 2026-11-13: admin is exempt from the same-rank rule
        ("admin", "super_admin", False),
        ("Admin", "admin", True),
        ("admin", "accountant", True),
        ("accountant", "accountant", False),
        ("accountant", "shop_manager", True),
        ("shop_manager", "shop_manager", False),
        ("shop_manager", "cashier", True),
        ("cashier", "entity_base", False),  # below shop manager manages nobody
    ],
)
def test_you_manage_only_roles_below_yours_admin_excepted(actor, target, allowed):
    assert can_manage_role_assignment(actor, target) is allowed


def test_the_roles_an_actor_may_give():
    assert assignable_roles("super_admin") == ["admin", "accountant", "shop_manager", "cashier"]
    assert assignable_roles("admin") == ["admin", "accountant", "shop_manager", "cashier"]
    assert assignable_roles("accountant") == ["shop_manager", "cashier"]
    assert assignable_roles("shop_manager") == ["cashier"]
    assert assignable_roles("cashier") == []
