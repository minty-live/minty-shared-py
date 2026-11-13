"""Minty's roles and permissions - the rules every backend applies, in one place (owner, 2026-11-12:
the five copies are one). Pure: a rank ladder and a rules table. Each service keeps only how it
finds a person's membership (its own database reads) and builds its gate on these.

RANK, NOT A TABLE. The ``role`` / ``permission`` / ``role_permission`` tables exist in the schema
but nothing reads them; who may do what is decided here, from the ladder and ``PERMISSION_RULES``.

SUPERUSER READ-ONLY. A system superuser reaches any company without a membership, but there only
the views in ``READONLY_ALLOWED_PERMISSIONS`` pass; every write is refused.

STRICT ROLE MANAGEMENT (owner, 2026-11-12). You manage only roles BELOW your own: an accountant
manages shop managers and cashiers, a shop manager cashiers. Admin is exempt (owner, 2026-11-13):
an admin invites, re-roles and removes admins too, never a super_admin.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    ENTITY_BASE = "entity_base"
    CASHIER = "cashier"
    SHOP_MANAGER = "shop_manager"
    ACCOUNTANT = "accountant"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class Permission(StrEnum):
    USER_INVITE = "user_invite"
    USER_VIEW_ALL = "user_view_all"
    USER_ROLE_ASSIGN = "user_role_assign"
    USER_ROLE_DELETE = "user_role_delete"
    ENTITY_CREATE = "entity_create"
    ENTITY_VIEW = "entity_view"
    ENTITY_UPDATE = "entity_update"
    ENTITY_RENAME = "entity_rename"
    ENTITY_DELETE = "entity_delete"
    MODULE_VIEW = "module_view"
    MODULE_MANAGE = "module_manage"
    SALES_METHOD_VIEW = "sales_method_view"
    SALES_METHOD_CREATE = "sales_method_create"
    SALES_METHOD_UPDATE = "sales_method_update"
    SALES_METHOD_DELETE = "sales_method_delete"
    SALES_METHOD_REORDER = "sales_method_reorder"
    COA_VIEW = "coa_view"
    COA_CREATE = "coa_create"
    COA_UPDATE = "coa_update"
    COA_DELETE = "coa_delete"
    XERO_SETTINGS_VIEW = "xero_settings_view"
    XERO_SETTINGS_UPDATE = "xero_settings_update"
    REPORT_VIEW_OWN = "report_view_own"
    REPORT_VIEW_ENTITY = "report_view_entity"
    REPORT_EDIT_OWN = "report_edit_own"
    REPORT_EDIT_ENTITY = "report_edit_entity"
    REPORT_DELETE_OWN = "report_delete_own"
    REPORT_DELETE_ENTITY = "report_delete_entity"
    REPORT_PUBLISH = "report_publish"
    CONTACT_CREATE = "contact_create"
    SUBSCRIPTION_VIEW = "subscription_view"


ROLE_RANK: dict[str, int] = {
    Role.ENTITY_BASE.value: 0,
    Role.CASHIER.value: 1,
    Role.SHOP_MANAGER.value: 2,
    Role.ACCOUNTANT.value: 3,
    Role.ADMIN.value: 4,
    Role.SUPER_ADMIN.value: 5,
}

#: Spellings seen in real rows. An unrecognised role normalises to ENTITY_BASE (rank 0), never to
#: a guess -- an unknown value must not be granted anything.
ROLE_ALIASES: dict[str, str] = {
    "user": Role.ENTITY_BASE.value,
    "no_role": Role.ENTITY_BASE.value,
    "none": Role.ENTITY_BASE.value,
    "entity base": Role.ENTITY_BASE.value,
    "entity-base": Role.ENTITY_BASE.value,
    "client": Role.CASHIER.value,
    "shop manager": Role.SHOP_MANAGER.value,
    "shop-manager": Role.SHOP_MANAGER.value,
    "super admin": Role.SUPER_ADMIN.value,
    "super-admin": Role.SUPER_ADMIN.value,
}


@dataclass(frozen=True)
class PermissionRule:
    min_role: Role
    entity_scoped: bool = True


PERMISSION_RULES: dict[Permission, PermissionRule] = {
    Permission.USER_INVITE: PermissionRule(Role.SHOP_MANAGER),
    Permission.USER_VIEW_ALL: PermissionRule(Role.SHOP_MANAGER),
    Permission.USER_ROLE_ASSIGN: PermissionRule(Role.SHOP_MANAGER),
    Permission.USER_ROLE_DELETE: PermissionRule(Role.ACCOUNTANT),
    Permission.ENTITY_CREATE: PermissionRule(Role.ENTITY_BASE, entity_scoped=False),
    Permission.ENTITY_VIEW: PermissionRule(Role.CASHIER),
    Permission.ENTITY_UPDATE: PermissionRule(Role.ACCOUNTANT),
    Permission.ENTITY_RENAME: PermissionRule(Role.ADMIN),
    Permission.ENTITY_DELETE: PermissionRule(Role.ADMIN),
    # Which modules a company has is information every member needs; changing them is admin's.
    Permission.MODULE_VIEW: PermissionRule(Role.CASHIER),
    Permission.MODULE_MANAGE: PermissionRule(Role.ADMIN),
    Permission.SALES_METHOD_VIEW: PermissionRule(Role.CASHIER),
    Permission.SALES_METHOD_CREATE: PermissionRule(Role.ACCOUNTANT),
    Permission.SALES_METHOD_UPDATE: PermissionRule(Role.ACCOUNTANT),
    Permission.SALES_METHOD_DELETE: PermissionRule(Role.ACCOUNTANT),
    Permission.SALES_METHOD_REORDER: PermissionRule(Role.ACCOUNTANT),
    Permission.COA_VIEW: PermissionRule(Role.CASHIER),
    Permission.COA_CREATE: PermissionRule(Role.ACCOUNTANT),
    Permission.COA_UPDATE: PermissionRule(Role.ACCOUNTANT),
    Permission.COA_DELETE: PermissionRule(Role.ACCOUNTANT),
    Permission.XERO_SETTINGS_VIEW: PermissionRule(Role.CASHIER),
    Permission.XERO_SETTINGS_UPDATE: PermissionRule(Role.ACCOUNTANT),
    Permission.REPORT_VIEW_OWN: PermissionRule(Role.CASHIER),
    Permission.REPORT_VIEW_ENTITY: PermissionRule(Role.CASHIER),
    Permission.REPORT_EDIT_OWN: PermissionRule(Role.CASHIER),
    Permission.REPORT_EDIT_ENTITY: PermissionRule(Role.SHOP_MANAGER),
    Permission.REPORT_DELETE_OWN: PermissionRule(Role.CASHIER),
    Permission.REPORT_DELETE_ENTITY: PermissionRule(Role.SHOP_MANAGER),
    Permission.REPORT_PUBLISH: PermissionRule(Role.ACCOUNTANT),
    # A cashier adds a supplier while filing an expense; Settings' contacts need COA_* (accountant).
    Permission.CONTACT_CREATE: PermissionRule(Role.CASHIER),
    # minty-subscription-api's subscription pages
    Permission.SUBSCRIPTION_VIEW: PermissionRule(Role.ACCOUNTANT),
}

#: What a superuser may do on a company they hold no membership on. Views only.
READONLY_ALLOWED_PERMISSIONS: frozenset[Permission] = frozenset(
    {
        Permission.USER_VIEW_ALL,
        Permission.ENTITY_VIEW,
        Permission.MODULE_VIEW,
        Permission.SALES_METHOD_VIEW,
        Permission.COA_VIEW,
        Permission.XERO_SETTINGS_VIEW,
        Permission.REPORT_VIEW_OWN,
        Permission.REPORT_VIEW_ENTITY,
    }
)


def normalize_role(role: Any) -> str:
    """The canonical role for a stored value; anything unknown is ENTITY_BASE."""
    if role is None:
        return Role.ENTITY_BASE.value
    role_text = str(role).strip().lower()
    if role_text in ROLE_ALIASES:
        return ROLE_ALIASES[role_text]
    if role_text in ROLE_RANK:
        return role_text
    return Role.ENTITY_BASE.value


def rank(role: Any) -> int:
    return ROLE_RANK.get(normalize_role(role), 0)


def role_at_least(role: Any, minimum_role: Any) -> bool:
    return rank(role) >= rank(minimum_role)


def role_allows(role: Any, permission: Permission) -> bool:
    """The rules table alone: does this role reach the permission's minimum? An unknown permission
    is refused (a typo must fail closed). Membership, approval and the superuser's read-only visit
    are the caller's to decide first."""
    rule = PERMISSION_RULES.get(permission)
    return bool(rule) and role_at_least(role, rule.min_role.value)


def can_manage_role_assignment(actor_role: Any, target_role: Any) -> bool:
    """Strict (owner, 2026-11-12): you act only on a role below your own, and only from shop
    manager up. Invite, cancel or resend an invite, change a role, remove a member.
    Admin is the exception (owner, 2026-11-13): an admin also manages admins, themselves included;
    the services' own guards keep the only admin and the payer. Never a super_admin."""
    if not role_at_least(actor_role, Role.SHOP_MANAGER.value):
        return False
    if normalize_role(actor_role) == Role.ADMIN.value:
        return rank(target_role) <= rank(Role.ADMIN.value)
    return rank(actor_role) > rank(target_role)


def assignable_roles(actor_role: Any) -> list[str]:
    """The roles an actor may give, highest first: those below their own (an admin's own too), from cashier up (a
    super_admin is never given; nor is entity_base, which is no role at all)."""
    offered = [Role.ADMIN, Role.ACCOUNTANT, Role.SHOP_MANAGER, Role.CASHIER]
    return [r.value for r in offered if can_manage_role_assignment(actor_role, r.value)]
