from __future__ import annotations

from collections.abc import Iterable

ROLE_ORG_OWNER = "org_owner"
ROLE_ORG_ADMIN = "org_admin"
ROLE_ORG_MEMBER = "org_member"

PERMISSION_PAPERS_READ = "papers:read"
PERMISSION_PAPERS_WRITE = "papers:write"
PERMISSION_PAPERS_DELETE = "papers:delete"
PERMISSION_JOBS_READ = "jobs:read"
PERMISSION_JOBS_MANAGE = "jobs:manage"
PERMISSION_ORG_SETTINGS_READ = "org_settings:read"
PERMISSION_ORG_SETTINGS_WRITE = "org_settings:write"

ROLE_PERMISSION_MAP: dict[str, frozenset[str]] = {
    ROLE_ORG_OWNER: frozenset(
        {
            PERMISSION_PAPERS_READ,
            PERMISSION_PAPERS_WRITE,
            PERMISSION_PAPERS_DELETE,
            PERMISSION_JOBS_READ,
            PERMISSION_JOBS_MANAGE,
            PERMISSION_ORG_SETTINGS_READ,
            PERMISSION_ORG_SETTINGS_WRITE,
        }
    ),
    ROLE_ORG_ADMIN: frozenset(
        {
            PERMISSION_PAPERS_READ,
            PERMISSION_PAPERS_WRITE,
            PERMISSION_PAPERS_DELETE,
            PERMISSION_JOBS_READ,
            PERMISSION_JOBS_MANAGE,
            PERMISSION_ORG_SETTINGS_READ,
            PERMISSION_ORG_SETTINGS_WRITE,
        }
    ),
    ROLE_ORG_MEMBER: frozenset(
        {
            PERMISSION_PAPERS_READ,
            PERMISSION_JOBS_READ,
            PERMISSION_ORG_SETTINGS_READ,
        }
    ),
}


def permissions_for_role(role: str | None) -> frozenset[str]:
    return ROLE_PERMISSION_MAP.get((role or "").strip(), ROLE_PERMISSION_MAP[ROLE_ORG_MEMBER])


def has_permission(role: str | None, permission: str) -> bool:
    return permission in permissions_for_role(role)


def list_permissions(role: str | None) -> list[str]:
    return sorted(permissions_for_role(role))


def require_known_role(role: str | None) -> str:
    normalized = (role or "").strip()
    if normalized in ROLE_PERMISSION_MAP:
        return normalized
    return ROLE_ORG_MEMBER


def has_all_permissions(role: str | None, permissions: Iterable[str]) -> bool:
    granted = permissions_for_role(role)
    return all(permission in granted for permission in permissions)
