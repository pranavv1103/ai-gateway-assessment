"""Bearer-token authentication and role resolution."""

import hashlib
import hmac
import re

_BEARER_RE = re.compile(r"^Bearer ([\x21-\x7e]{1,512})$", re.IGNORECASE)


def extract_bearer_token(authorization_header: str | None) -> str | None:
    if not authorization_header:
        return None
    match = _BEARER_RE.match(authorization_header.strip())
    return match.group(1) if match else None


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


class TokenRoleResolver:
    """Resolves a presented token to a server-side role using constant-time comparison."""

    def __init__(self, token_roles: dict[str, str]):
        self._entries = [(_digest(token), role) for token, role in token_roles.items()]

    def resolve(self, token: str) -> str | None:
        presented = _digest(token)
        matched_role: str | None = None
        for digest, role in self._entries:  # no early exit: timing is independent of position
            if hmac.compare_digest(digest, presented):
                matched_role = role
        return matched_role
