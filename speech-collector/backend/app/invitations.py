import secrets
import string
from datetime import datetime, timezone


CODE_ALPHABET = string.ascii_uppercase.replace("O", "").replace("I", "") + "23456789"


def generate_invitation_code(dialect_hint: str = "") -> str:
    prefix = {"ruian": "RA", "wenzhou": "WZ"}.get(dialect_hint, "SC")
    first = "".join(secrets.choice(CODE_ALPHABET) for _ in range(4))
    second = "".join(secrets.choice(CODE_ALPHABET) for _ in range(4))
    return f"{prefix}-{first}-{second}"


def invitation_is_usable(row) -> bool:
    if not row or not row["active"]:
        return False
    if row["max_uses"] and row["used_count"] >= row["max_uses"]:
        return False
    if row["expires_at"]:
        try:
            expires = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
        except ValueError:
            return False
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires <= datetime.now(timezone.utc):
            return False
    return True
