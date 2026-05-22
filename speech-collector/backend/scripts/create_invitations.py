import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.db import get_conn, init_db
from app.invitations import generate_invitation_code


def create_codes(count: int, dialect: str, label: str, max_uses: int, expires_at: str, note: str) -> list[str]:
    init_db()
    codes: list[str] = []
    with get_conn() as conn:
        for _ in range(count):
            code = generate_invitation_code(dialect)
            while conn.execute("SELECT code FROM invitations WHERE code = ?", (code,)).fetchone():
                code = generate_invitation_code(dialect)
            conn.execute(
                """
                INSERT INTO invitations(code, label, dialect_hint, max_uses, expires_at, note)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (code, label, dialect, max_uses, expires_at, note),
            )
            codes.append(code)
    return codes


def main() -> None:
    parser = argparse.ArgumentParser(description="Create invitation codes for speech collection.")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--dialect", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--max-uses", type=int, default=0)
    parser.add_argument("--expires-at", default="")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    codes = create_codes(args.count, args.dialect, args.label, args.max_uses, args.expires_at, args.note)
    print("code\tdialect_hint\tlabel\tmax_uses\texpires_at\tnote")
    for code in codes:
        print(f"{code}\t{args.dialect}\t{args.label}\t{args.max_uses}\t{args.expires_at}\t{args.note}")


if __name__ == "__main__":
    main()
