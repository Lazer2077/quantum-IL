from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a project status update.")
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    path = Path("PROJECT_STATUS.md")
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    entry = f"\n- {timestamp}: {args.message}\n"
    if not path.exists():
        path.write_text("# Project Status\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    print(f"updated {path}")


if __name__ == "__main__":
    main()
