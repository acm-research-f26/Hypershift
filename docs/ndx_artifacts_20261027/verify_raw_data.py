"""Verify the raw-data publication using only the standard library."""

import csv
import hashlib
from pathlib import Path, PurePosixPath


def main():
    root = Path(__file__).resolve().parents[2]
    manifest = Path(__file__).with_name("raw_data_manifest.csv")
    failures = []
    count = 0
    total_bytes = 0
    with manifest.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            relative = PurePosixPath(row["path"])
            path = root.joinpath(*relative.parts)
            if relative.is_absolute() or not path.resolve().is_relative_to(root):
                failures.append(f"Invalid manifest path: {row['path']}")
                continue
            if not path.is_file():
                failures.append(f"Missing: {relative}")
                continue
            digest = hashlib.sha256()
            with path.open("rb") as data:
                for block in iter(lambda: data.read(1024 * 1024), b""):
                    digest.update(block)
            size = path.stat().st_size
            if size != int(row["bytes"]) or digest.hexdigest() != row["sha256"]:
                failures.append(f"Changed: {relative}")
            count += 1
            total_bytes += size
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"Verified {count:,} files ({total_bytes:,} bytes); all SHA-256 checks passed.")


if __name__ == "__main__":
    main()
