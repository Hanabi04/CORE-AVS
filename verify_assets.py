"""Verify immutable distributed assets against their SHA-256 inventory."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    inventory = json.loads((root / "ASSET_SHA256.json").read_text())
    for name, digest in inventory.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or unsafe asset: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Hash mismatch: {name}")
    print(f"Verified {len(inventory)} assets")
    file_inventory = root / "FILE_SHA256.json"
    if file_inventory.exists():
        files = json.loads(file_inventory.read_text(encoding="utf-8"))
        for name, digest in files.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing or unsafe release file: {name}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Release file hash mismatch: {name}")
        print(f"Verified {len(files)} release files")


if __name__ == "__main__":
    main()
