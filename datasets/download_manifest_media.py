"""Restore authorized prepared assets from an explicit URL map, with hashes."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--sources",
        type=Path,
        required=True,
        help="JSON list: uid, asset, url, sha256, relative_path",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    entries = {
        e["uid"]: e
        for e in json.loads(a.manifest.read_text(encoding="utf-8"))["entries"]
    }
    sources = json.loads(a.sources.read_text(encoding="utf-8"))
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for row in sources:
        entry = entries[row["uid"]]
        expected = row["sha256"]
        fingerprints = entry["fingerprints"]
        if expected not in fingerprints.values():
            raise ValueError(
                f"Hash absent from manifest: {row['uid']} / {row['asset']}"
            )
        target = (root / row["relative_path"]).resolve()
        if not target.is_relative_to(root) or target == root:
            raise ValueError("Output path escapes destination")
        if target.exists():
            raise FileExistsError(target)
        url = row["url"]
        scheme = urllib.parse.urlsplit(url).scheme.lower()
        if scheme not in ("https", "file"):
            raise ValueError("Use an authorized HTTPS URL or file URI")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=target.parent, prefix=".download-")
        try:
            with (
                os.fdopen(fd, "wb") as out,
                urllib.request.urlopen(url, timeout=120) as response,
            ):
                shutil.copyfileobj(response, out)
            digest = hashlib.sha256(Path(temp).read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(f"Hash mismatch: {row['uid']} / {row['asset']}")
            with target.open("xb") as out, open(temp, "rb") as src:
                shutil.copyfileobj(src, out)
        finally:
            Path(temp).unlink(missing_ok=True)
        print(f"Verified {row['uid']} / {row['asset']}")


if __name__ == "__main__":
    main()
