"""Compare newly computed metrics with the distributed reference reports."""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    "main_tables.json": "main_tables.json",
    "main_three_runs.json": "main_three_runs.json",
    "compact_three_runs.json": "compact_three_runs.json",
    "extended/metrics.json": "extended.json",
}


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, f"{prefix}/{key}" if prefix else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten(child, f"{prefix}/{index}")
    elif isinstance(value, (int, float)):
        yield prefix, value


def compare_results(directory, expected_root=None):
    directory = Path(directory)
    expected_root = Path(expected_root) if expected_root else ROOT / "reported_results"
    rows = []
    for filename, reference in REPORTS.items():
        expected = dict(
            flatten(
                json.loads((expected_root / reference).read_text(encoding="utf-8"))[
                    "results"
                ]
            )
        )
        path = directory / filename
        observed = (
            dict(flatten(json.loads(path.read_text(encoding="utf-8"))["results"]))
            if path.exists()
            else {}
        )
        if filename == "main_tables.json" and any(
            "95% image-bootstrap CI" in k for k in observed
        ):
            bootstrap = json.loads(
                (expected_root / "bootstrap.json").read_text(encoding="utf-8")
            )["results"]
            expected.update(
                {k: v for k, v in flatten(bootstrap) if "95% image-bootstrap CI" in k}
            )
        for key in sorted(expected.keys() | observed.keys()):
            target, actual = expected.get(key), observed.get(key)
            status = (
                "MISSING"
                if actual is None
                else "UNEXPECTED"
                if target is None
                else "PASS"
                if math.isfinite(actual) and f"{actual:.3f}" == f"{target:.3f}"
                else "DIFFERENT"
            )
            rows.append(
                {
                    "report": filename,
                    "record": key,
                    "expected": target,
                    "observed": actual,
                    "difference": actual - target
                    if actual is not None and target is not None
                    else None,
                    "status": status,
                }
            )
    result = {
        "criterion": "Equal after rounding to three decimals",
        "passed": sum(r["status"] == "PASS" for r in rows),
        "total": len(rows),
        "results": rows,
    }
    (directory / "comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"Reference comparison: {result['passed']}/{result['total']} values match at three decimals"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    result = compare_results(parser.parse_args().directory)
    raise SystemExit(0 if result["passed"] == result["total"] else 1)
