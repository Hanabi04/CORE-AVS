"""Recompute the main tables and three-run curriculum summaries."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

from export_results import export_results
from check_results import compare_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--with_ci", action="store_true", help="Add 10,000 image bootstraps"
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    root = Path(__file__).resolve().parents[1]
    output = (
        args.output
        or root / "runs" / datetime.now().strftime("reproduction_%Y%m%d_%H%M%S")
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    common = ["--device", args.device, "--threads", str(args.threads)]
    commands = [
        ("assets", ["verify_assets.py"]),
        (
            "main_tables",
            [
                "eval.py",
                "--datasets",
                "controlled",
                "core_a",
                "core_b",
                "core_c",
                "core",
                "--baselines",
                "--output",
                str(output / "main_tables.json"),
                *common,
                *(["--bootstrap", "10000"] if args.with_ci else []),
            ],
        ),
        (
            "main_three_runs",
            [
                "eval.py",
                "--eval_seeds",
                "--output",
                str(output / "main_three_runs.json"),
                *common,
            ],
        ),
        (
            "compact_three_runs",
            [
                "eval.py",
                "--eval_seeds",
                "--ablation_16ep",
                "--output",
                str(output / "compact_three_runs.json"),
                *common,
            ],
        ),
        (
            "architecture_transfer",
            ["reproduce_extended.py", "--output", str(output / "extended"), *common],
        ),
    ]
    started = time.perf_counter()
    for name, command in commands:
        print(f"Running {name} ...", flush=True)
        result = subprocess.run(
            [sys.executable, *command],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        (output / f"{name}.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        print(result.stdout, end="")
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(result.returncode)
    timing = {
        "wall_seconds_including_child_imports": time.perf_counter() - started,
        "device": args.device,
        "threads": args.threads,
        "bootstrap": args.with_ci,
    }
    (output / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    comparison = compare_results(output)
    export_results(output)
    if comparison["passed"] != comparison["total"]:
        raise SystemExit(
            "Some metrics differ. Inspect tables/08_reference_comparison.csv."
        )
    print(
        f"Completed in {timing['wall_seconds_including_child_imports']:.2f} s; reports: {output}"
    )


if __name__ == "__main__":
    main()
