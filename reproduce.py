"""Install the CPU environment and reproduce the released experiments."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, help="Python 3.12 executable for setup")
    parser.add_argument(
        "--current-env",
        action="store_true",
        help="Use the active environment without installing packages",
    )
    parser.add_argument(
        "--skip-ci", action="store_true", help="Skip the 10,000 paired-image bootstraps"
    )
    parser.add_argument("--output", type=Path, help="New output directory")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    root = Path(__file__).resolve().parent
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output = (args.output or root / "reproduction_results" / run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1"
    )
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    with (output / "launcher.log").open("w", encoding="utf-8") as log:

        def run(command, process_env=None):
            command = [str(c) for c in command]
            with subprocess.Popen(
                command,
                cwd=root,
                env=process_env or env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
            ) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                status = process.wait()
            if status:
                raise subprocess.CalledProcessError(status, command)

        try:
            python = Path(sys.executable)
            if not args.current_env:
                runtime = root / ".runtime"
                runtime.mkdir(exist_ok=True)
                requirements = root / "requirements-lock-cpu.txt"
                environment = runtime / "venv"
                python = environment / (
                    "Scripts/python.exe" if os.name == "nt" else "bin/python"
                )
                if not python.is_file():
                    candidate = args.python or (
                        Path(sys.executable)
                        if sys.version_info[:2] == (3, 12)
                        else None
                    )
                    if candidate:
                        candidate = candidate.resolve()
                        version = subprocess.check_output(
                            [
                                str(candidate),
                                "-c",
                                "import sys; print('%d.%d' % sys.version_info[:2])",
                            ],
                            text=True,
                            env=env,
                        ).strip()
                        if version != "3.12":
                            raise ValueError("--python must select Python 3.12")
                        run([candidate, "-m", "venv", environment])
                    else:
                        tools = runtime / "tools"
                        if not (tools / "uv/__main__.py").is_file():
                            run(
                                [
                                    sys.executable,
                                    "-m",
                                    "pip",
                                    "install",
                                    "--disable-pip-version-check",
                                    "--target",
                                    tools,
                                    "uv==0.10.7",
                                ]
                            )
                        uv_env = dict(
                            env,
                            PYTHONPATH=str(tools),
                            UV_PYTHON_INSTALL_DIR=str(runtime / "python"),
                            UV_CACHE_DIR=str(runtime / "cache"),
                        )
                        run(
                            [
                                sys.executable,
                                "-m",
                                "uv",
                                "venv",
                                "--seed",
                                "--python",
                                "3.12",
                                "--managed-python",
                                environment,
                            ],
                            uv_env,
                        )
                marker = environment / ".requirements-installed.json"
                expected = {
                    "sha256": hashlib.sha256(requirements.read_bytes()).hexdigest()
                }
                if (
                    not marker.exists()
                    or json.loads(marker.read_text(encoding="utf-8")) != expected
                ):
                    run(
                        [
                            python,
                            "-m",
                            "pip",
                            "install",
                            "--disable-pip-version-check",
                            "-r",
                            requirements,
                        ]
                    )
                    marker.write_text(json.dumps(expected), encoding="utf-8")
                run([python, "-m", "pip", "check"])
            command = [
                python,
                "scripts/reproduce_all.py",
                "--output",
                output / "metrics",
                "--threads",
                args.threads,
            ]
            if not args.skip_ci:
                command.append("--with_ci")
            run(command)
            run([python, "-m", "unittest", "discover", "-s", "tests", "-v"])
            (output / "status.json").write_text(
                json.dumps(
                    {"status": "passed", "confidence_intervals": not args.skip_ci}
                ),
                encoding="utf-8",
            )
            print(
                f"\nCompleted. Open {output / 'metrics/tables/reproduction_results.xlsx'}",
                flush=True,
            )
            return 0
        except Exception as error:
            (output / "status.json").write_text(
                json.dumps({"status": "failed", "error": str(error)}), encoding="utf-8"
            )
            print(
                f"\nFailed: {error}\nDetails: {output / 'launcher.log'}",
                file=sys.stderr,
            )
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
