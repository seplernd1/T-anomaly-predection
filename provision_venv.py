#!/usr/bin/env python
"""Self-healing environment bootstrap for the nightly automation.

Ensures ``.venv`` exists, has pip, and can import every package pinned in
requirements.txt — repairing whatever is missing before the nightly jobs
run. Idempotent: a healthy environment costs one import probe (~1 s).

Why this exists
---------------
The scheduled entry points (automate_harvest.bat, run_current_state_snapshot.bat)
hard-require ``.venv\\Scripts\\python.exe``, but that venv was created without
pip and never had its dependencies installed — three separate nightly failures
(requests, papermill, pandas all missing) traced back to this. Instead of
assuming the environment is correct, every nightly run now repairs it from
the committed dependency manifest.

Usage::

    python provision_venv.py            # create/repair .venv as needed
    python provision_venv.py --check    # exit 0 if healthy, 1 + list if not
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
# Windows layout for the scheduled-task environment this project targets.
VENV_PY = VENV_DIR / "Scripts" / "python.exe"
REQUIREMENTS = ROOT / "requirements.txt"

#: requirements.txt name -> importable module name (they differ for several pins).
IMPORT_NAMES = {
    "requests": "requests",
    "pandas": "pandas",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "sqlalchemy": "sqlalchemy",
    "psycopg2-binary": "psycopg2",
    "python-dotenv": "dotenv",
    "tqdm": "tqdm",
    "openpyxl": "openpyxl",
    "papermill": "papermill",
    "ipykernel": "ipykernel",
    "pytest": "pytest",
}


def requirements_packages() -> list[str]:
    """Package names pinned in requirements.txt (comments stripped)."""
    out: list[str] = []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.split("==")[0].split(">=")[0].strip().lower())
    return out


def missing_packages(python_exe: Path | str) -> list[str]:
    """Requirements names whose import probe fails under the given interpreter."""
    missing: list[str] = []
    for req_name, mod in IMPORT_NAMES.items():
        probe = subprocess.run(
            [str(python_exe), "-c", f"import {mod}"],
            capture_output=True,
        )
        if probe.returncode != 0:
            missing.append(req_name)
    return missing


def ensure_pip(python_exe: Path) -> None:
    probe = subprocess.run([str(python_exe), "-m", "pip", "--version"], capture_output=True)
    if probe.returncode != 0:
        print("[provision] pip missing -> running ensurepip")
        subprocess.run([str(python_exe), "-m", "ensurepip", "--upgrade"], check=True)


def install_requirements(python_exe: Path, packages: list[str]) -> None:
    print(f"[provision] installing {len(packages)} missing package(s): {', '.join(packages)}")
    subprocess.run(
        [str(python_exe), "-m", "pip", "install", "-q", "--disable-pip-version-check",
         "-r", str(REQUIREMENTS)],
        check=True,
    )


def provision() -> tuple[bool, list[str]]:
    """Bring .venv to a healthy state. Returns (healthy, actions_taken)."""
    actions: list[str] = []
    if not VENV_PY.is_file():
        print(f"[provision] .venv missing -> creating with {sys.executable}")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        actions.append("created .venv")
    ensure_pip(VENV_PY)

    missing = missing_packages(VENV_PY)
    if missing:
        install_requirements(VENV_PY, missing)
        actions.append(f"installed {len(missing)} package(s)")
        still = missing_packages(VENV_PY)
        if still:
            print(f"[provision] FAILED: still missing after install: {', '.join(still)}")
            return False, actions
    return True, actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="Only report health; exit 1 with the missing list if not healthy.")
    args = parser.parse_args(argv)

    if args.check:
        if not VENV_PY.is_file():
            print(".venv missing")
            return 1
        missing = missing_packages(VENV_PY)
        if missing:
            print("missing: " + ", ".join(missing))
            return 1
        print("venv OK")
        return 0

    try:
        healthy, actions = provision()
    except subprocess.CalledProcessError as exc:
        print(f"[provision] FAILED: {exc}")
        return 1
    if healthy:
        tail = f" ({'; '.join(actions)})" if actions else ""
        print(f"[provision] venv OK{tail}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
