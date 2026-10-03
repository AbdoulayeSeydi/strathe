#!/usr/bin/env python3
"""Strathe launcher: `python strathe.py --help`. Works from any folder."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration"))
# MLflow 3.x prints a hint meant for AI coding agents whenever it's imported; labs don't need it. Every step
# Strathe runs inherits this.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# Strathe prints a few symbols (✅ ❌ ℹ️). On Windows, output sent to a file or pipe (the web page's job logs, CI)
# defaults to cp1252, which can't encode them and crashes the run. UTF-8 for this process and every step it runs.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from strathe_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
