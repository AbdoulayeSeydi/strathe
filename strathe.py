#!/usr/bin/env python3
"""Strathe launcher: `python strathe.py --help`. Works from any folder."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration"))
# MLflow 3.x prints a hint meant for AI coding agents whenever it's imported; labs don't need it. Every step
# Strathe runs inherits this.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from strathe_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
