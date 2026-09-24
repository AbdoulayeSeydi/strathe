#!/usr/bin/env python3
"""Strathe launcher: `python strathe.py --help`. Works from any folder."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration"))

from strathe_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
