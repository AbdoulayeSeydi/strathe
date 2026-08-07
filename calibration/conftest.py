# Standard pytest mechanism: pytest automatically adds the directory
# containing conftest.py to sys.path, so tests/ can import dynamics_model
# and other calibration/ modules without needing PYTHONPATH set manually.