"""STRATHE_REQUIRE_ENGINES=1: a test skipped because a simulator isn't here fails instead, so a machine that should run them all can't pass by
skipping them (PROJECT_STATUS.md #97). Left unset, as on a lab's laptop, they skip."""
import os
import re

import pytest

ENGINE_WORDS = re.compile(r"replay tool|docker image|gazebo|stonefish|holoocean|mujoco", re.I)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if os.environ.get("STRATHE_REQUIRE_ENGINES") and report.skipped and ENGINE_WORDS.search(str(report.longrepr)):
        report.outcome = "failed"
        report.longrepr = f"STRATHE_REQUIRE_ENGINES: skipped for a missing simulator: {report.longrepr}"
