#!/usr/bin/env python3
"""
test_f3_4_policy_fixtures.py -- F3.4 three-fixture policy interface test.

Fixture 1: A real, valid PPO model passes validate_policy_compatibility.

Fixture 2: A dummy model with deliberately wrong observation shape fails
           at pre-flight with the specific, correct named-mismatch message --
           not a raw exception somewhere inside env.step().

Fixture 3: A model trained entirely outside the shipped pretrained-model
           workflow (formation_hold_ppo_my_run3.zip, produced by gate2 in
           this session) passes through run_fleet_experiment.py with zero
           source modifications. The ONLY change is a YAML config file.
           This is the concrete proof that "Mera owns the environment and
           experiment; the researcher owns the policy."

Run from calibration/:
    python3 test_f3_4_policy_fixtures.py

F3 done when: all three pass, and the interface contract doc states plainly
what's proven vs. supported-but-unvalidated vs. explicitly out of scope.
"""
import os
import sys
import subprocess
import tempfile

import numpy as np
import yaml

from validate_policy_compatibility import (
    validate_policy_compatibility,
    validate_all_policies,
    PolicyIncompatibleError,
    POLICY_INTERFACE_CONTRACT,
    PROVEN_ALGORITHMS,
    SUPPORTED_UNVALIDATED_ALGORITHMS,
)

passed = failed = 0


def _run(test_fn):
    global passed, failed
    try:
        test_fn()
        print(f"PASS  {test_fn.__name__}")
        passed += 1
    except AssertionError as e:
        print(f"FAIL  {test_fn.__name__}: {e}")
        failed += 1
    except Exception as e:
        print(f"ERROR {test_fn.__name__}: {type(e).__name__}: {e}")
        failed += 1


# ---------------------------------------------------------------------------
# FIXTURE 1: real valid PPO model passes
# ---------------------------------------------------------------------------

def test_fixture1_valid_model_passes():
    """A real, valid PPO model (one of the pre-shipped curriculum policies)
    passes validate_policy_compatibility cleanly."""
    from stable_baselines3 import PPO

    model = PPO.load("formation_hold_ppo_curriculum_v5")
    result = validate_policy_compatibility(model,
                                           model_path="formation_hold_ppo_curriculum_v5")
    assert result is True, f"expected True from valid model, got {result!r}"


def test_fixture1_contract_states_ppo_proven():
    """The interface contract must explicitly state PPO is proven and
    SAC/TD3 are supported-but-unvalidated -- not implied as equivalent."""
    assert "PPO: PROVEN" in POLICY_INTERFACE_CONTRACT, \
        "contract does not state PPO as proven"
    assert "SAC: SUPPORTED" in POLICY_INTERFACE_CONTRACT and \
           "UNVALIDATED" in POLICY_INTERFACE_CONTRACT, \
        "contract does not explicitly name SAC as unvalidated"
    assert "TD3: SUPPORTED" in POLICY_INTERFACE_CONTRACT, \
        "contract does not mention TD3"
    assert "ppo" in PROVEN_ALGORITHMS, \
        f"ppo not in PROVEN_ALGORITHMS: {PROVEN_ALGORITHMS}"
    assert "sac" in SUPPORTED_UNVALIDATED_ALGORITHMS, \
        f"sac not in SUPPORTED_UNVALIDATED_ALGORITHMS: {SUPPORTED_UNVALIDATED_ALGORITHMS}"


def test_fixture1_contract_names_unsolved_boundary():
    """The contract must explicitly name what it does NOT verify, as a
    named unsolved boundary -- not leave it implicit."""
    required = ["recurrent", "normalization", "ordering", "SCOPE BOUNDARY"]
    for term in required:
        assert term in POLICY_INTERFACE_CONTRACT or term.lower() in POLICY_INTERFACE_CONTRACT.lower(), \
            f"contract missing required term {term!r}"


# ---------------------------------------------------------------------------
# FIXTURE 2: wrong-shape model fails at pre-flight with the correct message
# ---------------------------------------------------------------------------

def test_fixture2_single_agent_obs_rejected():
    """Box(11,) -- the single-agent Arena obs shape -- must be rejected
    with a message that names 'single-agent', both shapes, AND fires
    before any episode runs (not inside env.step())."""

    class SingleAgentFakeModel:
        class _Space:
            def __init__(self, shape):
                self.shape = shape
                self.low = np.full(shape, -1.0)
                self.high = np.full(shape, 1.0)
        observation_space = _Space((11,))   # single-agent -- wrong
        action_space = _Space((12,))        # fleet action -- coincidentally right

    try:
        validate_policy_compatibility(SingleAgentFakeModel(),
                                      model_path="single_agent_policy.zip")
        raise AssertionError("should have raised PolicyIncompatibleError")
    except PolicyIncompatibleError as e:
        msg = str(e)
        assert "(11,)" in msg, f"missing model shape in message: {msg[:200]}"
        assert "(24,)" in msg, f"missing expected shape in message: {msg[:200]}"
        assert "single-agent" in msg.lower(), \
            f"missing single-agent hint in message: {msg[:200]}"


def test_fixture2_wrong_action_shape_rejected():
    """Wrong action shape must also be caught with a specific message."""

    class WrongActionModel:
        class _Space:
            def __init__(self, shape):
                self.shape = shape
                self.low = np.full(shape, -1.0)
                self.high = np.full(shape, 1.0)
        observation_space = _Space((24,))   # correct obs
        action_space = _Space((6,))         # single-agent action -- wrong

    try:
        validate_policy_compatibility(WrongActionModel(),
                                      model_path="wrong_action.zip")
        raise AssertionError("should have raised PolicyIncompatibleError")
    except PolicyIncompatibleError as e:
        msg = str(e)
        assert "(6,)" in msg, f"missing model action shape in message: {msg[:200]}"
        assert "(12,)" in msg, f"missing expected action shape in message: {msg[:200]}"


def test_fixture2_check_fires_before_any_episode():
    """Confirm the check fires before env.step() is ever called -- the
    whole point of pre-flight. A CountingEnv wraps step() to record calls;
    it must remain at zero after a failed validation."""

    episodes_stepped = [0]

    class CountingEnv:
        class _Space:
            def __init__(self, shape):
                self.shape = shape
                self.low = np.full(shape, -1.0)
                self.high = np.full(shape, 1.0)
        observation_space = _Space((24,))
        action_space = _Space((12,))
        def step(self, action):
            episodes_stepped[0] += 1
            return None, 0.0, True, False, {}

    class WrongShapeModel:
        class _Space:
            def __init__(self, shape):
                self.shape = shape
                self.low = np.full(shape, -1.0)
                self.high = np.full(shape, 1.0)
        observation_space = _Space((11,))
        action_space = _Space((12,))

    try:
        validate_policy_compatibility(WrongShapeModel(), env=CountingEnv())
        raise AssertionError("should have raised")
    except PolicyIncompatibleError:
        pass

    assert episodes_stepped[0] == 0, \
        f"env.step() was called {episodes_stepped[0]} time(s) before validation error -- " \
        f"pre-flight check did not fire first"


# ---------------------------------------------------------------------------
# FIXTURE 3: external model, zero source modifications
# ---------------------------------------------------------------------------

def test_fixture3_external_model_zero_source_mods():
    """
    THE KEY TEST: a model trained entirely outside the shipped pretrained-
    model workflow passes through run_fleet_experiment.py with zero source
    modifications.

    formation_hold_ppo_my_run3.zip was produced in this session by:
        python train_formation_baseline.py --mode gate2 \\
            --total-timesteps 20000 --run-name my_run3

    It is NOT one of the four pre-shipped curriculum models. It was trained
    with different hyperparameters, in a separate run, by the developer
    acting as a researcher would.

    The only input to Mera below is:
        1. The .zip model file (externally produced)
        2. A YAML config (no source edits)

    Proof: run_fleet_experiment.py completes successfully with return code 0.
    """
    external_model = "formation_hold_ppo_my_run3"
    if not os.path.exists(f"{external_model}.zip"):
        raise AssertionError(
            f"{external_model}.zip not found. Produce it first:\n"
            f"  python train_formation_baseline.py --mode gate2 "
            f"--total-timesteps 20000 --run-name my_run3\n"
            f"then re-run this test."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "results")

        # The ONLY change a researcher makes: a YAML config file.
        # No edits to any source file in the Mera repo.
        config = {
            "name": "f3_4_external_model_proof",
            "vehicles": [
                {"model_path": external_model, "algorithm": "ppo",
                 "role": "externally_trained_1"},
                {"model_path": external_model, "algorithm": "ppo",
                 "role": "externally_trained_2"},
            ],
            "communication": {"severity": "L0"},
            "task": {
                "target_separation_m": 2.0,
                "tolerance_m": 0.2,
                "hold_steps": 20,
            },
            "evaluation": {
                "episodes": 8,
                "comm_seeds": [1],      # 1 seed -- keeps the test fast (~15s)
                "grid_seeds": 1,        # 1 seed per grid point
            },
            "dynamics_backend": "stand_in",
            "output_dir": output_dir,
        }

        config_path = os.path.join(tmpdir, "external_model_test.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Zero source modifications -- only the config path is different.
        result = subprocess.run(
            [sys.executable, "run_fleet_experiment.py", "--config", config_path],
            capture_output=True, text=True, timeout=300,
        )

        combined = result.stdout + result.stderr

        assert result.returncode == 0, (
            f"run_fleet_experiment.py exited with code {result.returncode}.\n"
            f"Output:\n{combined[-1500:]}"
        )

        # Pre-flight check must have run (not silently skipped)
        assert ("[2/4]" in combined or "pre-flight" in combined.lower()), \
            "pre-flight step not visible in output -- was it skipped?"

        # Sweep must have completed
        assert "EXPERIMENT COMPLETE" in combined, \
            f"sweep did not complete.\nLast 500 chars:\n{combined[-500:]}"

        # Report must exist on disk
        report = os.path.join(output_dir, "summary_report.md")
        assert os.path.exists(report), \
            f"summary_report.md not found at {report}"

        # Report must reference the external model name
        with open(report) as f:
            report_text = f.read()
        assert external_model in report_text, \
            f"external model name not in report -- was the right model used?"


# ---------------------------------------------------------------------------
# RUN ALL
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_fixture1_valid_model_passes,
        test_fixture1_contract_states_ppo_proven,
        test_fixture1_contract_names_unsolved_boundary,
        test_fixture2_single_agent_obs_rejected,
        test_fixture2_wrong_action_shape_rejected,
        test_fixture2_check_fires_before_any_episode,
        test_fixture3_external_model_zero_source_mods,
    ]

    print("Running F3.4 three-fixture policy interface test...\n")
    print("Note: fixture 3 runs run_fleet_experiment.py as a subprocess.")
    print("Expected total runtime: ~20-30 seconds.\n")

    for t in tests:
        _run(t)

    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        raise SystemExit(1)
