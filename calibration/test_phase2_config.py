#!/usr/bin/env python3
"""
test_phase2_config.py -- verifies C0.4's additions to phase2_config.py
(Claim.C_MULTIAGENT_COMM_BEHAVIOR / c_multiagent_comm_behavior), AND
that everything else in the file survived being fully rewritten
unchanged -- since this file didn't previously exist in this session's
own sandbox, adding CLAIM_C meant retyping the WHOLE file (PHASE1_HELDOUT_
DATES, check_result_type, output_dir_for, CLAIM_A/B included) rather than
a small, targeted edit, which is exactly the kind of change most likely
to introduce a silent transcription error somewhere unrelated to the
actual intended change.

Usage:
    python3 test_phase2_config.py
"""
import phase2_config as cfg
from phase2_config import Claim


def test_phase1_dates_unchanged():
    assert cfg.PHASE1_HELDOUT_DATES == ["16-02-2024", "19-02-2024"]
    assert cfg.PHASE1_TRAIN_DATES == ["12-01-2024", "14-02-2024"]


def test_backend_constants_unchanged():
    assert cfg.TRAINING_BACKEND == "stand_in"
    assert cfg.VALIDATION_BACKEND == "mujoco"
    assert cfg.INDEPENDENT_CROSSCHECK_BACKEND == "holoocean"


def test_claim_a_unchanged():
    assert Claim.A_OPEN_LOOP_REPLAY_ACCURACY == (
        "CLAIM_A: open-loop replay RMSE vs. real data — the rigorous, "
        "direct MVP acceptance criterion"
    )


def test_claim_b_constant_unchanged():
    assert Claim.B_TRAINED_POLICY_BEHAVIOR == (
        "CLAIM_B: trained PPO policy behavior under calibrated vs. default "
        "physics — complementary, not a second accuracy proof (no live "
        "hardware yet — see Phase 2.5)"
    )


def test_claim_b_default_matches_constant():
    """The invariant B's own pattern relies on: calling the parameterized
    method with its default argument must reproduce the hardcoded
    constant exactly -- if these ever drifted apart, callers using the
    bare constant and callers using the method would silently disagree."""
    assert Claim.b_trained_policy_behavior() == Claim.B_TRAINED_POLICY_BEHAVIOR


def test_claim_b_substitutes_algorithm():
    result = Claim.b_trained_policy_behavior(algorithm="sac")
    assert "SAC" in result
    assert "PPO" not in result


def test_check_result_type_still_enforces_final_requires_metrics():
    raised = False
    try:
        cfg.check_result_type("final", metrics_path=None)
    except ValueError:
        raised = True
    assert raised, "check_result_type('final', metrics_path=None) should raise"


def test_check_result_type_rejects_bad_value():
    raised = False
    try:
        cfg.check_result_type("nonsense")
    except ValueError:
        raised = True
    assert raised, "check_result_type should reject an illegal result_type"


def test_output_dir_for_unchanged():
    assert cfg.output_dir_for("calibration_output", "final") == "calibration_output_final"
    assert cfg.output_dir_for("calibration_output", "placeholder") == "calibration_output_placeholder"


# --- C0.4's actual new additions ---

def test_claim_c_exists_and_says_claim_c():
    assert Claim.C_MULTIAGENT_COMM_BEHAVIOR.startswith("CLAIM_C:")


def test_claim_c_default_matches_constant():
    """Same invariant as test_claim_b_default_matches_constant, applied
    to the new addition -- the default (validation_backend="mujoco")
    must reproduce the hardcoded constant exactly."""
    assert Claim.c_multiagent_comm_behavior() == Claim.C_MULTIAGENT_COMM_BEHAVIOR


def test_claim_c_substitutes_backend_in_both_places():
    """The claim text mentions the backend TWICE ('under X-validated
    physics' and 'not X-based training') -- confirm the substitution
    actually reached both occurrences, not just the first. Expects
    "HoloOcean" (this project's own stylized capitalization), NOT
    "HOLOOCEAN" -- catches the exact .upper() bug found and fixed here."""
    result = Claim.c_multiagent_comm_behavior(validation_backend="holoocean")
    assert result.count("HoloOcean") == 2, (
        f"expected 'HoloOcean' to appear exactly twice, found {result.count('HoloOcean')}: {result}"
    )
    assert "HOLOOCEAN" not in result, "backend name should be 'HoloOcean', not all-caps"
    assert "MuJoCo" not in result, "default backend name leaked into a non-default call"


def test_claim_c_states_complementary_not_third_proof():
    """The specific framing requested: complementary to A and B, not a
    third accuracy proof -- confirmed present in the actual string, not
    just asserted in a docstring somewhere."""
    text = Claim.C_MULTIAGENT_COMM_BEHAVIOR
    assert "complementary to A and B" in text
    assert "not a third accuracy proof" in text


def test_all_three_claims_are_distinct():
    values = [Claim.A_OPEN_LOOP_REPLAY_ACCURACY, Claim.B_TRAINED_POLICY_BEHAVIOR,
              Claim.C_MULTIAGENT_COMM_BEHAVIOR]
    assert len(set(values)) == 3, "two or more claim constants are identical -- likely a copy-paste error"


def test_claim_c_disclaims_mujoco_training():
    """Explicitly guards the honest caveat this claim depends on: it
    must NOT be readable as a claim about MuJoCo-based training, since
    no such training was ever run (Decision 2's training-backend rule)."""
    assert "not MuJoCo-based training" in Claim.C_MULTIAGENT_COMM_BEHAVIOR


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        raise SystemExit(1)