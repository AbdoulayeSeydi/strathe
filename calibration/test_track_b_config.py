"""
Tests for track_b_config.py -- B0.7's decision record. Ensures the
chosen approach stays exactly what B0.5/B0.6's evidence supported, and
that the enforcement function actually enforces it, not just documents
it.
"""
from track_b_config import (MUJOCO_DYNAMICS_INTEGRATION_APPROACH,
                             MUJOCO_DYNAMICS_INTEGRATION_APPROACH_REJECTED,
                             assert_uses_full_override)


def test_chosen_approach_is_full_override():
    """B0.7's actual decision, pinned -- if this ever changes, it must be
    a deliberate edit to track_b_config.py's own Decision 1, not a typo
    or an accidental import of the wrong constant elsewhere."""
    assert MUJOCO_DYNAMICS_INTEGRATION_APPROACH == "full_override"


def test_rejected_approach_is_named_and_distinct():
    """The rejected alternative is tracked explicitly, and must never
    silently collide with the chosen one."""
    assert MUJOCO_DYNAMICS_INTEGRATION_APPROACH_REJECTED == "native_rigid_body"
    assert MUJOCO_DYNAMICS_INTEGRATION_APPROACH != MUJOCO_DYNAMICS_INTEGRATION_APPROACH_REJECTED


def test_assert_uses_full_override_passes_for_the_chosen_approach():
    """Should not raise when passed the correct, currently-decided approach."""
    assert_uses_full_override(MUJOCO_DYNAMICS_INTEGRATION_APPROACH)


def test_assert_uses_full_override_rejects_the_rejected_approach():
    """The actual enforcement, mirroring assert_uses_phase1_split's own
    fail-loudly pattern -- mujoco_backend.py (B1.1) calling this with
    'native_rigid_body' must be caught here, not discovered downstream."""
    try:
        assert_uses_full_override(MUJOCO_DYNAMICS_INTEGRATION_APPROACH_REJECTED)
        raised = False
    except ValueError:
        raised = True
    assert raised, "expected ValueError when passing the rejected approach"


def test_assert_uses_full_override_rejects_arbitrary_garbage():
    """Not just the one known rejected value -- any string that isn't the
    chosen approach must be refused, including typos or placeholder
    values a future caller might pass by mistake."""
    for bad_value in ("", "Full_Override", "full-override", "none", None):
        try:
            assert_uses_full_override(bad_value)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"expected ValueError for bad_value={bad_value!r}"


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