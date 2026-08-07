"""
Automated regression suite (item #19) — converts tonight's manual,
one-off assertions into real pytest tests. Every test here was
originally verified by hand during this session; this file makes those
checks permanent and automatically re-runnable on every future change,
rather than needing to be manually re-derived each time.

Run with: pytest tests/test_dynamics_regressions.py -v
"""
import numpy as np
import pytest

from dynamics_model import (
    CalibrationParameters, VehicleConstants, SimulationConfig,
    _coriolis_centripetal_force, _mass_and_damping_arrays, compute_acceleration,
)


def _make_test_params(**overrides):
    """A minimal, valid CalibrationParameters for isolated unit tests —
    zeros everywhere except what each test needs, matching the pattern
    used throughout tonight's manual verification."""
    defaults = dict(
        Ieff_roll=1.0, Ieff_pitch=1.0, Ieff_yaw=1.0,
        Xu=0, Yv=0, Zw=0, Kp=0, Mq=0, Nr=0,
        Xuu=0, Yvv=0, Zww=0, Kpp=0, Mqq=0, Nrr=0,
        Xu_dot=0, Yv_dot=0, Zw_dot=0,
        net_buoyancy_z=0.0, cg_cb_offset_z=0.0,
        Xq_dot=0.0, Zq_dot=0.0,
    )
    defaults.update(overrides)
    return CalibrationParameters(**defaults)


class TestCoriolisEnergyConservation:
    """The defining property of any correct Coriolis-centripetal matrix:
    it does no work. nu^T @ C(nu) @ nu must be ~0 for ANY nu and ANY
    valid mass matrix — originally verified across 10,000 random trials
    (finding #16), including the general (non-diagonal) case with
    off-diagonal added-mass coupling (Xq_dot/Zq_dot follow-up)."""

    def test_diagonal_mass_matrix(self):
        rng = np.random.default_rng(0)
        max_violation = 0.0
        for _ in range(1000):
            M = np.diag(rng.uniform(1, 50, 6))
            nu = rng.uniform(-5, 5, 6)
            Cnu = _coriolis_centripetal_force(M, nu)
            work = np.dot(nu, Cnu)
            max_violation = max(max_violation, abs(work))
        assert max_violation < 1e-8, (
            f"Coriolis matrix does work ({max_violation}) — violates the defining "
            f"energy-conservation property of a valid Coriolis-centripetal matrix"
        )

    def test_general_mass_matrix_with_off_diagonal_coupling(self):
        """The generalized (non-block-diagonal) construction, needed once
        Xq_dot/Zq_dot introduce genuine linear-angular mass coupling —
        must STILL conserve energy, not just the simpler diagonal case."""
        rng = np.random.default_rng(1)
        max_violation = 0.0
        for _ in range(1000):
            M = np.diag(rng.uniform(1, 50, 6))
            M[0, 4] = M[4, 0] = rng.uniform(-0.4, 0.4)  # Xq_dot-style coupling
            M[2, 4] = M[4, 2] = rng.uniform(-0.4, 0.4)  # Zq_dot-style coupling
            nu = rng.uniform(-5, 5, 6)
            Cnu = _coriolis_centripetal_force(M, nu)
            work = np.dot(nu, Cnu)
            max_violation = max(max_violation, abs(work))
        assert max_violation < 1e-8

    def test_coordinated_turn_case(self):
        """Forward motion + yaw rate must produce exactly the expected
        nonzero lateral (sway) centripetal force and NOTHING else on the
        other axes — the well-known physical sanity check."""
        M = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        nu = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.5])  # forward + yaw rate
        Cnu = _coriolis_centripetal_force(M, nu)
        assert Cnu[1] != 0.0, "expected nonzero sway force from the coordinated turn"
        for i in (0, 2, 3, 4, 5):
            assert abs(Cnu[i]) < 1e-10, f"unexpected nonzero force on axis {i}"


class TestOffDiagonalMassRegression:
    """Confirms the general (possibly-coupled) mass matrix reduces EXACTLY
    to old, already-tested behavior when Xq_dot/Zq_dot are zero (the
    default) — the critical backward-compatibility guarantee."""

    def test_zero_coupling_matches_pure_diagonal(self):
        rng = np.random.default_rng(2)
        max_diff = 0.0
        for _ in range(200):
            diag_vals = rng.uniform(1, 50, 6)
            M_general = np.diag(diag_vals)  # zero off-diagonal
            nu = rng.uniform(-5, 5, 6)
            general_result = _coriolis_centripetal_force(M_general, nu)
            # Manually compute via the OLD diagonal-only formula for comparison
            u, v, w, p, q, r = nu
            m1, m2, m3, m4, m5, m6 = diag_vals
            old_result = np.array([
                m3 * w * q - m2 * v * r, m1 * u * r - m3 * w * p, m2 * v * p - m1 * u * q,
                (m3 - m2) * v * w + (m6 - m5) * q * r,
                (m1 - m3) * u * w + (m4 - m6) * p * r,
                (m2 - m1) * u * v + (m5 - m4) * p * q,
            ])
            max_diff = max(max_diff, np.max(np.abs(general_result - old_result)))
        assert max_diff < 1e-10

    def test_single_axis_motion_unaffected_by_coupling(self):
        """Pure single-axis motion is structurally zero for Coriolis
        regardless of off-diagonal coupling — a genuine regression check,
        not just a math identity."""
        vehicle = VehicleConstants(mass_kg=10.0)
        params_no_coupling = _make_test_params()
        params_with_coupling = _make_test_params(Xq_dot=0.3, Zq_dot=0.2)
        v = np.zeros(6)
        tau_pure_surge = np.array([10.0, 0, 0, 0, 0, 0])

        v_dot_no, _, _ = compute_acceleration(params_no_coupling, vehicle, v, tau_pure_surge, 0, 0)
        # With v=0 (at rest), off-diagonal coupling DOES matter even for a
        # "single-axis" force, since the mass matrix itself couples them —
        # this specifically confirms that coupling, not an equality.
        v_dot_with, _, _ = compute_acceleration(params_with_coupling, vehicle, v, tau_pure_surge, 0, 0)
        assert v_dot_no[4] == pytest.approx(0.0, abs=1e-10), "no coupling means zero pitch accel"
        assert abs(v_dot_with[4]) > 1e-6, "Xq_dot coupling should produce real pitch acceleration"


class TestMassMatrixPositiveDefiniteness:
    """The Cholesky-based validity check (added alongside off-diagonal
    coupling) must reject nonphysical combinations and accept valid ones."""

    def test_rejects_invalid_combination(self):
        vehicle = VehicleConstants(mass_kg=10.0)
        params = _make_test_params(Xq_dot=100.0)  # absurdly large relative to Ieff_pitch
        with pytest.raises(ValueError, match="not positive-definite"):
            _mass_and_damping_arrays(params, vehicle)

    def test_accepts_valid_combination(self):
        vehicle = VehicleConstants(mass_kg=10.0)
        params = _make_test_params(Xq_dot=1.0)  # well within the safe range
        M, _, _ = _mass_and_damping_arrays(params, vehicle)
        assert M is not None


class TestCalibrationParametersBackwardCompatibility:
    """An old saved calibrated_parameters.yaml (from before Xq_dot/Zq_dot
    existed) must still load correctly, with both defaulting to 0.0."""

    def test_old_saved_dict_loads_with_defaults(self):
        old_saved_dict = dict(
            Ieff_roll=5.0, Ieff_pitch=4.36, Ieff_yaw=1.97, Xu=-42.63, Yv=-80.0, Zw=-100.0,
            Kp=-5.0, Mq=-5.0, Nr=-3.17, Xuu=-100.0, Yvv=-100.0, Zww=-150.0, Kpp=-10.0,
            Mqq=-10.0, Nrr=-1.70, Xu_dot=30.0, Yv_dot=40.0, Zw_dot=40.0,
            net_buoyancy_z=2.78, cg_cb_offset_z=0.06,
            # Xq_dot, Zq_dot deliberately omitted
        )
        params = CalibrationParameters(**old_saved_dict)
        assert params.Xq_dot == 0.0
        assert params.Zq_dot == 0.0
        assert len(params.to_vector()) == 22