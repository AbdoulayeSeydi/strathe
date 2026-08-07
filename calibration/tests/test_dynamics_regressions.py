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
    def test_diagonal_mass_matrix(self):
        rng = np.random.default_rng(0)
        max_violation = 0.0
        for _ in range(1000):
            M = np.diag(rng.uniform(1, 50, 6))
            nu = rng.uniform(-5, 5, 6)
            Cnu = _coriolis_centripetal_force(M, nu)
            work = np.dot(nu, Cnu)
            max_violation = max(max_violation, abs(work))
        assert max_violation < 1e-8

    def test_general_mass_matrix_with_off_diagonal_coupling(self):
        rng = np.random.default_rng(1)
        max_violation = 0.0
        for _ in range(1000):
            M = np.diag(rng.uniform(1, 50, 6))
            M[0, 4] = M[4, 0] = rng.uniform(-0.4, 0.4)
            M[2, 4] = M[4, 2] = rng.uniform(-0.4, 0.4)
            nu = rng.uniform(-5, 5, 6)
            Cnu = _coriolis_centripetal_force(M, nu)
            work = np.dot(nu, Cnu)
            max_violation = max(max_violation, abs(work))
        assert max_violation < 1e-8

    def test_coordinated_turn_case(self):
        M = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        nu = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        Cnu = _coriolis_centripetal_force(M, nu)
        assert Cnu[1] != 0.0
        for i in (0, 2, 3, 4, 5):
            assert abs(Cnu[i]) < 1e-10


class TestOffDiagonalMassRegression:
    def test_zero_coupling_matches_pure_diagonal(self):
        rng = np.random.default_rng(2)
        max_diff = 0.0
        for _ in range(200):
            diag_vals = rng.uniform(1, 50, 6)
            M_general = np.diag(diag_vals)
            nu = rng.uniform(-5, 5, 6)
            general_result = _coriolis_centripetal_force(M_general, nu)
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
        vehicle = VehicleConstants(mass_kg=10.0)
        params_no_coupling = _make_test_params()
        params_with_coupling = _make_test_params(Xq_dot=0.3, Zq_dot=0.2)
        v = np.zeros(6)
        tau_pure_surge = np.array([10.0, 0, 0, 0, 0, 0])
        v_dot_no, _, _ = compute_acceleration(params_no_coupling, vehicle, v, tau_pure_surge, 0, 0)
        v_dot_with, _, _ = compute_acceleration(params_with_coupling, vehicle, v, tau_pure_surge, 0, 0)
        assert v_dot_no[4] == pytest.approx(0.0, abs=1e-10)
        assert abs(v_dot_with[4]) > 1e-6


class TestMassMatrixPositiveDefiniteness:
    def test_rejects_invalid_combination(self):
        vehicle = VehicleConstants(mass_kg=10.0)
        params = _make_test_params(Xq_dot=100.0)
        with pytest.raises(ValueError, match="not positive-definite"):
            _mass_and_damping_arrays(params, vehicle)

    def test_accepts_valid_combination(self):
        vehicle = VehicleConstants(mass_kg=10.0)
        params = _make_test_params(Xq_dot=1.0)
        M, _, _ = _mass_and_damping_arrays(params, vehicle)
        assert M is not None


class TestCalibrationParametersBackwardCompatibility:
    def test_old_saved_dict_loads_with_defaults(self):
        old_saved_dict = dict(
            Ieff_roll=5.0, Ieff_pitch=4.36, Ieff_yaw=1.97, Xu=-42.63, Yv=-80.0, Zw=-100.0,
            Kp=-5.0, Mq=-5.0, Nr=-3.17, Xuu=-100.0, Yvv=-100.0, Zww=-150.0, Kpp=-10.0,
            Mqq=-10.0, Nrr=-1.70, Xu_dot=30.0, Yv_dot=40.0, Zw_dot=40.0,
            net_buoyancy_z=2.78, cg_cb_offset_z=0.06,
        )
        params = CalibrationParameters(**old_saved_dict)
        assert params.Xq_dot == 0.0
        assert params.Zq_dot == 0.0
        assert len(params.to_vector()) == 22


class TestSplitByGroupAutomaticDefault:
    def test_holds_out_most_recent_groups_by_default(self):
        from data_split import split_by_group

        filepaths = [f"odom-{i:02d}-01-2024-run1.csv" for i in range(1, 11)]

        def group_key_fn(fp):
            return fp.split("-", 1)[1].rsplit("-run", 1)[0]

        def sort_key_fn(group):
            dd, mm, yyyy = group.split("-")
            return (int(yyyy), int(mm), int(dd))

        result = split_by_group(filepaths, group_key_fn, sort_key_fn, heldout_fraction=0.3)
        expected_heldout_days = {"08", "09", "10"}
        actual_heldout_days = {g.split("-")[0] for g in result.heldout_groups}
        assert actual_heldout_days == expected_heldout_days

    def test_explicit_heldout_groups_still_override_the_default(self):
        from data_split import split_by_group

        filepaths = [f"odom-{i:02d}-01-2024-run1.csv" for i in range(1, 11)]

        def group_key_fn(fp):
            return fp.split("-", 1)[1].rsplit("-run", 1)[0]

        def sort_key_fn(group):
            dd, mm, yyyy = group.split("-")
            return (int(yyyy), int(mm), int(dd))

        result = split_by_group(filepaths, group_key_fn, sort_key_fn,
                                 heldout_groups=["03-01-2024"])
        actual_heldout_days = {g.split("-")[0] for g in result.heldout_groups}
        assert actual_heldout_days == {"03"}
