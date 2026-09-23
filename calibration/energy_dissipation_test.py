"""
Energy dissipation test -- Abdoul's item #7, refined based on findings
from a preliminary run against a different (sandbox) calibrated
parameter set (see conversation for full derivation).

PART 1: Directly checks the sign of every calibrated damping
coefficient. This is a PROVABLE, decisive check, not an empirical one:
given d(KE+PE)/dt = v . damping_force (Coriolis is confirmed
energy-conserving elsewhere in this codebase; the restoring force's
power exchange is exactly -d(PE)/dt for a conservative force), and
damping_force = linear_damping*v + quad_damping*|v|*v, if every
linear_damping/quad_damping coefficient is negative, then v.damping_force
is a sum of non-positive terms for ANY v -- meaning total mechanical
energy is MATHEMATICALLY GUARANTEED never to increase, regardless of
orientation or trajectory. A positive coefficient would be a direct,
checkable violation of this guarantee for that DOF specifically.

PART 2: Empirical confirmation via a long-horizon (200s simulated),
zero-external-forcing (tau=0) integration starting from a HIGH-energy,
near-inverted state (matching the scale seen in real problem runs).
Tracks whether kinetic energy eventually settles (does NOT need to
reach exactly zero -- see PART 1's own note below on why a nonzero
plateau can be entirely expected) rather than growing unboundedly.

IMPORTANT INTERPRETATION NOTE, found in the sandbox run: KE settled at
a CONSTANT, NONZERO plateau rather than decaying to zero. This is NOT
necessarily concerning -- investigate the settled STATE (printed below)
before concluding anything. In the sandbox run, the settled state was
almost pure heave velocity at a roll=180/pitch=0 orientation -- a
genuine physical explanation (heave has a nonzero TERMINAL velocity
under constant net buoyancy even with tau=0, exactly like a dropped
object reaching terminal velocity in water; the angular part settled
exactly onto an equilibrium orientation, which a noiseless deterministic
simulation can sit on indefinitely even if that equilibrium is
technically unstable to real-world perturbation). Check whether YOUR
real parameters show the same pattern, or something different.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from dynamics_model import compute_acceleration_from_rotation_matrix, _mass_and_damping_arrays
from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec

vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
params = load_calibrated_parameters("calibrated_parameters_a1.yaml")
M, _, _ = _mass_and_damping_arrays(params, vehicle)

print("=" * 70)
print("PART 1: Damping coefficient sign check (provable, not empirical)")
print("=" * 70)
all_negative = True
print("Linear damping (expect all NEGATIVE):")
for name in ["Xu", "Yv", "Zw", "Kp", "Mq", "Nr"]:
    val = getattr(params, name)
    ok = val < 0
    all_negative = all_negative and ok
    print(f"  {name} = {val:.4f}" + ("" if ok else "  <-- WRONG SIGN, adds energy for this DOF"))
print("Quadratic damping (expect all NEGATIVE):")
for name in ["Xuu", "Yvv", "Zww", "Kpp", "Mqq", "Nrr"]:
    val = getattr(params, name)
    ok = val < 0
    all_negative = all_negative and ok
    print(f"  {name} = {val:.4f}" + ("" if ok else "  <-- WRONG SIGN, adds energy for this DOF"))

if all_negative:
    print("\nALL damping coefficients negative -- total mechanical energy is")
    print("MATHEMATICALLY GUARANTEED to never increase in this model, for any state.")
else:
    print("\nAT LEAST ONE WRONG-SIGN COEFFICIENT FOUND -- this IS a real, direct")
    print("explanation for energy growth on whichever DOF(s) are flagged above.")
    print("This would be a genuine calibration/model issue, not an artifact.")

print()
print("=" * 70)
print("PART 2: Long-horizon (200s), zero-forcing energy test")
print("=" * 70)


def energy_test(nu0, roll0, pitch0, n_steps, dt=0.005):
    v = nu0.copy()
    R = Rotation.from_euler("xyz", [roll0, pitch0, 0.0]).as_matrix()
    tau = np.zeros(6)
    energies = []
    for _ in range(n_steps):
        energies.append(0.5 * v @ M @ v)
        v_dot, _, _ = compute_acceleration_from_rotation_matrix(params, vehicle, v, tau, R)
        v = v + v_dot * dt
        incremental = Rotation.from_rotvec(v[3:] * dt).as_matrix()
        R = R @ incremental
        if not np.all(np.isfinite(v)):
            return np.array(energies), False, v, R
    return np.array(energies), True, v, R


nu0_high = np.array([0.0, 0.0, 0.0, 1.5, 1.0, 1.6])  # matching real problem-run peak angular rate scale
n_steps = 40000  # 200 seconds at dt=0.005
energies, finite, v_final, R_final = energy_test(
    nu0_high, roll0=np.radians(170), pitch0=np.radians(85), n_steps=n_steps)

checkpoints = [0, n_steps // 20, n_steps // 10, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
for cp in checkpoints:
    print(f"  step {cp:>6} (t={cp * 0.005:>6.1f}s): KE={energies[cp]:.6f}")
print(f"  overall min KE reached: {np.min(energies):.6f}")
print(f"  overall max KE reached: {np.max(energies):.6f}")
print(f"  finite: {finite}")

if finite:
    roll_f, pitch_f, yaw_f = Rotation.from_matrix(R_final).as_euler("xyz")
    print(f"\nFinal state (t=200s): v={v_final}")
    print(f"  roll={np.degrees(roll_f):.2f} deg, pitch={np.degrees(pitch_f):.2f} deg, yaw={np.degrees(yaw_f):.2f} deg")
    print(f"  |p|,|q|,|r| = {abs(v_final[3]):.6f}, {abs(v_final[4]):.6f}, {abs(v_final[5]):.6f}")
    print("\nIf angular rates (p,q,r) are ~0 and only a linear velocity (u,v,w) remains")
    print("nonzero and constant, that's consistent with a genuine terminal-velocity")
    print("explanation (not energy growth) -- check whether the orientation is a real")
    print("equilibrium (Tx,Ty both ~0 there) before concluding anything else.")
else:
    print("\nNOT FINITE -- this WOULD be a genuine, serious finding if it holds on your")
    print("real parameters. Do not dismiss this as expected without further investigation.")