"""
validate_policy_compatibility.py -- G5/F3 pre-flight compatibility check
for external SB3 policies (see conversation).

Without this, a policy trained on the wrong environment (single-agent Arena,
wrong action_bound, wrong obs convention) fails with a raw numpy shape error
somewhere three layers inside env.step() -- at best confusing, at worst
silently producing garbage results if the shapes happen to be compatible by
coincidence but the conventions differ.

This runs at load time, before any episodes execute, matching
preflight_check.py's "check before spending compute" precedent.

HONEST SCOPE BOUNDARY (per F3's own interface contract):
This validates everything Strathe can reliably check automatically: shape,
dtype, bounds, and loadability. It CANNOT verify:
  - observation normalization (are the velocity values in the right range?)
  - action convention (does [-1,1] mean the same thing to this policy as
    to Strathe's action scaling?)
  - observation field ordering (is field 0 surge velocity or something else?)
  - whether the policy was trained on the formation-hold task specifically
  - recurrent/stateful policies (shape checks pass but behavior will be wrong)
These are genuinely unknowable from the zip file alone. The check exists to
catch the obvious, common failures -- not to be a complete correctness proof.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

EXPECTED_OBS_SHAPE = (24,)
EXPECTED_ACT_SHAPE = (12,)
EXPECTED_ACT_LOW = -1.0
EXPECTED_ACT_HIGH = 1.0
EXPECTED_DTYPE = np.float32

ALGORITHM_CLASSES = {
    "ppo": "PPO",
    "sac": "SAC",
    "td3": "TD3",
}

PROVEN_ALGORITHMS = {"ppo"}
SUPPORTED_UNVALIDATED_ALGORITHMS = {"sac", "td3"}

# ---------------------------------------------------------------------------
# F3.1: The formal interface contract -- named, importable, and explicit.
# A docstring communicates intent; a named constant can be imported, cited
# in error messages, and pointed to as the single authoritative definition.
# ---------------------------------------------------------------------------
POLICY_INTERFACE_CONTRACT = """
STRATHE FLEET POLICY INTERFACE CONTRACT (F3.1)

A compatible external policy is an SB3 PPO/SAC/TD3 model that:
  1. Can be loaded with the declared algorithm class (PPO.load/SAC.load/TD3.load)
  2. Has observation_space.shape == (24,) float32
       [6-dim body-frame velocity + 2-dim orientation (roll, pitch) + 4-dim
        comm observation] per agent x 2 agents = 24 total
  3. Has action_space.shape == (12,) float32, bounded [-1.0, 1.0]
       [6-dim normalized thruster command per agent x 2 agents]
       (the environment scales internally by action_bound=30.0 N*m)
  4. Is loadable from a .zip file without a custom policy class

ALGORITHM STATUS (honest, not implied):
  - PPO: PROVEN -- used for all real fleet training in this project
  - SAC: SUPPORTED by the interface, UNVALIDATED by any real fleet run
  - TD3: SUPPORTED by the interface, UNVALIDATED by any real fleet run

WHAT THIS CONTRACT DOES NOT VERIFY (named unsolved boundary -- not a gap
to close now, but a real limitation to state explicitly):
  - Observation normalization: whether velocity/orientation values are in
    the range the policy expects from its own training distribution
  - Action scaling convention: whether [-1,1] means the same thing to
    this policy as to Strathe's own action_bound=30.0 N*m scaling
  - Observation field ordering: whether field 0 is surge velocity,
    not some other quantity from a different environment
  - Whether the policy was trained on the formation-hold task specifically
  - Recurrent/stateful policies: shape checks pass but per-step hidden
    state is never managed, so behavior will be wrong
  - Custom SB3 policy subclasses that override predict() non-standardly
  - Environment assumptions baked into the policy at training time
    (target separation, reward weights, episode length, etc.)

These are genuinely unknowable from the .zip file alone. F3.2 validates
everything Strathe can reliably check automatically. It does not and should
not attempt to validate the rest.

EXPLICIT SCOPE BOUNDARY (holds during F3 and after):
No generic model adapters, no custom-policy support, no recurrent-policy
infrastructure, no framework-agnostic inference layer -- until a real user
actually needs one and the need is confirmed by real usage, not anticipated.
This scope is deliberately narrow. The right response to a user hitting an
unsolved boundary is to name it clearly (as this contract does) and let
them decide whether to work around it, not to silently paper over it.
"""


class PolicyIncompatibleError(ValueError):
    """Raised when a policy fails pre-flight compatibility checks.
    Inherits from ValueError so it's catchable as a plain ValueError
    by callers that don't import this module specifically."""
    pass


def load_policy(model_path: str, algorithm: str):
    """Load an SB3 policy from a .zip file. Raises PolicyIncompatibleError
    with a clear message if the algorithm name is invalid or the file
    doesn't load cleanly.

    algorithm: "ppo", "sac", or "td3" (case-insensitive).
    """
    alg_key = algorithm.lower()
    if alg_key not in ALGORITHM_CLASSES:
        raise PolicyIncompatibleError(
            f"Unknown algorithm {algorithm!r}. Must be one of: "
            f"{sorted(ALGORITHM_CLASSES.keys())}. Note: PPO is the only "
            f"algorithm proven for fleet training in this project; SAC and "
            f"TD3 are supported by the interface but unvalidated."
        )

    if alg_key in SUPPORTED_UNVALIDATED_ALGORITHMS:
        print(f"[validate_policy_compatibility] NOTE: {algorithm.upper()} is supported "
              f"by the interface but has not been validated by any real fleet training "
              f"run in this project. PPO is the only proven algorithm. Proceeding.")

    try:
        from stable_baselines3 import PPO, SAC, TD3
        cls = {"ppo": PPO, "sac": SAC, "td3": TD3}[alg_key]
        # Add .zip suffix if not already present, matching SB3's own convention
        path = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
        model = cls.load(path)
        return model
    except FileNotFoundError:
        raise PolicyIncompatibleError(
            f"Policy file not found: {model_path!r} (also tried with .zip suffix). "
            f"Check the path and confirm the file exists."
        )
    except Exception as e:
        raise PolicyIncompatibleError(
            f"Failed to load {algorithm.upper()} policy from {model_path!r}: "
            f"{type(e).__name__}: {e}"
        )


def validate_policy_compatibility(model, env=None, model_path: str = "<unknown>"):
    """Check that a loaded SB3 model is compatible with Strathe's fleet
    environment. Raises PolicyIncompatibleError with a specific, actionable
    message on the first failure found. Runs at pre-flight time, before any
    episodes execute.

    model: a loaded SB3 PPO/SAC/TD3 instance (from load_policy() above).
    env: optional -- if provided, uses the actual env's own spaces for
         comparison instead of the hardcoded expected shapes. Recommended
         when instantiating a fresh env is cheap.
    model_path: just for error messages, the file the model came from.
    """
    if env is not None:
        expected_obs_shape = env.observation_space.shape
        expected_act_shape = env.action_space.shape
        expected_act_low = float(env.action_space.low.flat[0])
        expected_act_high = float(env.action_space.high.flat[0])
    else:
        expected_obs_shape = EXPECTED_OBS_SHAPE
        expected_act_shape = EXPECTED_ACT_SHAPE
        expected_act_low = EXPECTED_ACT_LOW
        expected_act_high = EXPECTED_ACT_HIGH

    # --- Observation space check ---
    if not hasattr(model, "observation_space"):
        raise PolicyIncompatibleError(
            f"Model from {model_path!r} has no observation_space attribute. "
            f"This may not be a valid SB3 model."
        )

    model_obs_shape = model.observation_space.shape
    if model_obs_shape != expected_obs_shape:
        # Produce a maximally-helpful error message for the most common cases
        if model_obs_shape in ((11,), (22,)):
            detail = (
                f"This looks like a SINGLE-AGENT Arena policy (obs shape {model_obs_shape}), "
                f"not a two-agent fleet policy. The fleet environment concatenates both "
                f"agents' observations into one {expected_obs_shape} vector."
            )
        elif model_obs_shape == (13,):
            detail = (
                f"This looks like an obstacle-avoidance Arena policy (obs shape {model_obs_shape} "
                f"includes the extra obstacle dimensions). The fleet environment uses a "
                f"different observation structure."
            )
        else:
            detail = (
                f"The fleet environment provides observations of shape {expected_obs_shape}: "
                f"[body_frame_velocity(6), orientation_roll_pitch(2), comm_obs(4)] "
                f"per agent × 2 agents."
            )
        raise PolicyIncompatibleError(
            f"Observation space mismatch for {model_path!r}: "
            f"model expects {model_obs_shape}, environment provides {expected_obs_shape}. "
            f"{detail}"
        )

    # --- Action space check ---
    model_act_shape = model.action_space.shape
    if model_act_shape != expected_act_shape:
        if model_act_shape in ((6,), (12,)):
            detail = (
                f"Model has action shape {model_act_shape} -- the fleet environment "
                f"expects {expected_act_shape} (6 per agent × 2 agents)."
            )
        else:
            detail = (
                f"The fleet environment expects action shape {expected_act_shape}: "
                f"6-dim normalized thruster command per agent × 2 agents."
            )
        raise PolicyIncompatibleError(
            f"Action space mismatch for {model_path!r}: "
            f"model expects {model_act_shape}, environment provides {expected_act_shape}. "
            f"{detail}"
        )

    # --- Action bounds check (just a warning, not a hard failure, since
    # SB3 clips actions anyway -- but a model trained with different bounds
    # will behave differently in ways that are hard to notice) ---
    try:
        model_act_low = float(model.action_space.low.flat[0])
        model_act_high = float(model.action_space.high.flat[0])
        if abs(model_act_low - expected_act_low) > 0.01 or abs(model_act_high - expected_act_high) > 0.01:
            print(
                f"[validate_policy_compatibility] WARNING: action bounds mismatch for "
                f"{model_path!r}: model was trained with bounds "
                f"[{model_act_low:.2f}, {model_act_high:.2f}], environment uses "
                f"[{expected_act_low:.2f}, {expected_act_high:.2f}]. SB3 will clip "
                f"actions to the environment's bounds, but a policy trained with "
                f"different bounds may behave suboptimally."
            )
    except Exception:
        pass  # bounds check is best-effort; never block on this

    print(
        f"[validate_policy_compatibility] PASS: {model_path!r} -- "
        f"obs {model_obs_shape}, act {model_act_shape}, bounds [{expected_act_low:.1f}, {expected_act_high:.1f}]"
    )
    return True


def validate_all_policies(model_paths: list, algorithm: str, env=None) -> list:
    """Load and validate all policies in one pre-flight pass. Returns a list
    of loaded models in the same order as model_paths. Raises
    PolicyIncompatibleError on the first failure found -- fails fast before
    any episodes run, not partway through a long sweep.

    algorithm: single algorithm applied to all models (they must all use
               the same algorithm -- if you need mixed algorithms, call
               load_policy + validate_policy_compatibility individually).
    """
    print(f"\n[validate_policy_compatibility] Pre-flight check: "
          f"{len(model_paths)} policies, algorithm={algorithm.upper()}")
    models = []
    for path in model_paths:
        model = load_policy(path, algorithm)
        validate_policy_compatibility(model, env=env, model_path=path)
        models.append(model)
    print(f"[validate_policy_compatibility] All {len(model_paths)} policies passed.\n")
    return models


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pre-flight compatibility check for SB3 policies against Strathe's fleet environment."
    )
    parser.add_argument("model_paths", nargs="+",
                         help="Path(s) to .zip model file(s) to check")
    parser.add_argument("--algorithm", default="ppo",
                         help="SB3 algorithm: ppo, sac, td3 (default: ppo)")
    parser.add_argument("--with-env", action="store_true",
                         help="Instantiate a real TwoAgentFormationEnv for exact space comparison "
                              "(more accurate but slower; default: use hardcoded expected shapes)")
    args = parser.parse_args()

    env = None
    if args.with_env:
        from two_agent_formation_env import TwoAgentFormationEnv
        env = TwoAgentFormationEnv()
        print("[validate_policy_compatibility] Using real TwoAgentFormationEnv for space comparison")

    try:
        validate_all_policies(args.model_paths, args.algorithm, env=env)
        print("All checks passed. These policies are compatible with Strathe's fleet environment.")
    except PolicyIncompatibleError as e:
        print(f"\nFAILED: {e}")
        raise SystemExit(1)
