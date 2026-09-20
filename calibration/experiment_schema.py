"""
experiment_schema.py -- single source of truth for fleet experiment
configuration (G6/F1, see conversation). Architecture decisions recorded
here explicitly, matching phase2_config.py's own Decision N discipline:
made explicit and importable, not just agreed on in conversation.

## Decision 1: severity + optional overrides, not raw parameters only

The communication configuration accepts a NAMED severity level (L0, L1,
L2, L3a, L3b) as the base, with an optional overrides block that may set
any of the underlying numeric parameters on top:
  near_distance_m, far_distance_m, near_field_prob, far_field_prob,
  bytes_per_tick, speed_of_sound_mps

No severity + only overrides is also valid -- tagged as "custom" in
MLflow and in reports, never mislabeled as a named severity it doesn't
actually match.

Alternative considered and rejected: raw numeric parameters only, no
named severities. This would be maximally flexible but would break the
connection to the vocabulary used everywhere else in this project
(evaluate_comm_degradation_curve.py's COMM_CONFIGS, README, PROJECT_STATUS.md,
MLflow tags). A result logged as "drop_prob=0.3" is not comparable to one
logged as "L2" without knowing those are the same thing. Named severities
are the shareable, reproducible vocabulary -- they are what the paper and
README cite, and they are what a lab partner would reproduce.

Alternative considered and rejected: severity only, no overrides. This
would force a source edit for any non-standard comm condition, which is
exactly the G3/G6 gap this file exists to close. Overrides let a
researcher explore the parameter space without touching source code.

The two can coexist without conflict because the named severity sets
defaults and overrides apply on top -- the effective parameters are always
fully resolved before any validation or MLflow logging, so the result is
never ambiguous about what actually ran.

## Decision 2: N=2 vehicles, rejected loudly if violated

This schema explicitly rejects anything other than exactly 2 vehicles,
with a clear error message naming generalization as explicitly out of scope.

Alternative considered and rejected: silently clamp or ignore extra
vehicles. A user who specifies 3 vehicles and gets a 2-agent result has
run a meaningless experiment without knowing it. Loud rejection is the
only honest behavior when the system can't do what was asked.

Alternative considered: accept N vehicles, implement N-agent support. Out
of scope for this track -- the orchestrator, environment, action space,
and observation space are all hardcoded to 2. The right time to relax
this constraint is when N-agent support is actually built, and this
validation is the right place to do it -- one deliberate change, not a
scattered set of silent assumptions.

## Decision 3: Pydantic BaseModel, not dataclasses

Matches vehicle_spec_schema.py's own pattern: Pydantic gives field-level
error messages ("vehicles[0].model_path: file not found"), type coercion,
and Literal constraints without hand-written if/else chains. The previous
dataclass + manual validate_experiment_config() approach was functionally
correct but structurally inconsistent with the rest of the project's
schema discipline.

Usage:
    from experiment_schema import load_and_validate
    config = load_and_validate("default_experiment.yaml")
    config.vehicles[0].model_path   # type-checked
"""
from __future__ import annotations

import os
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from comm_channel import PacketLossConfig, BandwidthConfig
from evaluate_comm_degradation_curve import COMM_CONFIGS
import mlflow_schema

VALID_DYNAMICS_BACKENDS = mlflow_schema.VALID_DYNAMICS_BACKENDS

# Type aliases -- same discipline as vehicle_spec_schema.py's Confidence alias.
# Literal gives Pydantic free validation AND documents the valid values inline.
SeverityLevel = Literal["L0", "L1", "L2", "L3a", "L3b"]
Algorithm = Literal["ppo", "sac", "td3"]
DynamicsBackend = Literal["stand_in", "mujoco", "both"]

REQUIRED_N_VEHICLES = 2  # explicit constant so the error message is clear


class VehicleConfig(BaseModel):
    model_path: str
    algorithm: Algorithm = "ppo"
    role: str = ""  # human-readable label for the report ("calibrated"/"generic" etc.)

    @model_validator(mode="after")
    def check_model_file_exists(self):
        path = (self.model_path if self.model_path.endswith(".zip")
                else f"{self.model_path}.zip")
        if not os.path.exists(path):
            raise ValueError(
                f"model_path {self.model_path!r} not found (also tried {path!r}). "
                f"Check the path relative to where you're running this script."
            )
        return self


class CommOverrides(BaseModel):
    """Optional numeric overrides applied on top of a named severity level.
    All fields are None by default -- only set the ones you want to change."""
    near_distance_m: Optional[float] = None
    far_distance_m: Optional[float] = None
    near_field_prob: Optional[float] = None
    far_field_prob: Optional[float] = None
    bytes_per_tick: Optional[float] = None
    speed_of_sound_mps: Optional[float] = None

    @model_validator(mode="after")
    def check_valid_ranges(self):
        for name, val in [("near_field_prob", self.near_field_prob),
                           ("far_field_prob", self.far_field_prob)]:
            if val is not None and not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        if self.speed_of_sound_mps is not None and self.speed_of_sound_mps <= 0:
            raise ValueError(
                f"speed_of_sound_mps must be positive, got {self.speed_of_sound_mps}"
            )
        return self


class CommConfig(BaseModel):
    severity: Optional[SeverityLevel] = None
    overrides: CommOverrides = Field(default_factory=CommOverrides)

    @model_validator(mode="after")
    def check_has_comm_spec(self):
        """At least one of severity or a real override must be set.
        Also validates the fully-resolved PacketLossConfig via its own
        .validate() -- catches cross-field constraints (far_distance_m <
        near_distance_m, negative distances) that can only be verified once
        base severity values and any overrides are both known.
        _build_packet_loss_config is defined later in this module; Python
        resolves it at call time, not definition time, so forward reference is safe."""
        has_severity = self.severity is not None
        has_overrides = any(
            getattr(self.overrides, f) is not None
            for f in self.overrides.model_fields
        )
        if not has_severity and not has_overrides:
            raise ValueError(
                "communication must specify at least a severity level or one override. "
                "Use severity: L2 for a named condition, or overrides: {...} for a custom one."
            )
        # Validate the fully-resolved PacketLossConfig -- this is what F1.3
        # requires: calling PacketLossConfig.validate() on the merged result,
        # not just checking individual override fields in isolation.
        try:
            plc = _build_packet_loss_config(self)
            plc.validate()
        except ValueError as e:
            raise ValueError(
                f"comm configuration invalid after resolving overrides: {e}"
            ) from e
        return self


class TaskConfig(BaseModel):
    target_separation_m: float = 2.0
    tolerance_m: float = 0.2
    hold_steps: int = 20

    @model_validator(mode="after")
    def check_positive(self):
        if self.target_separation_m <= 0:
            raise ValueError(
                f"target_separation_m must be positive, got {self.target_separation_m}"
            )
        if self.tolerance_m <= 0:
            raise ValueError(f"tolerance_m must be positive, got {self.tolerance_m}")
        if self.hold_steps <= 0:
            raise ValueError(f"hold_steps must be positive, got {self.hold_steps}")
        return self


class EvaluationConfig(BaseModel):
    episodes: int = 160
    comm_seeds: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    grid_seeds: int = 5

    @model_validator(mode="after")
    def check_positive(self):
        if self.episodes <= 0:
            raise ValueError(
                f"evaluation.episodes must be a positive integer, got {self.episodes}"
            )
        if self.grid_seeds <= 0:
            raise ValueError(
                f"evaluation.grid_seeds must be a positive integer, got {self.grid_seeds}"
            )
        if not self.comm_seeds:
            raise ValueError("evaluation.comm_seeds must be a non-empty list of integers")
        return self


class ExperimentConfig(BaseModel):
    name: str = ""
    vehicles: list[VehicleConfig]
    communication: CommConfig
    task: TaskConfig = Field(default_factory=TaskConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    dynamics_backend: DynamicsBackend = "stand_in"
    output_dir: str

    @model_validator(mode="after")
    def check_vehicle_count_and_output(self):
        """Decision 2: N=2 vehicles, rejected loudly if violated.
        Also validates output_dir writability by actually calling makedirs --
        proving write permission before any compute is spent, not partway
        through a long sweep when it's too late to recover cleanly."""
        if len(self.vehicles) != REQUIRED_N_VEHICLES:
            raise ValueError(
                f"Exactly {REQUIRED_N_VEHICLES} vehicles are required (hardcoded for "
                f"two-agent coordination). Got {len(self.vehicles)}. "
                f"Generalizing to N agents is explicitly out of scope for this version."
            )
        if not self.output_dir:
            raise ValueError("output_dir must be a non-empty path.")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            raise ValueError(
                f"output_dir {self.output_dir!r} cannot be created or is not writable: {e}"
            ) from e
        return self


# ---------------------------------------------------------------------------
# Helper functions: resolve the effective comm parameters, build comm objects
# ---------------------------------------------------------------------------

def _build_packet_loss_config(comm: CommConfig) -> PacketLossConfig:
    """Build a PacketLossConfig from the resolved comm settings."""
    base = COMM_CONFIGS.get(comm.severity or "", {}).get("packet_loss_config")
    ov = comm.overrides
    if base is not None:
        near_dist = ov.near_distance_m if ov.near_distance_m is not None else base.near_distance_m
        far_dist = ov.far_distance_m if ov.far_distance_m is not None else base.far_distance_m
        near_prob = ov.near_field_prob if ov.near_field_prob is not None else base.near_field_prob
        far_prob = ov.far_field_prob if ov.far_field_prob is not None else base.far_field_prob
    else:
        near_dist = ov.near_distance_m or 0.0
        far_dist = ov.far_distance_m or 1000.0
        near_prob = ov.near_field_prob or 0.0
        far_prob = ov.far_field_prob or 0.0
    return PacketLossConfig(near_dist, far_dist, near_prob, far_prob)


def _build_bandwidth_config(comm: CommConfig) -> BandwidthConfig:
    """Build a BandwidthConfig from the resolved comm settings."""
    base = COMM_CONFIGS.get(comm.severity or "", {}).get("bandwidth_config")
    if comm.overrides.bytes_per_tick is not None:
        return BandwidthConfig(bytes_per_tick=comm.overrides.bytes_per_tick)
    if base is not None:
        return base
    return BandwidthConfig(bytes_per_tick=1e9)


def _build_speed_of_sound(comm: CommConfig) -> float:
    """Return the effective speed of sound (m/s)."""
    return (comm.overrides.speed_of_sound_mps
            if comm.overrides.speed_of_sound_mps is not None else 1500.0)


def _effective_severity_label(comm: CommConfig) -> str:
    """Return the severity label for MLflow tagging.
    Uses the named severity if set, 'custom' if overrides-only."""
    has_overrides = any(
        getattr(comm.overrides, f) is not None
        for f in comm.overrides.model_fields
    )
    if comm.severity is not None and not has_overrides:
        return comm.severity
    if comm.severity is not None:
        return f"{comm.severity}_custom"
    return "custom"


def build_comm_objects(config: ExperimentConfig):
    """Return (packet_loss_config, bandwidth_config, speed_of_sound_mps)
    ready to pass to TwoAgentFormationEnv."""
    return (
        _build_packet_loss_config(config.communication),
        _build_bandwidth_config(config.communication),
        _build_speed_of_sound(config.communication),
    )


# ---------------------------------------------------------------------------
# Load and validate -- matching vehicle_spec_schema.py's own pattern exactly
# ---------------------------------------------------------------------------

def load_and_validate(yaml_path: str) -> ExperimentConfig:
    """Loads an experiment YAML and validates it against the schema above.
    Raises ValueError with Pydantic's field-level messages on any problem --
    matching vehicle_spec_schema.py's load_and_validate pattern exactly."""
    try:
        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(
            f"Experiment config file not found: {yaml_path!r}. "
            f"Copy default_experiment.yaml and edit it, or specify a different path."
        )
    except yaml.YAMLError as e:
        raise ValueError(f"Experiment config YAML parse error in {yaml_path!r}: {e}")

    from pydantic import ValidationError
    try:
        return ExperimentConfig(**raw)
    except ValidationError as e:
        raise ValueError(
            f"Experiment config validation failed in {yaml_path!r}:\n{e}"
        ) from e


# Backward-compatible alias so existing callers don't break while migrating.
load_experiment_config = load_and_validate


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "default_experiment.yaml"
    config = load_and_validate(path)
    print(f"OK -- {path} validated successfully.")
    print(f"Name: {config.name or '(unnamed)'}")
    print(f"Vehicles: {len(config.vehicles)}")
    for i, v in enumerate(config.vehicles):
        print(f"  [{i}] {v.model_path} ({v.algorithm.upper()})  role={v.role or '(none)'}")
    print(f"Communication: severity={config.communication.severity or 'custom'}, "
          f"label={_effective_severity_label(config.communication)}")
    print(f"Task: target_separation={config.task.target_separation_m}m, "
          f"tolerance={config.task.tolerance_m}m, hold_steps={config.task.hold_steps}")
    print(f"Evaluation: episodes={config.evaluation.episodes}, "
          f"comm_seeds={config.evaluation.comm_seeds}, grid_seeds={config.evaluation.grid_seeds}")
    print(f"Dynamics backend: {config.dynamics_backend}")
    print(f"Output: {config.output_dir}")
