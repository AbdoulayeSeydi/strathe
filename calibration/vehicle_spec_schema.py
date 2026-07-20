"""
Pydantic schema for vehicle_spec.yaml. Validates the file on load so a
malformed spec (bad bounds, missing provenance, wrong confidence label)
fails immediately and clearly — instead of surfacing as a confusing crash
deep inside the optimizer, or worse, silently producing garbage results.

Usage:
    from vehicle_spec_schema import load_and_validate
    spec = load_and_validate("vehicle_spec.yaml")
    spec.fixed_parameters.mass_kg.value   # -> 11.2, type-checked
"""
from typing import Literal
import yaml
from pydantic import BaseModel, field_validator

Confidence = Literal["published", "literature", "estimated", "assumed"]


class CalibrationParam(BaseModel):
    unit: str
    initial_guess: float
    bounds: tuple[float, float]
    bounds_rationale: str
    source: str
    confidence: Confidence

    @field_validator("bounds")
    @classmethod
    def bounds_ordered(cls, v):
        lo, hi = v
        if lo > hi:
            raise ValueError(f"bounds lower ({lo}) > upper ({hi})")
        return v

    @field_validator("initial_guess")
    @classmethod
    def guess_within_bounds(cls, v, info):
        bounds = info.data.get("bounds")
        if bounds and not (bounds[0] <= v <= bounds[1]):
            raise ValueError(f"initial_guess {v} falls outside bounds {bounds}")
        return v


class FixedMass(BaseModel):
    value: float
    unit: str
    source: str
    confidence: Confidence


class FixedParameters(BaseModel):
    mass_kg: FixedMass


class EffectiveInertia(BaseModel):
    """REPARAMETERIZED — was InertiaDiag (Ixx/Iyy/Izz) kept separate from
    AddedMassDiag's Kp_dot/Mq_dot/Nr_dot. A Jacobian-based identifiability
    analysis on real data found Izz/Nr_dot correlated at exactly -1.000 —
    proven structurally non-identifiable, not just hard to fit, since they
    only ever appear as a sum in the dynamics. See dynamics_model.py's
    "Structural non-identifiability" docstring section for the full story."""
    Ieff_roll: CalibrationParam
    Ieff_pitch: CalibrationParam
    Ieff_yaw: CalibrationParam


class LinearDamping(BaseModel):
    Xu: CalibrationParam
    Yv: CalibrationParam
    Zw: CalibrationParam
    Kp: CalibrationParam
    Mq: CalibrationParam
    Nr: CalibrationParam


class QuadraticDamping(BaseModel):
    Xuu: CalibrationParam
    Yvv: CalibrationParam
    Zww: CalibrationParam
    Kpp: CalibrationParam
    Mqq: CalibrationParam
    Nrr: CalibrationParam


class AddedMassDiag(BaseModel):
    # Rotational added mass (was Kp_dot/Mq_dot/Nr_dot here) moved into
    # EffectiveInertia above — see the reparameterization note there.
    Xu_dot: CalibrationParam
    Yv_dot: CalibrationParam
    Zw_dot: CalibrationParam


class Buoyancy(BaseModel):
    net_buoyancy_z: CalibrationParam
    cg_cb_offset_z: CalibrationParam


class CalibrationTargets(BaseModel):
    effective_inertia: EffectiveInertia
    linear_damping: LinearDamping
    quadratic_damping: QuadraticDamping
    added_mass_diag: AddedMassDiag
    buoyancy: Buoyancy


class Vehicle(BaseModel):
    name: str


class Environment(BaseModel):
    current_velocity_mps: float
    current_direction_deg: float
    source: str
    confidence: Confidence
    note: str


class Payload(BaseModel):
    description: str
    additional_mass_kg: float
    note: str


class ThrusterModel(BaseModel):
    calibrated: bool
    count: int
    hardware: str
    source: str
    confidence: Confidence
    note: str
    future_calibration_targets: dict


class OptimizationConfig(BaseModel):
    algorithm: str
    loss: str
    ftol: float


class ValidationConfig(BaseModel):
    primary_metric: str
    identifiability_note: str


class VehicleSpec(BaseModel):
    schema_version: str
    metadata: dict
    assumptions: list[str]
    vehicle: Vehicle
    environment: Environment
    payload: Payload
    thruster_model: ThrusterModel
    fixed_parameters: FixedParameters
    calibration_targets: CalibrationTargets
    optimization: OptimizationConfig
    validation: ValidationConfig


def load_and_validate(yaml_path: str) -> VehicleSpec:
    """Loads vehicle_spec.yaml and validates it against the schema above.
    Raises pydantic.ValidationError with a specific field-level message on
    any problem — bad bounds, missing provenance, wrong types, etc."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return VehicleSpec(**raw)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "vehicle_spec.yaml"
    spec = load_and_validate(path)
    print(f"OK — schema_version {spec.schema_version} validated successfully.")
    print(f"Vehicle: {spec.vehicle.name}, mass: {spec.fixed_parameters.mass_kg.value} kg")
    n_targets = (
        len(spec.calibration_targets.effective_inertia.model_fields)
        + len(spec.calibration_targets.linear_damping.model_fields)
        + len(spec.calibration_targets.quadratic_damping.model_fields)
        + len(spec.calibration_targets.added_mass_diag.model_fields)
        + len(spec.calibration_targets.buoyancy.model_fields)
    )
    print(f"Calibration targets: {n_targets} parameters")