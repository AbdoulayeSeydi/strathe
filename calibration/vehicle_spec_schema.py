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
from typing import Optional, Literal
import yaml
from pydantic import BaseModel, Field, model_validator

Confidence = Literal["measured", "published", "literature", "estimated", "assumed"]


class Reference(BaseModel):
    """An independent measurement of THIS vehicle's value (CAD, a pendulum or tank test, a published model of the same
    vehicle). The calibration report flags a fitted value far from it as physically implausible."""
    value: float
    source: str

    @model_validator(mode="after")
    def nonzero(self):
        if self.value == 0:
            raise ValueError("reference value can't be 0 (the report compares fitted / reference)")
        return self


class CalibrationParam(BaseModel):
    unit: str
    initial_guess: float
    bounds: tuple[float, float]
    bounds_rationale: str
    source: str
    confidence: Confidence
    reference: Optional[Reference] = None

    @model_validator(mode="after")
    def check_bounds_and_guess(self):
        """
        Runs AFTER every field on this model is already parsed and set —
        deliberately NOT a field_validator on individual fields. A prior
        version used @field_validator("initial_guess") reading
        info.data.get("bounds"), which silently never fired: pydantic v2
        validates fields in declaration order, and "bounds" is declared
        AFTER "initial_guess" in this class, so info.data didn't contain
        it yet when that validator ran. Confirmed broken in practice —
        initial_guess=999999 against bounds=[-50,0] passed validation
        silently. A model-level validator has no such ordering dependency.
        """
        lo, hi = self.bounds
        if lo > hi:
            raise ValueError(f"bounds lower ({lo}) > upper ({hi})")
        if not (lo <= self.initial_guess <= hi):
            raise ValueError(f"initial_guess {self.initial_guess} falls outside bounds {self.bounds}")
        return self


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


class ThrusterDynamics(BaseModel):
    """Optional (PROJECT_STATUS.md #66): without it the thruster lag is held at 0."""
    thruster_time_constant_s: CalibrationParam


class CalibrationTargets(BaseModel):
    effective_inertia: EffectiveInertia
    linear_damping: LinearDamping
    quadratic_damping: QuadraticDamping
    added_mass_diag: AddedMassDiag
    buoyancy: Buoyancy
    thruster_dynamics: Optional[ThrusterDynamics] = None


class Vehicle(BaseModel):
    name: str


class Environment(BaseModel):
    current_velocity_mps: float = Field(ge=0.0)
    current_direction_deg: float
    current_vertical_mps: float = 0.0
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