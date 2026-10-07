"""
fleet_experiment_schema.py -- the fleet v2 experiment config (YAML), validated.

Every field here reaches the simulation. (Fleet v1's run_fleet_experiment.py
validated task settings and comm overrides and then built the sweep from
the severity name alone: a changed target separation silently had no effect,
and any comm override crashed with a KeyError.)

    schema_version: 2
    name: my_experiment
    policies:                        # one or more per-vehicle policies (seeds)
      - {model_path: fleet_policy_seed1}             # PPO, SAC or TD3: read from the file
      - {model_path: my_sac_seed1, label: sac}       # optional label: policies with different labels
                                                     # are compared with each other in the report
    truth_params: calibrated_parameters_fleet_truth.yaml   # the "real" vehicle
    beliefs: [calibrated, generic, last_beacon, oracle]    # see BELIEF PRESETS
    compare: {a: calibrated, b: generic}                   # headline paired contrast
    communication:
      severities: [L2, L3a]          # named loss levels (one cell each), or:
      overrides: {near_field_prob: 0.45, far_field_prob: 0.45}  # one custom level
      beacon_interval_s: [1.0]       # seconds between state beacons (list = sweep)
      speed_of_sound_mps: 1500.0
    task: {target_separation_m: 2.0, tolerance_m: 0.2, hold_steps: 20,
           max_episode_steps: 200, collision_radius_m: 0.5}
    initial_conditions:
      separations_m: [2.0, 2.5, ..., 5.5]
      closing_speeds_mps: [-0.7, -0.35, 0.0, 0.35, 0.7]   # + = starting to separate
    evaluation: {comm_seeds: [1, 2, 3, 4, 5], workers: null}
    dynamics_backend: stand_in       # stand_in | mujoco | both
    mode: clean                      # clean | noisy | both. noisy = sensor noise (typical) + roll/pitch
                                     # disturbance (fitted) unless the blocks below say otherwise;
                                     # both = every condition run clean AND noisy, compared episode by episode
    sensor_noise:                    # optional; what each vehicle's own sensors report (sensor_noise.py)
      preset: typical                # none | tank | typical | path/to/fit_sensor_noise_output.yaml
      scale: 1.0                     # 0 = off, 2 = twice as noisy
    buoyancy:                        # optional; the true vehicles' net buoyancy vs the calibration (buoyancy.py)
      offset_n: -3.0                 # N; - = heavier (payload), + = floats up; or [-5, 5] drawn per episode
      known_to_beliefs: false        # physics teammate models know the change
    disturbance:                     # optional; default = clean simulation
      roll_pitch: fitted             # none | fitted | path/to/your_fit.yaml
      scale: 1.0                     # 0 = off, 1 = as fitted to real logs, 2 = twice as rough
    current:                         # optional water current; default = still water (tank)
      speed_mps: 0.2                 # or [0.0, 0.3]: drawn per episode
      direction_deg: 90              # direction the water flows toward, or [0, 360]
      vertical_mps: 0.0
      variability: {speed_std_mps: 0.05, direction_std_deg: 10, time_scale_s: 30}
      known_to_beliefs: true         # physics teammate models know the episode's mean current
    vehicle_spec: vehicle_spec.yaml  # optional: your vehicle's description (mass, inertia, bounds)
    thrusters: bluerov2_heavy        # optional: limit commands to what the thrusters can deliver;
                                     # bluerov2_heavy, an inline thrusters block, or a YAML file with
                                     # one (your ingest spec works). Default: no thruster limits.
    output_dir: results/my_experiment

BELIEF PRESETS -- how each vehicle predicts its teammate between beacons:
    calibrated         6-DOF physics with the truth parameters (a perfectly calibrated model)
    generic            6-DOF physics with vehicle_spec.yaml's uncalibrated defaults
    last_beacon        teammate assumed to stay where its last beacon said
    constant_velocity  last beacon extrapolated at its reported velocity
    oracle             true position (upper bound, not physically achievable)
    {physics_alpha: 0.5}            parameters 50% of the way from truth to generic
    {physics_params: my_fit.yaml, label: my_fit}   any calibration file of your own
"""
from __future__ import annotations

import os
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, PrivateAttr, ValidationError, field_validator, model_validator

from comm_channel import BandwidthConfig, PacketLossConfig
from evaluate_comm_degradation_curve import COMM_CONFIGS

SeverityLevel = Literal["L0", "L1", "L2", "L3a", "L3b"]
BELIEF_PRESETS = ("calibrated", "generic", "last_beacon", "constant_velocity", "oracle")
DEFAULT_SEPARATIONS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
DEFAULT_CLOSING_SPEEDS = [-0.7, -0.35, 0.0, 0.35, 0.7]


def _zip_path(p: str) -> str:
    return p if p.endswith(".zip") else f"{p}.zip"


class PolicyEntry(BaseModel):
    model_path: str
    algorithm: Optional[Literal["ppo", "sac", "td3"]] = None   # None = read from the file
    label: Optional[str] = None     # report group; policies with different labels are compared

    @model_validator(mode="after")
    def _exists(self):
        if not os.path.exists(_zip_path(self.model_path)):
            raise ValueError(f"policy file not found: {_zip_path(self.model_path)!r}")
        from policy_io import PolicyLoadError, resolve_algorithm
        try:
            self.algorithm = resolve_algorithm(_zip_path(self.model_path), self.algorithm)
        except PolicyLoadError as e:
            raise ValueError(str(e)) from e
        if self.label is not None and not self.label.strip():
            raise ValueError("a policy label must not be empty")
        return self

    @property
    def group(self) -> str:
        return self.label or self.algorithm


class PhysicsAlphaBelief(BaseModel):
    physics_alpha: float = Field(ge=0.0, le=1.0)
    label: Optional[str] = None


class PhysicsParamsBelief(BaseModel):
    physics_params: str
    label: str

    @model_validator(mode="after")
    def _exists(self):
        if not os.path.exists(self.physics_params):
            raise ValueError(f"belief parameter file not found: {self.physics_params!r}")
        return self


BeliefEntry = Union[Literal["calibrated", "generic", "last_beacon", "constant_velocity", "oracle"],
                    PhysicsAlphaBelief, PhysicsParamsBelief]


def belief_label(b) -> str:
    if isinstance(b, str):
        return b
    if isinstance(b, PhysicsAlphaBelief):
        return b.label or f"physics_alpha_{b.physics_alpha:g}"
    return b.label


class CompareConfig(BaseModel):
    a: str = "calibrated"
    b: str = "generic"


class CommOverrides(BaseModel):
    near_distance_m: Optional[float] = None
    far_distance_m: Optional[float] = None
    near_field_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    far_field_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    bytes_per_tick: Optional[float] = Field(default=None, gt=0)

    def any_set(self) -> bool:
        return any(getattr(self, f) is not None for f in type(self).model_fields)


LinkName = Literal["acoustic", "optical", "hybrid"]


class OpticalTerminalBlock(BaseModel):
    axis_body: list[float] = Field(min_length=3, max_length=3)
    beam_half_angle_deg: float = 180.0
    receiver_half_fov_deg: float = 180.0


class OpticalBlock(BaseModel):
    """An optical modem pair (comm_channel.OpticalLinkConfig). UNCALIBRATED: rough water values and a
    placeholder delivery curve. Angles are HALF-angles (180 = every direction). water may be a list (a sweep).
    compare_omni: also run every optical/hybrid condition with omnidirectional terminals, same episodes, so
    the report can show what pointing costs."""
    water: Union[str, list[str]] = "coastal"
    attenuation_per_m: Optional[float] = None
    terminals: Union[str, list[OpticalTerminalBlock]] = "omni"
    beam_half_angle_deg: Optional[float] = None
    receiver_half_fov_deg: Optional[float] = None
    data_rate_bps: float = 1.0e6
    latency_s: float = 0.001
    rated_range_m: float = 30.0
    curve_width: float = 1.0
    beacon_interval_s: float = 0.1
    compare_omni: bool = False
    calibrated_from: Optional[str] = None   # set by communication.link_calibration (#68)

    @model_validator(mode="after")
    def _valid(self):
        if isinstance(self.water, list) and not self.water:
            raise ValueError("communication.optical.water must not be empty")
        if abs(self.beacon_interval_s / 0.1 - round(self.beacon_interval_s / 0.1)) > 1e-9 or self.beacon_interval_s <= 0:
            raise ValueError(f"optical beacon_interval_s must be a positive multiple of the 0.1 s tick, "
                             f"got {self.beacon_interval_s}")
        cfgs = self.configs()
        if self.compare_omni and all(c.omni for c in cfgs):
            raise ValueError("compare_omni compares pointing cones with omnidirectional terminals, but these "
                             "terminals are already omnidirectional; set terminals to a cone preset")
        return self

    def waters(self) -> list:
        return list(self.water) if isinstance(self.water, list) else [self.water]

    def configs(self, omni: bool = False) -> list:
        """One validated OpticalLinkConfig per water type (omni=True: the omnidirectional control)."""
        from comm_channel import OpticalLinkConfig
        terms = (self.terminals if isinstance(self.terminals, str)
                 else [t.model_dump() for t in self.terminals])
        out = []
        for w in self.waters():
            c = OpticalLinkConfig(water=w, attenuation_per_m=self.attenuation_per_m if w == "custom" else None,
                                  terminals="omni" if omni else terms,
                                  beam_half_angle_deg=None if omni else self.beam_half_angle_deg,
                                  receiver_half_fov_deg=None if omni else self.receiver_half_fov_deg,
                                  data_rate_bps=self.data_rate_bps, latency_s=self.latency_s,
                                  rated_range_m=self.rated_range_m, curve_width=self.curve_width,
                                  beacon_interval_s=self.beacon_interval_s, calibrated_from=self.calibrated_from)
            c.validate()
            out.append(c)
        return out


class CommunicationConfig(BaseModel):
    severities: Optional[list[SeverityLevel]] = None
    severity: Optional[SeverityLevel] = None       # single-level shorthand
    overrides: CommOverrides = Field(default_factory=CommOverrides)
    beacon_interval_s: Union[float, list[float]] = 1.0
    speed_of_sound_mps: float = Field(default=1500.0, gt=0)
    # acoustic (default) | optical | hybrid, or a list to compare them on the same episodes
    link: Union[LinkName, list[LinkName]] = "acoustic"
    optical: Optional[OpticalBlock] = None
    hybrid_policy: Literal["optical_first", "redundant"] = "optical_first"
    link_up_ticks: int = Field(default=1, ge=1, le=600)   # optical_first: "up" = heard within this many ticks
    # a link_calibration.yaml from `strathe fit-link` (#68): its acoustic fit sets the loss ramp (overrides),
    # speed_of_sound_mps and acoustic_fixed_latency_s; its optical fit sets the optical water, curve and latency
    link_calibration: Optional[str] = None
    acoustic_fixed_latency_s: float = Field(default=0.0, ge=0.0, le=60.0)
    _acoustic_calibrated: bool = PrivateAttr(default=False)

    def acoustic_calibrated(self) -> bool:
        return self._acoustic_calibrated

    def _apply_link_calibration(self) -> None:
        from link_fit import load_link_calibration
        cal = load_link_calibration(self.link_calibration)
        self.link_calibration = cal["file"]
        links = self.links()
        if "acoustic" in cal and any(l in ("acoustic", "hybrid") for l in links):
            if self.severities and len(self.severities) > 1:
                raise ValueError("link_calibration replaces the assumed loss levels: give at most one severity "
                                 "(only its bandwidth is kept), or none")
            plc = cal["acoustic"]["packet_loss_config"]
            for k in ("near_distance_m", "far_distance_m", "near_field_prob", "far_field_prob"):
                setattr(self.overrides, k, getattr(plc, k))
            self.speed_of_sound_mps = cal["acoustic"]["speed_of_sound_mps"]
            self.acoustic_fixed_latency_s = cal["acoustic"]["fixed_latency_s"]
            self._acoustic_calibrated = True
        if "optical" in cal and any(l in ("optical", "hybrid") for l in links):
            o = cal["optical"]
            if self.optical is None:
                self.optical = OpticalBlock()
            self.optical.water = "custom"
            self.optical.attenuation_per_m = o["attenuation_per_m"]
            self.optical.rated_range_m = o["rated_range_m"]
            self.optical.curve_width = o["curve_width"]
            self.optical.latency_s = o["latency_s"]
            self.optical.calibrated_from = cal["file"]

    def links(self) -> list:
        return list(dict.fromkeys(self.link)) if isinstance(self.link, list) else [self.link]

    def link_configs(self) -> list:
        """[(LinkConfig, pointing)] for every link x water (x the omni control); pointing is the terminal
        preset's label, or 'omni'."""
        from comm_channel import LinkConfig
        out = []
        for name in self.links():
            if name == "acoustic":
                out.append((LinkConfig("acoustic"), "acoustic"))
                continue
            variants = [False] + ([True] if self.optical.compare_omni else [])
            for omni in variants:
                for oc in self.optical.configs(omni=omni):
                    lc = LinkConfig(name, oc, self.hybrid_policy, self.link_up_ticks)
                    lc.validate()
                    out.append((lc, "omni" if oc.omni else oc.pointing_label))
        return out

    @model_validator(mode="after")
    def _resolve(self):
        if self.severity is not None and self.severities is not None:
            raise ValueError("give either communication.severity or communication.severities, not both")
        if self.severity is not None:
            self.severities = [self.severity]
        links = self.links()
        if not links:
            raise ValueError("communication.link must not be empty")
        if self.link_calibration:
            self._apply_link_calibration()
        uses_optical = any(l in ("optical", "hybrid") for l in links)
        if uses_optical and self.optical is None:
            # written into the config itself so the saved experiment.yaml reruns the same thing
            self.optical = OpticalBlock()
        if not uses_optical and self.optical is not None:
            raise ValueError("communication.optical is set but link is acoustic; set link: optical or hybrid "
                             "(or a list, e.g. [acoustic, hybrid])")
        if self.overrides.any_set() and self.severities and len(self.severities) > 1:
            raise ValueError("overrides apply on top of ONE severity; list a single severity (or none) with overrides")
        if not self.severities and not self.overrides.any_set():
            if links == ["optical"]:
                self.severities = ["L0"]   # unused: an optical-only link has no acoustic channel
            else:
                raise ValueError("communication needs at least one severity (L0-L3b) or an overrides block")
        intervals = self.beacon_interval_s if isinstance(self.beacon_interval_s, list) else [self.beacon_interval_s]
        if not intervals:
            raise ValueError("communication.beacon_interval_s must not be empty")
        for b in intervals:
            if b <= 0 or abs(b / 0.1 - round(b / 0.1)) > 1e-9:
                raise ValueError(f"beacon_interval_s must be a positive multiple of the 0.1 s tick, got {b}")
        self.beacon_interval_s = intervals
        for _, plc, _ in self.levels():
            plc.validate()
        self.link_configs()   # raises ValueError with a specific message
        return self

    def levels(self) -> list:
        """[(label, PacketLossConfig, BandwidthConfig)] -- fully resolved."""
        if not self.overrides.any_set():
            return [(s, COMM_CONFIGS[s]["packet_loss_config"], COMM_CONFIGS[s]["bandwidth_config"])
                    for s in self.severities]
        base_sev = self.severities[0] if self.severities else None
        base = COMM_CONFIGS[base_sev] if base_sev else {
            "packet_loss_config": PacketLossConfig(0.0, 1000.0, 0.0, 0.0),
            "bandwidth_config": BandwidthConfig(1e9)}
        bp, ov = base["packet_loss_config"], self.overrides

        def pick(v, d):
            return d if v is None else v
        plc = PacketLossConfig(pick(ov.near_distance_m, bp.near_distance_m), pick(ov.far_distance_m, bp.far_distance_m),
                               pick(ov.near_field_prob, bp.near_field_prob), pick(ov.far_field_prob, bp.far_field_prob))
        bwc = BandwidthConfig(ov.bytes_per_tick) if ov.bytes_per_tick is not None else base["bandwidth_config"]
        label = "measured" if self._acoustic_calibrated else (f"{base_sev}_custom" if base_sev else "custom")
        return [(label, plc, bwc)]


class TaskConfig(BaseModel):
    target_separation_m: float = Field(default=2.0, gt=0)
    tolerance_m: float = Field(default=0.2, gt=0)
    hold_steps: int = Field(default=20, gt=0)
    max_episode_steps: int = Field(default=200, gt=0)
    collision_radius_m: float = Field(default=0.5, gt=0)

    @model_validator(mode="after")
    def _consistent(self):
        if self.hold_steps > self.max_episode_steps:
            raise ValueError(f"hold_steps ({self.hold_steps}) > max_episode_steps ({self.max_episode_steps}): "
                             f"success would be impossible")
        if self.target_separation_m - self.tolerance_m <= self.collision_radius_m:
            raise ValueError("target_separation_m - tolerance_m must exceed collision_radius_m, "
                             "or holding formation would count as a collision")
        return self


class InitialConditions(BaseModel):
    separations_m: list[float] = Field(default_factory=lambda: list(DEFAULT_SEPARATIONS))
    closing_speeds_mps: list[float] = Field(default_factory=lambda: list(DEFAULT_CLOSING_SPEEDS))

    @field_validator("separations_m", "closing_speeds_mps")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("must list at least one value")
        return v


class EvaluationConfig(BaseModel):
    comm_seeds: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5], min_length=1)
    workers: Optional[int] = Field(default=None, ge=1)


class DisturbanceBlock(BaseModel):
    """Random roll/pitch torques fitted to the motion real logs show but the
    thrusters don't explain (disturbance.py). 'fitted' = the shipped fit for
    the BlueROV2 fleet truth (disturbances/bluerov2_rollpitch.yaml)."""
    roll_pitch: str = "none"
    scale: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def _exists(self):
        if self.roll_pitch not in ("none", "fitted") and not os.path.exists(self.roll_pitch):
            raise ValueError(f"disturbance.roll_pitch must be 'none', 'fitted' or an existing file, "
                             f"got {self.roll_pitch!r}")
        return self

    def path(self) -> Optional[str]:
        if self.roll_pitch == "none" or self.scale == 0:
            return None
        from disturbance import DEFAULT_FIT_PATH
        return DEFAULT_FIT_PATH if self.roll_pitch == "fitted" else self.roll_pitch


class SensorNoiseBlock(BaseModel):
    """What each vehicle's own sensors report (sensor_noise.py). preset: none, tank (fitted to the reference
    tank logs), typical (a DVL + AHRS dead-reckoning vehicle; stated assumptions), open_water (measured on a
    BlueROV2 at a fish farm, SOLAQUA; biases assumed) or a file from fit_sensor_noise.py. start_fix_error_m:
    each vehicle's starting position fix error, set on its own (PROJECT_STATUS.md #60)."""
    preset: str = "none"
    scale: float = Field(default=1.0, ge=0.0, le=10.0)
    # error of each vehicle's starting position fix (m per axis, dead reckoning), set on its own; None = the preset's
    start_fix_error_m: Optional[float] = Field(default=None, ge=0.0, le=20.0)

    @model_validator(mode="after")
    def _valid(self):
        if self.preset != "none" or self.start_fix_error_m is None:
            self.config()
        return self

    def config(self):
        from sensor_noise import resolve_sensor_noise
        return resolve_sensor_noise(self.preset, self.scale, self.start_fix_error_m)


class BuoyancyBlock(BaseModel):
    """Net-buoyancy change of the true vehicles from the calibration (buoyancy.py), N."""
    offset_n: Union[float, list[float]] = 0.0
    known_to_beliefs: bool = False

    @model_validator(mode="after")
    def _valid(self):
        self.config()
        return self

    def config(self):
        from buoyancy import BuoyancyConfig
        return BuoyancyConfig.from_dict(self.model_dump())


class CurrentVariability(BaseModel):
    speed_std_mps: float = Field(default=0.0, ge=0.0)
    direction_std_deg: float = Field(default=0.0, ge=0.0)
    time_scale_s: float = Field(default=30.0, ge=1.0)


class CurrentBlock(BaseModel):
    """Water current (current.py). Absent = still water. A number or a
    [min, max] pair for speed and direction; pairs are drawn per episode."""
    speed_mps: Union[float, list[float]] = 0.0
    direction_deg: Union[float, list[float]] = 0.0
    vertical_mps: float = 0.0
    variability: CurrentVariability = Field(default_factory=CurrentVariability)
    known_to_beliefs: bool = True

    @model_validator(mode="after")
    def _valid(self):
        self.to_config()   # raises ValueError with a specific message
        return self

    def to_config(self):
        from current import CurrentConfig
        def rng(x):
            if isinstance(x, list):
                if len(x) != 2:
                    raise ValueError(f"a range must be [min, max], got {x}")
                return (float(x[0]), float(x[1]))
            return float(x)
        cfg = CurrentConfig(speed_mps=rng(self.speed_mps), direction_deg=rng(self.direction_deg),
                            vertical_mps=self.vertical_mps, speed_std_mps=self.variability.speed_std_mps,
                            direction_std_deg=self.variability.direction_std_deg,
                            time_scale_s=self.variability.time_scale_s, known_to_beliefs=self.known_to_beliefs)
        cfg.validate()
        return cfg


class FleetExperimentConfig(BaseModel):
    schema_version: Literal[2]
    name: str = ""
    policies: list[PolicyEntry] = Field(min_length=1)
    truth_params: str = "calibrated_parameters_fleet_truth.yaml"
    beliefs: list[BeliefEntry] = Field(default_factory=lambda: ["calibrated", "generic", "last_beacon", "oracle"],
                                       min_length=1)
    compare: Optional[CompareConfig] = Field(default_factory=CompareConfig)
    communication: CommunicationConfig
    task: TaskConfig = Field(default_factory=TaskConfig)
    initial_conditions: InitialConditions = Field(default_factory=InitialConditions)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    dynamics_backend: Literal["stand_in", "mujoco", "both"] = "stand_in"
    mode: Literal["clean", "noisy", "both"] = "clean"
    sensor_noise: Optional[SensorNoiseBlock] = None
    buoyancy: Optional[BuoyancyBlock] = None
    disturbance: DisturbanceBlock = Field(default_factory=DisturbanceBlock)
    current: Optional[CurrentBlock] = None
    vehicle_spec: str = "vehicle_spec.yaml"
    thrusters: Optional[Union[str, dict]] = None
    output_dir: str

    def thruster_config(self):
        """The thrusters in simulation form (torques about the model's origin), or None."""
        return resolve_thrusters(self.thrusters)

    def modes(self) -> list:
        return ["clean", "noisy"] if self.mode == "both" else [self.mode]

    def variant(self, mode: str) -> tuple:
        """(SensorNoiseConfig or None, disturbance file or None, disturbance scale) for one mode of this run.
        With mode: both, the clean half has neither; everything else (current, thrusters, buoyancy) is shared."""
        if self.mode == "both" and mode == "clean":
            return None, None, 0.0
        noise = self.sensor_noise.config() if self.sensor_noise is not None else None
        return noise, self.disturbance.path(), self.disturbance.scale

    def buoyancy_config(self):
        return self.buoyancy.config() if self.buoyancy is not None else None

    @model_validator(mode="after")
    def _cross_checks(self):
        if self.mode in ("noisy", "both"):
            # noisy's defaults, written into the config itself so the saved experiment.yaml reruns the same thing
            if self.sensor_noise is None:
                self.sensor_noise = SensorNoiseBlock(preset="typical")
            elif "preset" not in self.sensor_noise.model_fields_set:
                # e.g. only start_fix_error_m given: the noisy mode's default preset, with that starting fix
                self.sensor_noise = SensorNoiseBlock(preset="typical", scale=self.sensor_noise.scale,
                                                     start_fix_error_m=self.sensor_noise.start_fix_error_m)
            if "disturbance" not in self.model_fields_set:
                self.disturbance = DisturbanceBlock(roll_pitch="fitted")
        if (self.mode == "clean" and self.sensor_noise is not None and self.sensor_noise.preset == "none"
                and self.sensor_noise.start_fix_error_m is not None):
            raise ValueError("sensor_noise.start_fix_error_m needs sensor noise on: set mode: noisy (or both), or a "
                             "sensor_noise preset")
        if not os.path.exists(self.truth_params):
            raise ValueError(f"truth_params file not found: {self.truth_params!r}")
        if not os.path.exists(self.vehicle_spec):
            raise ValueError(f"vehicle_spec file not found: {self.vehicle_spec!r}")
        try:
            from optimizer import load_vehicle_spec
            load_vehicle_spec(self.vehicle_spec)
        except Exception as e:
            raise ValueError(f"vehicle_spec {self.vehicle_spec!r} is invalid: {e}") from e
        if self.thrusters is not None:
            resolve_thrusters(self.thrusters)  # raises ValueError with a specific message
        labels = [belief_label(b) for b in self.beliefs]
        if len(set(labels)) != len(labels):
            raise ValueError(f"belief labels must be unique, got {labels}")
        if self.compare is not None:
            for side in (self.compare.a, self.compare.b):
                if side not in labels:
                    raise ValueError(f"compare refers to belief {side!r}, which is not in beliefs {labels}")
        too_close = [d for d in self.initial_conditions.separations_m if d <= self.task.collision_radius_m]
        if too_close:
            raise ValueError(f"initial separations {too_close} are inside collision_radius_m "
                             f"({self.task.collision_radius_m}); episodes would start collided")
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            raise ValueError(f"output_dir {self.output_dir!r} can't be created: {e}") from e
        return self


class ConfigError(ValueError):
    pass


def resolve_thrusters(block):
    """thrusters: field -> thruster_model.ThrusterConfig for the simulators, or None.
    'bluerov2_heavy' is shorthand for the preset; a string ending in .yaml/.yml is a file holding a
    thrusters block (an ingest spec works); a mapping is an inline block."""
    if block is None:
        return None
    import thruster_model as tm
    if isinstance(block, str) and not block.endswith((".yaml", ".yml")):
        block = {"preset": block}
    elif isinstance(block, str) and not os.path.exists(block):
        raise ValueError(f"thrusters file not found: {block!r}")
    try:
        return tm.thruster_block(block).for_simulation()
    except (OSError, KeyError) as e:
        raise ValueError(f"thrusters: {e}") from e


def load_fleet_experiment(path: str) -> FleetExperimentConfig:
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"experiment config not found: {path!r}")
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parse error in {path!r}: {e}")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path!r} is not a YAML mapping")
    if "schema_version" not in raw and "vehicles" in raw:
        raise ConfigError(
            f"{path!r} is a fleet v1 config (it has 'vehicles', no 'schema_version'). Fleet v2 replaced the "
            f"centralized two-policy setup with one per-vehicle policy and a teammate-belief model; start from "
            f"default_experiment.yaml (see CUSTOM_EXPERIMENT.md). v1 is still runnable with run_fleet_experiment_v1.py.")
    try:
        return FleetExperimentConfig(**raw)
    except ValidationError as e:
        raise ConfigError(f"invalid experiment config {path!r}:\n{e}") from e
