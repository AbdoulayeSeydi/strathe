"""
A1.1, generic half — builds the un-fitted, "off-the-shelf" CalibrationParameters
for the generic-physics agent, straight from vehicle_spec.yaml's own
initial_guess values, no fitting involved.

Deliberately self-contained: load_vehicle_spec()/initial_parameters() in
optimizer.py never touch the data-loading side of the pipeline at all, so
this needs nothing beyond dynamics_model.py, optimizer.py, and
vehicle_spec.yaml — no objective.py, no data_split.py, no real BlueROV2
logs. Confirmed by this file's own import list, not just asserted.
"""
from optimizer import load_vehicle_spec, initial_parameters

VEHICLE_SPEC_PATH = "vehicle_spec.yaml"


def build_generic_parameters():
    loaded = load_vehicle_spec(VEHICLE_SPEC_PATH)
    return initial_parameters(loaded)


if __name__ == "__main__":
    params = build_generic_parameters()
    print("Generic (uncalibrated) CalibrationParameters — straight from vehicle_spec.yaml's initial_guess values:\n")
    for name in params.param_names:
        print(f"  {name:16s} = {getattr(params, name)}")