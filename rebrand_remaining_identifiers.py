# 1. vehicle_spec.yaml comment
with open("calibration/vehicle_spec.yaml", encoding="utf-8") as f:
    spec_content = f.read()

old1 = "    BlueROV2 Heavy (8-thruster) vehicle spec for the marine-sim-mvp\n    calibration pipeline. Separates MEASURED/PUBLISHED constants"
new1 = "    BlueROV2 Heavy (8-thruster) vehicle spec for the Tryton\n    calibration pipeline. Separates MEASURED/PUBLISHED constants"

count1 = spec_content.count(old1)
print("vehicle_spec.yaml occurrences:", count1)
if count1 == 1:
    spec_content = spec_content.replace(old1, new1, 1)
    with open("calibration/vehicle_spec.yaml", "w", encoding="utf-8") as f:
        f.write(spec_content)
    print("vehicle_spec.yaml updated successfully.")
else:
    print("REFUSING vehicle_spec.yaml - count mismatch.")

# 2. mlflow_logging.py experiment name
with open("calibration/mlflow_logging.py", encoding="utf-8") as f:
    mlflow_content = f.read()

old2 = 'EXPERIMENT_NAME = "marine_sim_calibration"'
new2 = 'EXPERIMENT_NAME = "tryton_calibration"'

count2 = mlflow_content.count(old2)
print("mlflow_logging.py occurrences:", count2)
if count2 == 1:
    mlflow_content = mlflow_content.replace(old2, new2, 1)
    with open("calibration/mlflow_logging.py", "w", encoding="utf-8") as f:
        f.write(mlflow_content)
    print("mlflow_logging.py updated successfully.")
else:
    print("REFUSING mlflow_logging.py - count mismatch.")