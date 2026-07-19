#!/usr/bin/env bash
# Downloads the oceansystemslab public BlueROV2 trajectory dataset.
# CSV columns (per repo README):
#   x,y,z                  - position, world frame
#   qx,qy,qz,qw             - orientation quaternion
#   Bu,Bv,Bw                - linear velocity, body frame
#   Bq,Bp,Br                - angular velocity, body frame
#   Ux,Uy,Uz,Vx,Vy,Vz       - normalized input force, body frame
#   Fx,Fy,Fz,Tx,Ty,Tz       - input forces/torques, body frame
#   pwm_1 ... pwm_8         - raw PWM to each of the 8 thrusters
set -euo pipefail

OUT_DIR="${1:-$(dirname "$0")/raw}"
mkdir -p "$OUT_DIR"

echo "== Cloning Bluerov2_trajectory_dataset =="
git clone https://github.com/oceansystemslab/Bluerov2_trajectory_dataset.git \
  "$OUT_DIR/Bluerov2_trajectory_dataset"

echo "== Files pulled: =="
find "$OUT_DIR/Bluerov2_trajectory_dataset" -name "*.csv"
