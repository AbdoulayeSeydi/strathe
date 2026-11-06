"""
simfit/fin_models.py -- a vehicle description's fins, written in each simulator's own terms (PROJECT_STATUS.md #97, FINS.md).

Gazebo: a link on a revolute joint fin_<name> with a LiftDrag plugin. MuJoCo: a child body on a hinge fin_<name> with a thin ellipsoid
geom. Stonefish: a rudder actuator fin_<name> on the base link. The replay tools hold each at the logged angle, and the simulator's own
physics makes the force. Everything is at the fin's centre of pressure; axes convert forward-right-down to forward-left-up.
"""
from __future__ import annotations

import math

import numpy as np

from fin_force import Actuators, Fin


def _flu(v) -> tuple:
    return float(v[0]), float(-v[1]), float(-v[2])


def _slope(f: Fin, params) -> float:
    return float(params.fin_lift_slope_b if f.group else params.fin_lift_slope_a) if params is not None else 3.0


def _rpy_to_span(span: tuple) -> tuple:
    """Roll about x taking a body z axis onto the fin's span, in the same axes: every fin's chord stays along x."""
    return math.atan2(-span[1], span[2]), 0.0, 0.0


def sdf_fins(act: Actuators, base_link: str, rho: float, params=None) -> tuple:
    """(links and joints, plugins) as SDF text. A LiftDrag's upward is the fin's lift direction: span x chord."""
    links, plugins = [], []
    for f in act.fins:
        x, y, z = _flu(f.position_m)
        s = _flu(f.span_axis)
        n = np.cross(s, (1.0, 0.0, 0.0))
        links.append(f"""      <link name="fin_{f.name}_link"><pose>{x:.6g} {y:.6g} {z:.6g} 0 0 0</pose>
        <inertial><mass>0.01</mass><inertia><ixx>1e-5</ixx><iyy>1e-5</iyy><izz>1e-5</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      </link>
      <joint name="fin_{f.name}" type="revolute"><parent>{base_link}</parent><child>fin_{f.name}_link</child>
        <axis><xyz>{s[0]:.6g} {s[1]:.6g} {s[2]:.6g}</xyz><limit><lower>{-f.limit_rad:.6g}</lower><upper>{f.limit_rad:.6g}</upper></limit></axis>
      </joint>""")
        plugins.append(f"""      <plugin filename="gz-sim-lift-drag-system" name="gz::sim::systems::LiftDrag">
        <air_density>{rho:.6g}</air_density><a0>0</a0><cla>{_slope(f, params):.6g}</cla><cla_stall>-1.1</cla_stall><cda>0.02</cda><cda_stall>0.03</cda_stall>
        <area>{f.area_m2:.6g}</area><alpha_stall>{f.stall_rad:.6g}</alpha_stall><forward>1 0 0</forward><upward>{n[0]:.6g} {n[1]:.6g} {n[2]:.6g}</upward>
        <cp>0 0 0</cp><link_name>fin_{f.name}_link</link_name>
      </plugin>""")
    return "\n".join(links), "\n".join(plugins)


def mjcf_fins(act: Actuators) -> str:
    out = []
    for f in act.fins:
        x, y, z = _flu(f.position_m)
        s = _flu(f.span_axis)
        chord = f.area_m2 / f.span_m
        roll = math.atan2(s[2], s[1]) - math.pi / 2        # the ellipsoid's long axis along the span
        out.append(f"""      <body name="fin_{f.name}" pos="{x:.6g} {y:.6g} {z:.6g}">
        <joint name="fin_{f.name}" type="hinge" axis="{s[0]:.6g} {s[1]:.6g} {s[2]:.6g}"/>
        <geom name="fin_{f.name}_geom" type="ellipsoid" size="{chord / 2:.6g} {f.span_m / 2:.6g} 0.004" euler="{math.degrees(roll):.6g} 0 0" mass="0.01"
              fluidshape="ellipsoid" fluidcoef="0.5 0.25 1.5 2.0 0" contype="0" conaffinity="0"/>
      </body>""")
    return "\n".join(out)


def stonefish_fins(act: Actuators, link: str, mesh: str = "icosphere.obj", material: str = "Neutral", look: str = "gray", params=None) -> str:
    out = []
    for f in act.fins:
        x, y, z = f.position_m
        r, _, _ = _rpy_to_span(f.span_axis)
        out.append(f"""  <actuator name="fin_{f.name}" type="rudder">
   <specs lift_coeff="{_slope(f, params):.6g}" drag_coeff="0.1" max_angle="{f.limit_rad:.6g}" area="{f.area_m2:.6g}" stall_angle="{f.stall_rad:.6g}" max_angular_rate="{act.servo_rate_rad_s:.6g}" inverted="false"/>
   <visual><mesh filename="{mesh}" scale="0.01"/><material name="{material}"/><look name="{look}"/><origin rpy="0.0 0.0 0.0" xyz="0.0 0.0 0.0"/></visual>
   <origin rpy="{r:.6g} 0.0 0.0" xyz="{x:.6g} {y:.6g} {z:.6g}"/>
   <link name="{link}"/>
  </actuator>""")
    return "\n".join(out)
