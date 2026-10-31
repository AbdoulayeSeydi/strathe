"""`strathe describe-vehicle` (vehicle_describe.py, PROJECT_STATUS.md #95): a vehicle description from its mass and
size, or from its model in Gazebo, URDF, MuJoCo or HoloOcean. The real-data proof (calibrating each generated spec
against the hand-written one) is in STRESS_TEST_93_95.md."""
import os

import pytest
import yaml

import strathe_cli as cli
import vehicle_describe as VD
from optimizer import load_vehicle_spec

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
BLUE = dict(mass=11.5, length=0.457, width=0.575, height=0.254)
ALPHA = dict(mass=33.0, length=1.5, diameter=0.254)


def _t(spec):
    return {k: v for g in spec["calibration_targets"].values() for k, v in g.items() if isinstance(v, dict) and "bounds" in v}


def test_a_frame_and_a_torpedo_get_their_own_rules():
    blue, _ = VD.describe_hull(**BLUE, rho=1000)
    alpha, notes = VD.describe_hull(**ALPHA)
    assert "box" in blue["metadata"]["description"] and "torpedo" in alpha["metadata"]["description"]
    for spec in (blue, alpha):
        for k, v in _t(spec).items():
            lo, hi = v["bounds"]
            assert lo < hi and lo <= v["initial_guess"] <= hi, k
    # the hand-written BlueROV2 description's starting values sit inside the limits sized from its size
    hand = _t(yaml.safe_load(open(os.path.join(CAL, "vehicle_spec.yaml"))))
    tb = _t(blue)
    inside = [k for k in tb if k in hand and tb[k]["bounds"][0] <= hand[k]["initial_guess"] <= tb[k]["bounds"][1]]
    assert len(inside) >= len([k for k in tb if k in hand]) - 2
    assert "hull_lift" not in alpha["calibration_targets"] and any("hull-lift" in n for n in notes)
    assert "hull_lift" in VD.describe_hull(**ALPHA, hull_lift=True)[0]["calibration_targets"]


def test_a_shape_the_size_does_not_suggest_is_warned_about():
    """A BlueROV2 described as a torpedo lost 0.077 held-out skill, with two values cut off (#95)."""
    for dims, wrong, right in ((BLUE, "torpedo", "box"), (ALPHA, "box", "torpedo")):
        spec, notes = VD.describe_hull(**dims, shape=wrong)
        assert wrong in spec["metadata"]["description"]   # still done as asked
        assert any("goes against the size" in n and f"reads as a {right}" in n for n in notes)
        for shape in (right, None):
            assert not any("goes against the size" in n for n in VD.describe_hull(**dims, shape=shape)[1])


def test_the_limits_follow_the_size_mass_and_water():
    base = _t(VD.describe_hull(**BLUE)[0])
    wider = _t(VD.describe_hull(**{**BLUE, "width": 0.575 * 1.3})[0])
    heavier = _t(VD.describe_hull(**{**BLUE, "mass": 11.5 * 1.3})[0])
    fresh = _t(VD.describe_hull(**BLUE, rho=1000)[0])
    assert wider["Xuu"]["bounds"][0] < base["Xuu"]["bounds"][0]          # a bigger face: more drag allowed
    assert heavier["Ieff_roll"]["bounds"][1] > base["Ieff_roll"]["bounds"][1]
    assert heavier["Xu_dot"]["bounds"][1] > base["Xu_dot"]["bounds"][1]
    assert fresh["Xuu"]["bounds"][0] > base["Xuu"]["bounds"][0]          # less dense water, less drag
    longer = _t(VD.describe_hull(**{**ALPHA, "length": 1.5 * 1.3})[0])
    assert longer["Yv_dot"]["bounds"][1] > _t(VD.describe_hull(**ALPHA)[0])["Yv_dot"]["bounds"][1]


def test_what_you_know_is_used():
    s, _ = VD.describe_hull(**BLUE, inertia=(0.2, 0.3, 0.4), cg_below_cb=0.03, net_buoyancy=5.0,
                            torque_reference_offset=(0, 0, -0.085))
    t = _t(s)
    box_roll = 11.5 * (0.575 ** 2 + 0.254 ** 2) / 12
    assert t["Ieff_roll"]["initial_guess"] == pytest.approx(0.2 + 0.75 * box_roll, rel=1e-3)   # yours, plus the water
    # (the water a frame carries depends on its shape, not on where its mass sits, so it stays the size estimate)
    assert t["cg_cb_offset_z"]["initial_guess"] == pytest.approx(0.03)
    assert t["net_buoyancy_z"]["initial_guess"] == pytest.approx(5.0)
    assert s["data_conventions"]["torque_reference_offset_m"] == [0.0, 0.0, -0.085]
    _, notes = VD.describe_hull(**BLUE, max_force=[200.0, 200.0, 200.0])
    assert notes == []
    _, notes = VD.describe_hull(**BLUE, max_force=[50000.0, 1.0, 1.0])
    assert any("faster than" in n for n in notes)


def test_plain_refusals():
    for kw, msg in ((dict(mass=10, length=1), "give the vehicle's size"), (dict(mass=0, length=1, diameter=0.2), "above 0"),
                    (dict(mass=10, length=1, width=-1, height=0.2), "above 0"),
                    (dict(**ALPHA, shape="sphere"), "box or torpedo"),
                    (dict(**ALPHA, inertia=(1, 0, 1)), "three values above 0")):
        with pytest.raises(VD.DescribeError, match=msg):
            VD.describe_hull(**kw)


def test_from_each_simulators_model(tmp_path):
    gz = VD.from_model(os.path.join(CAL, "simulators", "gazebo", "example", "world.sdf"), "gazebo")
    mj = VD.from_model(os.path.join(CAL, "simulators", "mujoco", "example", "auv.xml"), "mujoco")
    assert gz["mass"] == pytest.approx(10.0) and gz["extents"]
    assert mj["mass"] == pytest.approx(10.0) and mj["extents"] == pytest.approx((0.5, 0.3, 0.16), rel=1e-6)
    urdf = tmp_path / "r.urdf"
    urdf.write_text('<robot name="r"><link name="base"><inertial><mass value="20"/>'
                    '<inertia ixx="1" iyy="2" izz="3" ixy="0" ixz="0" iyz="0"/></inertial>'
                    '<collision><origin xyz="0 0 0" rpy="0 1.5707963 0"/><geometry><cylinder radius="0.1" length="1.2"/>'
                    '</geometry></collision></link></robot>')
    u = VD.from_model(str(urdf), "urdf")
    assert u["mass"] == 20 and u["inertia"] == [1, 2, 3]
    assert u["extents"] == pytest.approx((1.2, 0.2, 0.2), abs=1e-6)   # the cylinder turned to lie along x
    ho = VD.from_model(os.path.join(CAL, "simulators", "holoocean", "example", "scenario.json"), "holoocean")
    assert ho["mass"] == 30.0 and ho["extents"] is None
    with pytest.raises(VD.DescribeError, match="--sim is gazebo"):
        VD.from_model(str(urdf), "isaac")


def test_the_command_writes_a_description_every_command_loads(tmp_path, capsys):
    out = tmp_path / "s.yaml"
    assert VD.main([*sum(([f"--{k}", str(v)] for k, v in BLUE.items()), []), "--out", str(out)]) == 0
    load_vehicle_spec(str(out))
    m = os.path.join(CAL, "simulators", "gazebo", "example", "world.sdf")
    assert VD.main(["--from-model", m, "--out", str(tmp_path / "g.yaml")]) == 0   # .sdf: Gazebo
    assert VD.main(["--from-model", os.path.join(CAL, "simulators", "holoocean", "example", "scenario.json"),
                    "--out", str(tmp_path / "h.yaml")]) == 2
    assert "no length and diam" in capsys.readouterr().out
    assert VD.main(["--mass", "10", "--length", "1", "--diameter", "0.2", "--water", "salty",
                    "--out", str(tmp_path / "w.yaml")]) == 2


def test_a_hull_sized_only_to_float_the_vehicle_is_flagged(tmp_path, capsys):
    """URI's ALPHA RISE in Gazebo: a 0.17 m cylinder displacing its 33 kg; the real hull is 0.254 m (#95)."""
    def urdf(mass):
        p = tmp_path / f"r{mass}.urdf"
        p.write_text(f'<robot name="r"><link name="base"><inertial><mass value="{mass}"/>'
                     '<inertia ixx="1" iyy="2" izz="3" ixy="0" ixz="0" iyz="0"/></inertial>'
                     '<collision><origin xyz="0 0 0" rpy="0 1.5707963 0"/><geometry><cylinder radius="0.1" length="1.2"/>'
                     '</geometry></collision></link></robot>')
        return str(p)
    for mass, more, flagged in ((38.7, [], True), (20, [], False), (38.7, ["--length", "1.5", "--diameter", "0.3"], False)):
        assert VD.main(["--from-model", urdf(mass), "--sim", "urdf", *more, "--out", str(tmp_path / "u.yaml")]) == 0
        assert ("sized to float the vehicle" in capsys.readouterr().out) == flagged


def test_strathe_and_the_web_page_build_it(tmp_path):
    import strathe_web as web
    calls = []
    cli.main(["describe-vehicle", "--mass", "11.5", "--length", "0.457", "--width", "0.575", "--height", "0.254",
              "--hull-lift", "--out", "rel.yaml"], runner=lambda c: calls.append(c) or 0)
    argv = calls[0]
    assert os.path.basename(argv[1]) == "vehicle_describe.py" and "--hull-lift" in argv
    assert os.path.isabs(argv[argv.index("--out") + 1]) and argv[argv.index("--mass") + 1] == "11.5"
    a = web.describe_vehicle_cli_args({"mass": "11.5", "length": "0.457", "width": "0.575", "height": "0.254",
                                       "water": "fresh", "shape": "box"})
    assert a[a.index("--water") + 1] == "fresh" and a[a.index("--shape") + 1] == "box"
    with pytest.raises(ValueError, match="mass and length"):
        web.describe_vehicle_cli_args({})
    with pytest.raises(ValueError, match="Gazebo, URDF"):
        web.describe_vehicle_cli_args({"from_model": os.path.join(CAL, "simulators", "mujoco", "example", "auv.xml")})


@pytest.mark.parametrize("kw", [dict(mass=0.075, length=0.05, width=0.05, height=0.03),     # a 5 cm, 75 g frame
                                dict(mass=0.43, length=0.3, diameter=0.043),                # a 30 cm torpedo
                                dict(mass=0.014, length=0.137, diameter=0.0137)])           # 14 g: inertia ~1e-5
def test_a_small_hull_gets_limits_in_order_and_a_description_that_loads(tmp_path, kw):
    """The fixed upper damping limits (-0.05, -0.2) used to sit below a small hull's size-scaled lower ones, and its
    inertia rounded to 0: the description didn't load."""
    spec, _ = VD.describe_hull(**kw)
    for k, v in _t(spec).items():
        lo, hi = v["bounds"]
        assert lo < hi and lo <= v["initial_guess"] <= hi, k
        if lo < 0:
            assert hi >= 0.5 * lo, k   # at least 2:1 wide
    assert all(_t(spec)[k]["initial_guess"] > 0 for k in ("Ieff_roll", "Ieff_pitch", "Ieff_yaw"))
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(spec))
    load_vehicle_spec(str(p))


def test_the_thrusters_from_a_preset_a_layout_or_an_allocation_matrix(tmp_path):
    preset = VD.max_force_of("bluerov2_heavy")
    layout = VD.max_force_of(os.path.join(CAL, "templates", "ingest_csv_pwm_custom_thrusters.yaml"))  # a Heavy's layout
    alloc = tmp_path / "alloc.yaml"
    alloc.write_text(yaml.safe_dump({"wrench": {"thrusters": {
        "allocation_csv": os.path.join(CAL, "templates", "allocation_template.csv"),
        "directions": [-1, -1, 1, 1, -1, 1, 1, -1]}}}))
    assert layout == pytest.approx(preset, rel=0.01)
    assert VD.max_force_of(str(alloc)) == pytest.approx(preset, rel=1e-6)
    dfki = VD.max_force_of(os.path.join(CAL, "examples", "dfki_tether", "ingest_dfki_tether.yaml"))
    assert dfki[0] == pytest.approx(preset[0], rel=0.15) and 0.35 < dfki[2] / preset[2] < 0.55   # 2 vertical, not 4
    empty = tmp_path / "no_thrusters.yaml"
    empty.write_text("source:\n  format: csv\n")
    with pytest.raises(VD.DescribeError, match="no usable thrusters block"):
        VD.max_force_of(str(empty))
