"""Every refusal the sensor suite can give (PROJECT_STATUS.md #96, "no errors no matter what": each stop names the
sensor, the problem and the fix) is reached by at least one test: a test asserts on a piece of its text. A message
added without a test fails here. Read from the code, so the list can't fall behind it."""
import ast
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
SOURCES = ("sensor_ingest.py", "measurements.py", "state_smoother.py")
# data_ingestion.py's messages about the sensor blocks (#96); its older messages predate the suite
SENSOR_WORDS = re.compile(r"depth|position|latitude|accelerometer|gyro|fix|usbl|gps|pressure|delay_s|track|odometry"
                          r"|sensor", re.I)
# can't be reached from any input: the channel kinds are a closed set made by the code itself
EXEMPT = {"unknown channel kind"}


def _pieces(a):
    if isinstance(a, ast.JoinedStr):
        return [v.value for v in a.values if isinstance(v, ast.Constant)]
    if isinstance(a, ast.Constant) and isinstance(a.value, str):
        return [a.value]
    if isinstance(a, ast.BinOp):
        return _pieces(a.left) + _pieces(a.right)
    return []


def _messages(path):
    """(line, text) of each raise and each check's out["problem"] = ... in the file."""
    for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
        arg = None
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and node.exc.args:
            arg = node.exc.args[0]
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
            sl = node.targets[0].slice
            if isinstance(sl, ast.Constant) and sl.value == "problem":
                arg = node.value
        text = re.sub(r"\s+", " ", " ".join(p for p in _pieces(arg) if p.strip())) if arg is not None else ""
        if len(text.strip()) >= 12:
            yield node.lineno, text


def _asserted_strings():
    out = set()
    for f in glob.glob(os.path.join(HERE, "*.py")):
        for node in ast.walk(ast.parse(open(f, encoding="utf-8").read())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value.strip()) >= 8:
                out.add(re.sub(r"\s+", " ", node.value.strip()))
            if isinstance(node, ast.JoinedStr):   # match=f"{block}.delay_s ...": its fixed part
                for v in node.values:
                    if isinstance(v, ast.Constant) and len(v.value.strip()) >= 8:
                        out.add(re.sub(r"\s+", " ", v.value.strip()))
    return out


def test_every_sensor_refusal_has_a_test():
    asserted = _asserted_strings()
    found, missing = 0, []
    for name in SOURCES + ("data_ingestion.py",):
        for line, text in _messages(os.path.join(CAL, name)):
            if name == "data_ingestion.py" and not SENSOR_WORDS.search(text):
                continue
            if any(text.strip().startswith(e) for e in EXEMPT):
                continue
            found += 1
            if not any(s in text for s in asserted):
                missing.append(f"{name}:{line}: {text[:100]}")
    assert found >= 20, f"only {found} messages found: the scan is broken"
    assert not missing, "no test reaches these refusals:\n" + "\n".join(missing)
