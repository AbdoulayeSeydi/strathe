"""
simfit/editors.py -- read and change numbers in a lab's own model file, leaving everything else byte for byte
(PROJECT_STATUS.md #81). The fit writes a runnable copy at every step and the calibrated model at the end; a lab
compares it with the original, so only the tuned numbers may change: comments, formatting and attribute order stay.

Where a number is, by format:
  xml   (SDF, URDF, MJCF, Stonefish .scn, any XML) a small XPath: /sdf/model/link[@name='base_link']/inertial/mass,
        //plugin[@name='gz::sim::systems::Hydrodynamics']/xUabsU, //geom[@name='hull']/@fluidcoef#0
        (steps separated by / or // (anywhere below); a step is a tag or *, with [@attr='value'] and/or [n] (1-based);
        a final /@attr is an attribute; #k picks the k-th of space-separated numbers, from 0)
  json  a JSON pointer: /agents/0/dynamics/mass (the file is rewritten with the same key order; JSON has no comments)
  yaml  a JSON pointer into the YAML (rewritten by PyYAML: comments are not kept, the guide says so)
  regex a regular expression with one capture group around the number: mass="([^"]+)"
"""
from __future__ import annotations

import json
import re
import xml.parsers.expat
from dataclasses import dataclass, field
from typing import Optional

import yaml

from simfit.core import SimFitError

FORMATS = ("xml", "json", "yaml", "regex")


def fmt_number(v: float) -> str:
    return f"{float(v):.9g}"


# ---------------------------------------------------------------------------------------------------------------------
# XML: offsets from expat, edits as text
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class _Node:
    tag: str
    attrs: dict
    start: int                     # byte offset of '<'
    head_end: int = 0              # byte offset just after the start tag's '>'
    end: int = 0                   # byte offset of the end tag's '<' (= head_end for <a/>)
    parent: Optional["_Node"] = None
    children: list = field(default_factory=list)


def _parse_xml(data: bytes) -> _Node:
    root = _Node("#document", {}, 0)
    stack = [root]
    p = xml.parsers.expat.ParserCreate()
    p.ordered_attributes = False

    def start(tag, attrs):
        i = p.CurrentByteIndex
        n = _Node(tag, attrs, i, parent=stack[-1])
        n.head_end = _tag_end(data, i)
        stack[-1].children.append(n)
        stack.append(n)

    def end(tag):
        n = stack.pop()
        n.end = n.head_end if data[n.head_end - 2:n.head_end] == b"/>" else p.CurrentByteIndex

    p.StartElementHandler, p.EndElementHandler = start, end
    try:
        p.Parse(data, True)
    except xml.parsers.expat.ExpatError as e:
        raise SimFitError(f"the model file isn't valid XML: {e}")
    return root


def _tag_end(data: bytes, i: int) -> int:
    """The offset just after the '>' closing the tag that starts at i (quotes respected)."""
    q = None
    for j in range(i, len(data)):
        c = data[j:j + 1]
        if q:
            if c == q:
                q = None
        elif c in (b'"', b"'"):
            q = c
        elif c == b">":
            return j + 1
    raise SimFitError("the model file has an unclosed tag")


_STEP = re.compile(r"^([\w:.\-*]+)((?:\[[^\]]+\])*)$")


def _split_path(path: str):
    """'//a/b[@x='1']/@attr#2' -> ([(descendant?, tag, predicates)], attr, component)."""
    comp = None
    m = re.match(r"^(.*)#(\d+)$", path)
    if m:
        path, comp = m.group(1), int(m.group(2))
    attr = None
    m = re.match(r"^(.*)/@([\w:.\-]+)$", path)
    if m:
        path, attr = m.group(1), m.group(2)
    steps, i = [], 0
    if not path.startswith("/"):
        raise SimFitError(f"an XML path starts with / or //: {path!r}")
    while i < len(path):
        desc = path.startswith("//", i)
        i += 2 if desc else 1
        j = i
        depth = 0
        while j < len(path) and (path[j] != "/" or depth):
            depth += path[j] == "["
            depth -= path[j] == "]"
            j += 1
        step = path[i:j]
        sm = _STEP.match(step)
        if not sm:
            raise SimFitError(f"can't read the XML path step {step!r} in {path!r}")
        preds = re.findall(r"\[([^\]]+)\]", sm.group(2))
        steps.append((desc, sm.group(1), preds))
        i = j
    return steps, attr, comp


def _matches(n: _Node, tag: str, preds: list) -> bool:
    if tag != "*" and n.tag != tag:
        return False
    for pr in preds:
        m = re.match(r"""^@([\w:.\-]+)\s*=\s*(['"])(.*)\2$""", pr.strip())
        if m:
            if n.attrs.get(m.group(1)) != m.group(3):
                return False
        elif not pr.strip().isdigit():
            raise SimFitError(f"unsupported XML predicate [{pr}]: use [@attr='value'] or [n]")
    return True


def _descendants(n: _Node):
    for c in n.children:
        yield c
        yield from _descendants(c)


def _select(root: _Node, steps) -> list:
    cur = [root]
    for desc, tag, preds in steps:
        nxt = []
        for n in cur:
            pool = list(_descendants(n)) if desc else n.children
            hits = [c for c in pool if _matches(c, tag, preds)]
            idx = [int(p) for p in preds if p.strip().isdigit()]
            if idx:
                hits = [hits[idx[0] - 1]] if 0 < idx[0] <= len(hits) else []
            nxt += hits
        cur = nxt
    return cur


class XmlFile:
    def __init__(self, text: str):
        self.data = text.encode("utf-8")

    @property
    def text(self) -> str:
        return self.data.decode("utf-8")

    def _locate(self, path: str):
        """(start, end) byte span of the value's text, and the component index (or None)."""
        steps, attr, comp = _split_path(path)
        hits = _select(_parse_xml(self.data), steps)
        if len(hits) != 1:
            raise SimFitError(f"the model file has {len(hits)} matches for {path!r}, not one: make the path more "
                              f"specific (e.g. [@name='...'])" if hits else f"the model file has nothing at {path!r}")
        n = hits[0]
        if attr:
            head = self.data[n.start:n.head_end]
            m = re.search(rb"(?<![\w:.\-])" + re.escape(attr.encode()) + rb"""\s*=\s*(["'])(.*?)\1""", head, re.S)
            if not m:
                raise SimFitError(f"{path!r}: the element has no attribute {attr!r}")
            a, b = n.start + m.start(2), n.start + m.end(2)
        else:
            if n.children:
                raise SimFitError(f"{path!r} has child elements; point at the element that holds the number")
            a, b = n.head_end, n.end
        return a, b, comp

    @staticmethod
    def _component(raw: str, comp: Optional[int]):
        """(numbers text, offset of the component inside it, its length)"""
        if comp is None:
            s = raw.strip()
            return raw.index(s) if s else 0, len(s)
        parts = list(re.finditer(r"\S+", raw))
        if comp >= len(parts):
            raise SimFitError(f"component #{comp} asked for, but the value {raw.strip()!r} has {len(parts)}")
        return parts[comp].start(), parts[comp].end() - parts[comp].start()

    def get(self, path: str) -> float:
        a, b, comp = self._locate(path)
        raw = self.data[a:b].decode("utf-8")
        o, ln = self._component(raw, comp)
        try:
            return float(raw[o:o + ln])
        except ValueError:
            raise SimFitError(f"{path!r} is {raw[o:o + ln]!r}, not a number")

    def set(self, path: str, value: float) -> None:
        a, b, comp = self._locate(path)
        raw = self.data[a:b].decode("utf-8")
        o, ln = self._component(raw, comp)
        new = raw[:o] + fmt_number(value) + raw[o + ln:]
        self.data = self.data[:a] + new.encode("utf-8") + self.data[b:]


# ---------------------------------------------------------------------------------------------------------------------
# JSON / YAML (JSON pointer) and regex
# ---------------------------------------------------------------------------------------------------------------------

def _pointer(path: str) -> list:
    if not path.startswith("/"):
        raise SimFitError(f"a JSON pointer starts with /: {path!r}")
    return [p.replace("~1", "/").replace("~0", "~") for p in path[1:].split("/")]


class TreeFile:
    def __init__(self, text: str, kind: str):
        self.kind = kind
        try:
            self.doc = json.loads(text) if kind == "json" else yaml.safe_load(text)
        except (ValueError, yaml.YAMLError) as e:
            raise SimFitError(f"the model file isn't valid {kind.upper()}: {e}")
        self.indent = 2
        if kind == "json":
            m = re.search(r"\n( +)\S", text)
            self.indent = len(m.group(1)) if m else 2

    def _walk(self, path: str):
        keys = _pointer(path)
        node = self.doc
        for k in keys[:-1]:
            node = node[int(k)] if isinstance(node, list) else node[k]
        last = keys[-1]
        return node, (int(last) if isinstance(node, list) else last)

    def get(self, path: str) -> float:
        try:
            node, k = self._walk(path)
            return float(node[k])
        except (KeyError, IndexError, TypeError, ValueError):
            raise SimFitError(f"the model file has no number at {path!r}")

    def set(self, path: str, value: float) -> None:
        self.get(path)
        node, k = self._walk(path)
        node[k] = float(value)

    @property
    def text(self) -> str:
        if self.kind == "json":
            return json.dumps(self.doc, indent=self.indent) + "\n"
        return yaml.safe_dump(self.doc, sort_keys=False)


class RegexFile:
    def __init__(self, text: str):
        self.text = text

    def _match(self, pattern: str):
        ms = list(re.finditer(pattern, self.text))
        if len(ms) != 1 or (ms and ms[0].re.groups != 1):
            raise SimFitError(f"the pattern {pattern!r} must match once, with one ( ) around the number; it matched "
                              f"{len(ms)} time(s)")
        return ms[0]

    def get(self, pattern: str) -> float:
        m = self._match(pattern)
        try:
            return float(m.group(1))
        except ValueError:
            raise SimFitError(f"{pattern!r} captured {m.group(1)!r}, not a number")

    def set(self, pattern: str, value: float) -> None:
        m = self._match(pattern)
        self.text = self.text[:m.start(1)] + fmt_number(value) + self.text[m.end(1):]


def open_model(text: str, fmt: str):
    """An editor for a model file's text: .get(path) -> float, .set(path, value), .text"""
    if fmt == "xml":
        return XmlFile(text)
    if fmt in ("json", "yaml"):
        return TreeFile(text, fmt)
    if fmt == "regex":
        return RegexFile(text)
    raise SimFitError(f"unknown model format {fmt!r}: use one of {', '.join(FORMATS)}")


def guess_format(path: str) -> str:
    ext = path.lower().rsplit(".", 1)[-1]
    return {"json": "json", "yaml": "yaml", "yml": "yaml"}.get(ext, "xml")
