import sys

PATH = "phase2_config.py"

ANCHOR = "not day-to-day validation.\n\n## Decision 3"

ADDENDUM_INSERT = """not day-to-day validation.

## Decision 2, addendum: B2.1's HoloOcean-relative criterion (see conversation)
Track B's own stated completion criterion for B2.1 reads "held-out RMSE
is in a sane range relative to the other two backends" -- written before
Decision 2's revision above, and taken completely literally it implies a
live three-way comparison including HoloOcean. Attempting to actually
run that comparison surfaced two separate, concrete blockers, not just
inconvenience:

  1. HoloOcean is not installed in this project's working environment,
     and this machine is macOS -- HoloOcean's own documented requirements
     list only "Linux or Windows 64bit". Even on a supported platform,
     installation is not a simple pip command: recent versions cannot be
     distributed via PyPI at all ("in accordance with the Unreal Engine
     EULA"), requiring an Epic Games account linked to GitHub, EULA
     acceptance, then a source install plus a separate multi-gigabyte
     world-binary download.

  2. The fallback approach -- reusing an EXISTING historical HoloOcean
     report and re-running its exact parameters through MuJoCo for a
     direct, live comparison -- is also blocked. Five historical
     replay_comparison.py output directories exist
     (replay_holoocean_{clean_inertia_test,segment3,segment4,ieff_fixed,
     segment5}_placeholder), but replay_comparison.py's own report
     schema never records which --calibrated-params file produced a
     given result, and none of the five retained a copy of their own
     source parameters. calibration_output_holoocean_coriolis -- the one
     directory this file's own Decision 2 text above cites by name, with
     a real metrics.json -- no longer exists on this machine at all. This
     is the same class of provenance gap PROJECT_STATUS.md already
     flagged for the segment5 report specifically (its
     calibrated_parameters.yaml didn't match its own
     calibrated_parameters_segment5_backup.yaml); here it's total
     absence rather than a mismatch, but the practical effect -- no
     verifiable link between a saved number and the parameters that
     produced it -- is the same.

Given both paths are blocked, this criterion is being closed on the
strength of Decision 2's OWN evidence, not a fresh live comparison:
MuJoCo and the stand-in agree on B2.1's own held-out result to a ~1.00x
ratio (0.320402 vs. 0.320404) -- close enough that MuJoCo can be treated
as inheriting the stand-in's own relationship to HoloOcean, without a
separate direct measurement. An indirect substitute exists too: a
--no-holoocean (stand-in) replay of HoloOcean-fit parameters
(calibration_output_holoocean_coriolis, while it still existed) gave
0.2575 -- a broadly comparable magnitude, not a contradiction.

One honest, UNRESOLVED caveat this investigation surfaced, not silently
dropped: the four non-segment5 historical reports (clean_inertia_test,
segment3, segment4, ieff_fixed -- all testing earlier calibration
attempts, NOT calibrated_parameters_a1.yaml) show HoloOcean's own
scale-normalized calibrated-vs-default improvement ranging from -21.5%
to +8.0% -- markedly worse than the ~30-40% improvements this project has
consistently seen via stand-in/MuJoCo. Without the source parameters,
this cannot be attributed to either (a) those specifically being weaker,
earlier attempts, or (b) a genuine, systematic HoloOcean-vs-stand-in
divergence this project's calibrated physics hasn't caught. This
ambiguity is exactly the kind of thing HoloOcean's narrowed
"independent physics cross-check" role above exists to eventually
resolve -- it just hasn't been resolved yet, and B2.1's closure here
should not be read as having resolved it.

RECOMMENDATION for any future rigorous three-way comparison (e.g. a
result_type="final" CLAIM_A, per Decision 3's own labeling discipline):
(1) HoloOcean would need a working install on an actually-supported
platform, not this machine as currently configured; (2) any future
HoloOcean-backed run should have its own output directory save a literal
copy of its --calibrated-params input alongside the report -- the
concrete, fixable gap that made all five existing historical directories
unverifiable against a fresh run.

## Decision 3"""

with open(PATH, encoding="utf-8") as f:
    content = f.read()

count = content.count(ANCHOR)
if count == 1:
    new_content = content.replace(ANCHOR, ADDENDUM_INSERT)
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Inserted successfully -- anchor matched exactly once.")
    sys.exit(0)
elif count > 1:
    print(f"ABORTING -- anchor matched {count} times (expected exactly 1). File not modified.")
    sys.exit(1)
else:
    print("ABORTING -- anchor not found. File not modified. Paste the relevant Decision 2 closing lines back.")
    sys.exit(1)
