"""
Precise patch script for diagnose_calibration.py — inserts a multi-
parameter identifiability check (SVD near-null-space directions) right
after the existing condition-number computation, using ONLY the exact
text already confirmed present in the real file (via the user's own
pasted grep/sed output) as the anchor. Does not touch anything beyond
that anchor point.
"""
with open("diagnose_calibration.py", encoding="utf-8") as f:
    content = f.read()

anchor = '''    singular_values = np.linalg.svd(J, compute_uv=False)
    smallest = singular_values[-1]
    if smallest < 1e-12:
        condition_number = float("inf")
        print("Smallest singular value is ~0 — Jacobian is rank-deficient "
              "(at least one parameter direction has essentially no effect "
              "on the residuals at this point).")
    else:
        condition_number = float(singular_values[0] / smallest)'''

insertion = '''    singular_values = np.linalg.svd(J, compute_uv=False)
    smallest = singular_values[-1]
    if smallest < 1e-12:
        condition_number = float("inf")
        print("Smallest singular value is ~0 — Jacobian is rank-deficient "
              "(at least one parameter direction has essentially no effect "
              "on the residuals at this point).")
    else:
        condition_number = float(singular_values[0] / smallest)

    # --- Multi-parameter identifiability: near-null-space directions ---
    # The pairwise correlation matrix (below) only catches TWO-parameter
    # degeneracies. A flat direction can involve 3+ parameters
    # simultaneously with no single pair crossing the correlation
    # threshold — exactly the pattern found for Zw_dot after adding
    # Coriolis terms (pinned at its bound, high overall condition number,
    # but no pairwise correlation >=0.7 in the existing analysis). The
    # right singular vectors corresponding to the SMALLEST singular values
    # directly reveal these: each is a parameter COMBINATION the data
    # barely constrains as a group, even if no single pair within it is
    # individually degenerate. Reuses the SAME Jacobian already computed
    # above — no additional expensive simulate_trajectory() calls.
    from dynamics_model import CalibrationParameters
    param_names = CalibrationParameters.from_vector(param_vec).param_names
    _, S, Vt = np.linalg.svd(J, full_matrices=False)
    n_flat_to_report = min(3, len(S))
    print(f"\\nSmallest {n_flat_to_report} singular value(s) and their parameter-combination "
          f"directions (multi-parameter flat-direction check, not just pairwise):")
    for i in range(1, n_flat_to_report + 1):
        sv = S[-i]
        direction = Vt[-i]
        order = np.argsort(-np.abs(direction))
        print(f"  Singular value {sv:.4e} (rank {len(S) - i + 1}/{len(S)}) — "
              f"parameters with the largest role in this barely-constrained combination:")
        for j in order[:5]:
            print(f"      {param_names[j]:<18}{direction[j]:+.3f}")'''

if anchor not in content:
    print("ANCHOR NOT FOUND — do not proceed, the real file must differ from what was pasted. "
          "Paste the current content again so this can be redone correctly.")
else:
    new_content = content.replace(anchor, insertion, 1)
    with open("diagnose_calibration.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Patch applied successfully.")