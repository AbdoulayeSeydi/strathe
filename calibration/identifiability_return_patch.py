# Safe, self-verifying patch: identifiability_analysis() already COMPUTES
# the near-null-space (multi-parameter flat-direction) findings and
# prints them, but never returns them - meaning an automated report can't
# access the exact finding that solved the Zw_dot mystery earlier
# tonight. This captures what's already computed into the return dict -
# no new computation, just not discarding data already produced.

with open("diagnose_calibration.py", encoding="utf-8") as f:
    content = f.read()

old = '''    for i in range(1, n_flat_to_report + 1):
        sv = S[-i]
        direction = Vt[-i]
        order = np.argsort(-np.abs(direction))
        print(f"  Singular value {sv:.4e} (rank {len(S) - i + 1}/{len(S)}) — "
              f"parameters with the largest role in this barely-constrained combination:")
        for j in order[:5]:
            print(f"      {param_names[j]:<18}{direction[j]:+.3f}")

    JtJ = J.T @ J
    try:
        # Cov(params) ~ (J^T J)^-1 up to an unknown noise-variance scale
        # factor — that factor cancels out of the CORRELATION matrix even
        # though it affects the absolute covariance magnitude.
        cov = np.linalg.inv(JtJ)
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
    except np.linalg.LinAlgError:
        corr = None

    return {"jacobian": J, "condition_number": condition_number, "correlation": corr}'''

new = '''    near_null_directions = []
    for i in range(1, n_flat_to_report + 1):
        sv = S[-i]
        direction = Vt[-i]
        order = np.argsort(-np.abs(direction))
        print(f"  Singular value {sv:.4e} (rank {len(S) - i + 1}/{len(S)}) — "
              f"parameters with the largest role in this barely-constrained combination:")
        top_params = []
        for j in order[:5]:
            print(f"      {param_names[j]:<18}{direction[j]:+.3f}")
            top_params.append({"parameter": param_names[j], "loading": float(direction[j])})
        near_null_directions.append({"singular_value": float(sv), "top_loadings": top_params})

    JtJ = J.T @ J
    try:
        # Cov(params) ~ (J^T J)^-1 up to an unknown noise-variance scale
        # factor — that factor cancels out of the CORRELATION matrix even
        # though it affects the absolute covariance magnitude.
        cov = np.linalg.inv(JtJ)
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
    except np.linalg.LinAlgError:
        corr = None

    return {
        "jacobian": J, "condition_number": condition_number, "correlation": corr,
        "near_null_directions": near_null_directions,
    }'''

count = content.count(old)
if count != 1:
    print(f"REFUSING TO PATCH - expected exactly 1 occurrence, found {count}.")
else:
    content = content.replace(old, new, 1)
    with open("diagnose_calibration.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Return-value fix applied successfully - near_null_directions now captured, not discarded.")