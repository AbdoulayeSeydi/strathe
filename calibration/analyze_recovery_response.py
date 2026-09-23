import pandas as pd
import numpy as np

df = pd.read_csv('trace_recovery_cases.csv')

CASES = ['collision', 'robust_failure', 'success']
TOLERANCE_LOW, TOLERANCE_HIGH = 1.8, 2.2
RECEDING_THRESHOLD = 0.05  # d_dot above this counts as "meaningfully receding"

print("=== PART A: Action effectiveness conditioned on d_dot sign ===")
print("(does the policy change its action MORE when already receding vs approaching,")
print(" and does the separation-rate trend improve or worsen following each?)")
for case in CASES:
    sub = df[df['case'] == case].reset_index(drop=True)
    d_dot = sub['d_dot'].values
    fx = sub['cal_Fx_raw'].values
    ty = sub['cal_Ty_raw'].values
    n = len(sub)

    delta_ddot = d_dot[1:] - d_dot[:-1]
    delta_fx = np.abs(fx[1:] - fx[:-1])
    delta_ty = np.abs(ty[1:] - ty[:-1])

    receding_mask = d_dot[:-1] > 0
    approaching_mask = d_dot[:-1] <= 0

    m_ddot_r = delta_ddot[receding_mask].mean() if receding_mask.sum() > 0 else float('nan')
    m_ddot_a = delta_ddot[approaching_mask].mean() if approaching_mask.sum() > 0 else float('nan')
    m_dfx_r = delta_fx[receding_mask].mean() if receding_mask.sum() > 0 else float('nan')
    m_dfx_a = delta_fx[approaching_mask].mean() if approaching_mask.sum() > 0 else float('nan')
    m_dty_r = delta_ty[receding_mask].mean() if receding_mask.sum() > 0 else float('nan')
    m_dty_a = delta_ty[approaching_mask].mean() if approaching_mask.sum() > 0 else float('nan')

    print(f"{case:16s}: n_receding_steps={receding_mask.sum()}, n_approaching_steps={approaching_mask.sum()}")
    print(f"{'':16s}  mean delta_d_dot | receding={m_ddot_r:+.4f}  approaching={m_ddot_a:+.4f}")
    print(f"{'':16s}  mean |delta_Fx|  | receding={m_dfx_r:.2f}  approaching={m_dfx_a:.2f}")
    print(f"{'':16s}  mean |delta_Ty|  | receding={m_dty_r:.2f}  approaching={m_dty_a:.2f}")

print()
print("=== PART B: Recovery response rate ===")
print(f"For steps where d_dot > {RECEDING_THRESHOLD} AND separation outside [{TOLERANCE_LOW},{TOLERANCE_HIGH}]:")
print("what fraction of the next N steps have d_dot < 0?")
for case in CASES:
    sub = df[df['case'] == case].reset_index(drop=True)
    sep = sub['separation'].values
    d_dot = sub['d_dot'].values
    n = len(sub)

    event_idxs = [t for t in range(n) if d_dot[t] > RECEDING_THRESHOLD
                  and (sep[t] < TOLERANCE_LOW or sep[t] > TOLERANCE_HIGH)]

    print(f"{case:16s}: {len(event_idxs)} qualifying 'receding & out-of-tolerance' events out of {n} steps")
    if len(event_idxs) == 0:
        print(f"{'':16s}  (no qualifying events)")
        continue
    for N in [1, 3, 5, 10]:
        fractions = []
        for t in event_idxs:
            window = d_dot[t + 1:t + 1 + N]
            if len(window) == 0:
                continue
            fractions.append((window < 0).mean())
        mean_frac = np.mean(fractions) if fractions else float('nan')
        print(f"{'':16s}  N={N:2d}: mean fraction of next-N-steps with d_dot<0 = {mean_frac:.2f} "
              f"(over {len(fractions)} events)")