import pandas as pd
import numpy as np

df = pd.read_csv('trace_recovery_cases.csv')

def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return np.corrcoef(a, b)[0, 1]

def run_lengths(signs):
    runs, cur = [], 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return runs

CASES = ['collision', 'robust_failure', 'success']

print('=== 1. SURGE-PITCH COORDINATION ===')
for case in CASES:
    sub = df[df['case'] == case].reset_index(drop=True)
    fx, pitch, q = sub['cal_Fx_raw'].values, sub['cal_pitch'].values, sub['cal_q'].values
    r_fx_pitch, r_fx_q = corr(fx, pitch), corr(fx, q)
    r_absfx_abspitch = corr(np.abs(fx), np.abs(pitch))
    mean_eff_frac = np.cos(pitch).mean()
    thresh_hi, thresh_lo = np.percentile(np.abs(pitch), 66), np.percentile(np.abs(pitch), 33)
    hi_mask, lo_mask = np.abs(pitch) >= thresh_hi, np.abs(pitch) <= thresh_lo
    mean_absfx_hi = np.abs(fx[hi_mask]).mean() if hi_mask.sum() > 0 else float('nan')
    mean_absfx_lo = np.abs(fx[lo_mask]).mean() if lo_mask.sum() > 0 else float('nan')
    print(f'{case:16s}: corr(Fx,pitch)={r_fx_pitch:+.2f}  corr(Fx,q)={r_fx_q:+.2f}  corr(|Fx|,|pitch|)={r_absfx_abspitch:+.2f}')
    print(f'{"":16s}  mean cos(pitch)={mean_eff_frac:.2f}   mean|Fx| hi-pitch={mean_absfx_hi:.1f} vs lo-pitch={mean_absfx_lo:.1f}')

print()
print('=== 2. TORQUE TIMING (lagged correlations) ===')
for case in CASES:
    sub = df[df['case'] == case].reset_index(drop=True)
    ty, q, pitch = sub['cal_Ty_raw'].values, sub['cal_q'].values, sub['cal_pitch'].values
    r1 = corr(ty[:-1], q[1:])
    r2 = corr(q[:-1], ty[1:])
    r3 = corr(pitch[:-1], ty[1:])
    print(f'{case:16s}: corr(Ty[t],q[t+1])={r1:+.2f}  corr(q[t],Ty[t+1])={r2:+.2f}  corr(pitch[t],Ty[t+1])={r3:+.2f}')

print()
print('=== 3. SATURATION vs d_dot ===')
for case in CASES:
    sub = df[df['case'] == case].reset_index(drop=True)
    sat_mask = sub['any_component_saturated'].astype(bool).values
    ddot = sub['d_dot'].values
    m_sat = ddot[sat_mask].mean() if sat_mask.sum() > 0 else float('nan')
    m_unsat = ddot[~sat_mask].mean() if (~sat_mask).sum() > 0 else float('nan')
    print(f'{case:16s}: mean d_dot WHILE saturated={m_sat:+.3f}  NOT saturated={m_unsat:+.3f}  (n_sat={sat_mask.sum()}, n_unsat={(~sat_mask).sum()})')

print()
print('=== 4. CONTROL SWITCHING FREQUENCY ===')
for case in CASES:
    sub = df[df['case'] == case].reset_index(drop=True)
    fx, ty = sub['cal_Fx_raw'].values, sub['cal_Ty_raw'].values
    dfx, dty = np.diff(fx), np.diff(ty)
    fx_flips = (np.sign(fx[1:]) != np.sign(fx[:-1])).sum()
    ty_flips = (np.sign(ty[1:]) != np.sign(ty[:-1])).sum()
    fx_runs, ty_runs = run_lengths(np.sign(fx)), run_lengths(np.sign(ty))
    print(f'{case:16s}: mean|dFx|={np.abs(dfx).mean():.2f}  mean|dTy|={np.abs(dty).mean():.2f}')
    print(f'{"":16s}  Fx flips={fx_flips}/{len(fx)-1} ({100*fx_flips/(len(fx)-1):.0f}%)  mean run={np.mean(fx_runs):.1f}, max={max(fx_runs)}')
    print(f'{"":16s}  Ty flips={ty_flips}/{len(ty)-1} ({100*ty_flips/(len(ty)-1):.0f}%)  mean run={np.mean(ty_runs):.1f}, max={max(ty_runs)}')