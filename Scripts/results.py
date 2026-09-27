"""All results numbers for the paper, computed from outputs/ and printed in
the order they appear in the text. Also written to outputs/results.json.

Sections
  1  validation ladder (gates 1a, 1b, 1c)
  2  GR baselines per event (Table 1)
  3  kernel verdicts, limits, windows, profile peaks (Table 2)
  4  cross-veto matrix (Table 3)
  5  joint constraint from the weighted estimator (needs joint_fit_outputs)
  6  per-event weighted vs sampled limits (Table 5) and distance scaling
  7  physical translation (peak |Re chi|, effective graviton mass)

Usage: python scripts/results.py
"""
import glob
import json
import os
import sys
import numpy as np
from scipy.special import logsumexp
import bilby

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import run_catalog as rc

VAL = os.path.join(rc.ROOT, "outputs", "validation_outputs")
JOINT = os.path.join(rc.ROOT, "outputs", "joint_fit_outputs")
RESULTS_JSON = os.path.join(rc.ROOT, "outputs", "results.json")
R = {}

FEATURES = {
    "GW151226@70":  (64, 77),
    "GW190814@106": (100, 110),
    "GW190408@110": (110, 114),
    "GW200311@114": (114, 121),
    "123Hz (O1)":   (121, 133),
    "GW170814@150": (143, 157),
}


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def q(x, d=2, pct=(5, 50, 95)):
    lo, m, hi = np.percentile(x, pct)
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def profile(f0, logL, edges=np.linspace(20, 200, 46)):
    c, p = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f0 >= lo) & (f0 < hi)
        if m.sum() > 5:
            c.append(0.5 * (lo + hi)); p.append(logL[m].max())
    return np.array(c), np.array(p)


def chi_eff(p):
    Mc, qq = p["chirp_mass"].values, p["mass_ratio"].values
    m1 = Mc * (1 + qq) ** 0.2 / qq ** 0.6
    m2 = qq * m1
    return (m1 * p["a_1"].values * np.cos(p["tilt_1"].values)
            + m2 * p["a_2"].values * np.cos(p["tilt_2"].values)) / (m1 + m2)


def wq95(values, weights):
    o = np.argsort(values)
    cc = np.cumsum(weights[o]) / weights[o].sum()
    return float(values[o][np.searchsorted(cc, 0.95)])


def load(path):
    return bilby.result.read_in_result(path)


# ============================================================ 1 validation
hdr("1  VALIDATION LADDER")
r1a = load(os.path.join(VAL, "1a_fixed", "1a_fixed_result.json"))
Mc = r1a.posterior["chirp_mass"].values
lo, m, hi = np.percentile(Mc, [5, 50, 95])
R["val_1a"] = dict(Mc_median=m, Mc_plus=hi - m, Mc_minus=m - lo,
                   Mc_true=(150 * 120) ** 0.6 / 270 ** 0.2,
                   lnBF_signal_noise=r1a.log_bayes_factor)
print(f"1a  Mc = {m:.1f} +{hi-m:.1f} -{m-lo:.1f}  (truth {R['val_1a']['Mc_true']:.1f});  "
      f"lnBF(signal vs noise) = {r1a.log_bayes_factor:.1f}")

MPC_SEC = rc.MPC_SEC
for stage, lab in (("marginal", "1b_v1"), ("loud", "1b_loud_v3")):
    r = load(os.path.join(VAL, lab, f"{lab}_result.json"))
    inj, p = r.injection_parameters, r.posterior
    tau = (inj["eps"] * (2 * np.pi * inj["f0_res"]) ** 2 * inj["luminosity_distance"]
           * MPC_SEC / (2 * np.pi * inj["gamma_res"]))
    f0 = p["f0_res"].values
    lo, m, hi = np.percentile(f0, [5, 50, 95])
    e_lo, e_hi = np.percentile(p["eps"].values, [5, 95])
    g_lo, g_hi = np.percentile(p["gamma_res"].values, [5, 95])
    R[f"val_1b_{stage}"] = dict(eps_inj=inj["eps"], tau=tau,
                                f0_median=m, f0_plus=hi - m, f0_minus=m - lo,
                                eps_ci90=[e_lo, e_hi], gamma_ci90=[g_lo, g_hi],
                                frac_eps_below_1e22=float((p["eps"].values < 1e-22).mean()),
                                lnBF_kernel_noise=r.log_bayes_factor)
    print(f"1b-{stage:8s} eps_inj={inj['eps']:.1e}  tau={tau:.3f}  "
          f"f0 = {m:.1f} +{hi-m:.1f} -{m-lo:.1f} Hz  "
          f"eps 90% [{e_lo:.1e}, {e_hi:.1e}]  gamma 90% [{g_lo:.2f}, {g_hi:.2f}] Hz  "
          f"frac(eps<1e-22)={R[f'val_1b_{stage}']['frac_eps_below_1e22']:.2f}")


def spurious_gain(r):
    c, p = profile(r.posterior["f0_res"].values, r.posterior["log_likelihood"].values)
    return float(p.max() - np.median(p))


gains = np.array([spurious_gain(load(os.path.join(VAL, f"1c_seed{i}", f"1c_seed{i}_result.json")))
                  for i in range(10)])
r1c = load(os.path.join(VAL, "1c_v1", "1c_v1_result.json"))
R["val_1c"] = dict(single_eps95=float(np.percentile(r1c.posterior["eps"].values, 95)),
                   single_frac_below_1e22=float((r1c.posterior["eps"].values < 1e-22).mean()),
                   null_gains=gains.tolist(), null_mean=gains.mean(), null_max=gains.max(),
                   null_p95=float(np.percentile(gains, 95)), threshold_used=rc.THRESHOLD)
print(f"1c  single: eps95={R['val_1c']['single_eps95']:.2e}, "
      f"frac(eps<1e-22)={R['val_1c']['single_frac_below_1e22']:.2f}")
print(f"1c  null (10 seeds): mean {gains.mean():.2f}, max {gains.max():.2f}, "
      f"95th pct {np.percentile(gains, 95):.2f}   (threshold in use: {rc.THRESHOLD})")

# ============================================================ 2 GR baselines
hdr("2  GR BASELINES (Table 1)   median [5%, 95%], detector-frame chirp mass")
evs = list(rc.QUEUE)
G, K = {}, {}
for ev in evs:
    for model, store in (("gr", G), ("kernel", K)):
        lab = f"{ev}_{model}_v1"
        store[ev] = load(os.path.join(rc.OUTDIR, lab, f"{lab}_result.json"))
print(f"{'event':<18}{'Mc_det':>20}{'q':>20}{'dL[Mpc]':>22}{'chi_eff':>22}"
      f"{'SNR':>6}{'nGR':>7}{'nK':>7}{'s_lnZ':>7}  approx")
R["gr_baselines"] = {}
for ev in evs:
    g = G[ev].posterior
    snr = float(np.sqrt(2 * G[ev].log_bayes_factor))
    slnz = float(np.hypot(G[ev].log_evidence_err, K[ev].log_evidence_err))
    approx = G[ev].meta_data["likelihood"]["waveform_arguments"]["waveform_approximant"]
    R["gr_baselines"][ev] = dict(
        Mc=np.percentile(g["chirp_mass"], [5, 50, 95]).tolist(),
        q=np.percentile(g["mass_ratio"], [5, 50, 95]).tolist(),
        dL=np.percentile(g["luminosity_distance"], [5, 50, 95]).tolist(),
        chi_eff=np.percentile(chi_eff(g), [5, 50, 95]).tolist(),
        snr_proxy=snr, n_gr=len(g), n_kernel=len(K[ev].posterior), sigma_lnZ=slnz,
        detectors=sorted(G[ev].meta_data["likelihood"]["interferometers"].keys()),
        approximant=approx)
    print(f"{ev:<18}{q(g['chirp_mass'],1):>20}{q(g['mass_ratio']):>20}"
          f"{q(g['luminosity_distance'],0):>22}{q(chi_eff(g)):>22}"
          f"{snr:>6.1f}{len(g):>7}{len(K[ev].posterior):>7}{slnz:>7.2f}  {approx}")

# ============================================================ 3 kernel verdicts
hdr("3  KERNEL VERDICTS, LIMITS, WINDOWS, PROFILE PEAKS (Table 2)")
print(f"{'event':<18}{'D[Mpc]':>7}{'lnB GR,n':>9}{'lnB k,GR':>9}{'+/-':>5}"
      f"{'eps95 f0>40':>12}{'eps95 marg':>12}{'eps/G UL':>10}{'123Hz':>7}{'68Hz':>7}"
      f"{'peak':>6}{'@f0':>5}")
R["kernel"] = {}
for ev in evs:
    k = K[ev].posterior
    eps, f0, gam, L = (k["eps"].values, k["f0_res"].values,
                       k["gamma_res"].values, k["log_likelihood"].values)
    lnBF = float(K[ev].log_evidence - G[ev].log_evidence)
    err = float(np.hypot(K[ev].log_evidence_err, G[ev].log_evidence_err))
    m40 = f0 > 40
    ul40 = float(np.percentile(eps[m40], 95)) if m40.sum() > 50 else None
    win = {}
    for tag, (lo, hi) in rc.WINDOWS.items():
        w = (f0 > lo) & (f0 < hi)
        win[tag] = float(L[w].max() - L[~w].max()) if w.sum() > 5 else None
    c, p = profile(f0, L)
    peak, fpk = float(p.max() - np.median(p)), float(c[np.argmax(p)])
    D = float(np.median(k["luminosity_distance"]))
    R["kernel"][ev] = dict(
        D_median=D, lnBF_gr_noise=float(G[ev].log_bayes_factor),
        lnBF_kernel_gr=lnBF, lnBF_err=err,
        eps95_f0gt40=ul40, eps95_marginal=float(np.percentile(eps, 95)),
        eps_over_gamma_95=float(np.percentile(eps / gam, 95)),
        frac_eps_below_1e22=float((eps < 1e-22).mean()),
        corr_log_eps_a1=float(np.corrcoef(np.log10(eps), k["a_1"])[0, 1]),
        corr_log_eps_a2=float(np.corrcoef(np.log10(eps), k["a_2"])[0, 1]),
        window_123Hz=win["123Hz"], window_68Hz=win["68Hz"],
        profile_peak=peak, profile_peak_f0=fpk,
        feature=bool(peak > rc.THRESHOLD))
    w1 = f"{win['123Hz']:+.1f}" if win["123Hz"] is not None else "--"
    w2 = f"{win['68Hz']:+.1f}" if win["68Hz"] is not None else "--"
    print(f"{ev:<18}{D:>7.0f}{G[ev].log_bayes_factor:>9.1f}{lnBF:>+9.2f}{err:>5.2f}"
          f"{(ul40 or np.nan):>12.2e}{R['kernel'][ev]['eps95_marginal']:>12.2e}"
          f"{R['kernel'][ev]['eps_over_gamma_95']:>10.2e}{w1:>7}{w2:>7}"
          f"{peak:>+6.1f}{fpk:>5.0f}{'  FEATURE' if peak > rc.THRESHOLD else ''}")
kv = R["kernel"]
n_feat = sum(v["feature"] for v in kv.values())
best = min(evs, key=lambda e: kv[e]["eps95_f0gt40"] or np.inf)
R["census"] = dict(n_events=len(evs), n_features=n_feat,
                   max_abs_lnBF=max(abs(v["lnBF_kernel_gr"]) for v in kv.values()),
                   best_single_event=best, best_single_eps95=kv[best]["eps95_f0gt40"],
                   feature_frequencies={e: kv[e]["profile_peak_f0"] for e in evs if kv[e]["feature"]})
print(f"\nmax |lnB k,GR| = {R['census']['max_abs_lnBF']:.2f}  (threshold {rc.THRESHOLD})")
print(f"census: {n_feat}/{len(evs)} events with profile peak > {rc.THRESHOLD}: "
      f"{R['census']['feature_frequencies']}")
print(f"best single-event limit: {best} eps < {kv[best]['eps95_f0gt40']:.2e}")

# ============================================================ 4 cross-veto
hdr("4  CROSS-VETO MATRIX (Table 3): max_in lnL - max_out lnL, per window x event")
print(f"{'window':<16}" + "".join(f"{e[-6:]:>8}" for e in evs))
R["cross_veto"] = {}
for tag, (lo, hi) in FEATURES.items():
    R["cross_veto"][tag] = {}
    line = f"{tag:<16}"
    for ev in evs:
        k = K[ev].posterior
        f0, L = k["f0_res"].values, k["log_likelihood"].values
        w = (f0 > lo) & (f0 < hi)
        v = float(L[w].max() - L[~w].max()) if w.sum() > 5 else None
        R["cross_veto"][tag][ev] = v
        line += f"{v:>+8.1f}" if v is not None else f"{'--':>8}"
    print(line)
for tag, row in R["cross_veto"].items():
    pos = [e for e, v in row.items() if v is not None and v > 0]
    print(f"  {tag:<16} positive in: {pos}")

# ============================================================ 5 joint
hdr("5  JOINT CONSTRAINT (weighted estimator)")
th_path, lw_path = (os.path.join(JOINT, "joint_theta_val.npy"), os.path.join(JOINT, "joint_logw_val.npy"))
if os.path.exists(th_path) and os.path.exists(lw_path):
    TH, lw = np.load(th_path), np.load(lw_path)
    w = np.exp(lw - lw.max())
    m40 = TH[:, 1] > 40
    mask = m40 & ~((TH[:, 1] > 98) & (TH[:, 1] < 114))
    ul = wq95(TH[m40, 0], w[m40])
    ulg_all = wq95(TH[m40, 0] / TH[m40, 2], w[m40])
    ulg_mask = wq95(TH[mask, 0] / TH[mask, 2], w[mask])
    ess = float(w.sum() ** 2 / (w * w).sum())
    fb = np.linspace(20, 200, 25)
    curve = [wq95(TH[mm, 0], w[mm]) if (mm := (TH[:, 1] >= a) & (TH[:, 1] < b)).sum() > 30 else None
             for a, b in zip(fb[:-1], fb[1:])]
    R["joint"] = dict(n_grid=len(TH), ess=ess, eps95_f0gt40=ul,
                      eps_over_gamma_95_f0gt40=ulg_all,
                      eps_over_gamma_95_masked_98_114=ulg_mask,
                      ratio_to_best_single=kv[best]["eps95_f0gt40"] / ul,
                      curve_f0_centres=(0.5 * (fb[:-1] + fb[1:])).tolist(), curve_eps95=curve)
    print(f"grid N = {len(TH)},  ESS = {ess:.0f}")
    print(f"eps       < {ul:.2e}   (f0 > 40 Hz, 95%, {len(evs)} events)")
    print(f"eps/Gamma < {ulg_all:.2e}   (f0 > 40)      "
          f"eps/Gamma < {ulg_mask:.2e}   (98-114 Hz masked)")
    print(f"improvement over best single event ({best}): x{kv[best]['eps95_f0gt40']/ul:.1f}")
    # per-event weighted limits (Table 5)
    ck = sorted(glob.glob(os.path.join(JOINT, "joint_dlnL_*_val.npy")))
    R["weighted_per_event"] = {}
    if ck:
        print(f"\n{'event':<18}{'weighted':>12}{'sampled':>12}{'ratio':>7}")
        for f in ck:
            ev = os.path.basename(f).split("joint_dlnL_")[1].rsplit("_val.npy", 1)[0]
            if ev not in kv:
                continue
            d = np.load(f)
            lwe = logsumexp(d, axis=1) - np.log(d.shape[1])
            ww = np.exp(lwe - lwe.max())
            uw = wq95(TH[m40, 0], ww[m40])
            R["weighted_per_event"][ev] = dict(weighted=uw, sampled=kv[ev]["eps95_f0gt40"], K=int(d.shape[1]))
            print(f"{ev:<18}{uw:>12.2e}{kv[ev]['eps95_f0gt40']:>12.2e}{uw/kv[ev]['eps95_f0gt40']:>7.2f}")
        print(f"(K = {d.shape[1]} binary-parameter draws per event)")
else:
    print("[joint arrays not found in outputs/joint_fit_outputs; section skipped]")
    R["joint"] = None

# ============================================================ 6 distance scaling
hdr("6  DISTANCE SCALING")
D = np.array([kv[e]["D_median"] for e in evs])
U = np.array([kv[e]["eps95_f0gt40"] for e in evs])
s, i = np.polyfit(np.log10(D), np.log10(U), 1)
R["distance_scaling"] = dict(sampled_slope=float(s), sampled_intercept=float(i))
print(f"sampled limits:  slope {s:+.2f}  (tau ∝ eps D predicts -1.00)  [{len(evs)} events]")
if R.get("weighted_per_event"):
    wevs = list(R["weighted_per_event"])
    Dw = np.array([kv[e]["D_median"] for e in wevs])
    Uw = np.array([R["weighted_per_event"][e]["weighted"] for e in wevs])
    sw, iw = np.polyfit(np.log10(Dw), np.log10(Uw), 1)
    R["distance_scaling"].update(weighted_slope=float(sw), weighted_intercept=float(iw))
    print(f"weighted limits: slope {sw:+.2f}   [{len(wevs)} events]")

# ============================================================ 7 physics
hdr("7  PHYSICAL TRANSLATION")
HBAR_EV_S = 6.582119569e-16
eps_lim = R["joint"]["eps95_f0gt40"] if R.get("joint") else kv[best]["eps95_f0gt40"]
src = "joint" if R.get("joint") else f"best single ({best})"
R["physics"] = dict(eps_limit_used=eps_lim, source=src, m_eff_eV={}, peak_ReChi={})
print(f"using eps < {eps_lim:.2e} ({src})")
for f0 in (40.0, 100.0, 200.0):
    m_eff = HBAR_EV_S * 2 * np.pi * f0 * np.sqrt(2 * eps_lim)
    R["physics"]["m_eff_eV"][str(f0)] = m_eff
    print(f"  m_eff c^2 (f0 = {f0:>5.0f} Hz) = hbar w0 sqrt(2 eps) = {m_eff:.2e} eV")
if R.get("joint"):
    ulg = R["joint"]["eps_over_gamma_95_masked_98_114"]
    for f0 in (40.0, 100.0):
        rechi = ulg * f0 / 2
        R["physics"]["peak_ReChi"][str(f0)] = rechi
        print(f"  peak |Re chi| (f0 = {f0:>5.0f} Hz) = (eps/Gamma) f0 / 2 = {rechi:.2e}")
print("  GW170817 speed bound on frequency-independent Re chi: ~5e-16 (literature)")

# ============================================================ write
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o

json.dump(clean(R), open(RESULTS_JSON, "w"), indent=2)
print(f"\nwritten: {RESULTS_JSON}")