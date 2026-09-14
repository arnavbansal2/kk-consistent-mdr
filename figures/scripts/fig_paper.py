"""All paper figures except the validation ladder (fig08). Run from repo root:
    python figures/scripts/fig_paper.py
Writes PNGs to figures/final/ (same plotting code as duag.py / reader.py).
"""
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import bilby
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_catalog as rc

OUT = os.path.join(ROOT, "figures", "final")
JOINT = os.path.join(ROOT, "outputs", "joint_fit_outputs")
os.makedirs(OUT, exist_ok=True)
EVENTS = list(rc.QUEUE)
NEV = len(EVENTS)
THRESHOLD = rc.THRESHOLD
plt.rcParams["legend.handlelength"] = 2.5

def load(ev, model):
    lab = f"{ev}_{model}_v1"
    return bilby.result.read_in_result(os.path.join(rc.OUTDIR, lab, f"{lab}_result.json"))


def prof_f0(post, fb=np.linspace(20, 200, 46)):
    f0, L = post["f0_res"].values, post["log_likelihood"].values
    c, p = [], []
    for lo, hi in zip(fb[:-1], fb[1:]):
        m = (f0 >= lo) & (f0 < hi)
        if m.sum() > 5:
            c.append(.5 * (lo + hi)); p.append(L[m].max())
    return np.array(c), np.array(p)


def shade(ax):
    ax.axvspan(115, 131, color="r", alpha=.12)
    ax.axvspan(66, 77, color="purple", alpha=.12)


def chieff(g):
    Mc, q = g["chirp_mass"], g["mass_ratio"]
    m1 = Mc * (1 + q) ** .2 / q ** .6; m2 = q * m1
    return (m1 * g["a_1"] * np.cos(g["tilt_1"]) + m2 * g["a_2"] * np.cos(g["tilt_2"])) / (m1 + m2)


def annotate_offset(ax, xs, ys, labels, fontsize=6):
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
        dx, dy = (3, 3) if i % 2 == 0 else (3, -9)
        ax.annotate(lab, (x, y), fontsize=fontsize, xytext=(dx, dy), textcoords="offset points")


def wq95(values, weights):
    o = np.argsort(values)
    cc = np.cumsum(weights[o]) / weights[o].sum()
    return values[o][np.searchsorted(cc, 0.95)]


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {name}.png")


G = {ev: load(ev, "gr") for ev in EVENTS}
K = {ev: load(ev, "kernel") for ev in EVENTS}
print(f"loaded {NEV} events")

# ---------- profiles grid (reader.py) ----------
ncol = 4; nrow = int(np.ceil(NEV / ncol))
fig, ax = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow), sharex=True, sharey=True)
for a, ev in zip(np.ravel(ax), EVENTS):
    c, p = prof_f0(K[ev].posterior)
    a.plot(c, p - p.max(), "-", lw=1)
    a.axhline(-THRESHOLD, color="r", ls=":", lw=1.2)
    a.set_title(ev, fontsize=8)
for a in np.ravel(ax)[NEV:]:
    a.axis("off")
fig.supxlabel(r"$f_0$ [Hz]"); fig.supylabel(r"profile $\ln\mathcal{L}$ - max")
save("fig_profiles_grid")

# ---------- fig_gw151226_kernel (duag.py) ----------
pk = K["GW151226"].posterior
eps, f0, L = pk["eps"].values, pk["f0_res"].values, pk["log_likelihood"].values
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
be = np.logspace(-25, -19, 30); c, pr = [], []
for lo, hi in zip(be[:-1], be[1:]):
    m = (eps >= lo) & (eps < hi)
    if m.sum() > 5:
        c.append(np.sqrt(lo * hi)); pr.append(L[m].max())
c, pr = np.array(c), np.array(pr)
ax[0].plot(np.log10(c), pr - pr.max(), "o-")
ax[0].axhline(-THRESHOLD, color="r", ls=":")
ax[0].set(xlabel=r"$\log_{10}\epsilon$", ylabel=r"profile $\ln\mathcal{L}$ - max",
          title=r"peaked $\epsilon$ profile")
cf, pf = prof_f0(pk)
ax[1].plot(cf, pf - pf.max(), "o-", ms=4); shade(ax[1])
ax[1].axhline(-THRESHOLD, color="r", ls=":")
ax[1].set(xlabel=r"$f_0$ [Hz]", title=r"$f_0$ profile")
ax[2].scatter(f0, np.log10(eps), s=2, alpha=.2); shade(ax[2])
ax[2].set(xlabel=r"$f_0$ [Hz]", ylabel=r"$\log_{10}\epsilon$", title="posterior structure")
save("fig_gw151226_kernel")

# ---------- fig_gw170608 (duag.py) ----------
pg = G["GW170608"].posterior
pk8 = K["GW170608"].posterior
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for a, (x, pub, lab) in zip(ax[0], [(pg["chirp_mass"], 8.5, r"chirp mass [$M_\odot$]"),
                                    (chieff(pg), 0.03, r"$\chi_{\rm eff}$"),
                                    (pg["luminosity_distance"], 320, r"$d_L$ [Mpc]")]):
    a.hist(x, bins=50); a.axvline(pub, color="r", ls=":", label=f"pub ~{pub}")
    a.set(xlabel=lab); a.legend()
e8, f8, L8 = pk8["eps"].values, pk8["f0_res"].values, pk8["log_likelihood"].values
bins = np.linspace(-25, -19, 50)
ax[1, 0].hist(np.log10(e8), bins=bins, density=True, alpha=.7, label="posterior")
ax[1, 0].hist(np.random.uniform(-25, -19, 100000), bins=bins, density=True, alpha=.3, label="prior")
ax[1, 0].axvline(np.log10(np.percentile(e8, 95)), color="k", ls="--", label="95% UL")
ax[1, 0].set(xlabel=r"$\log_{10}\epsilon$"); ax[1, 0].legend(fontsize=8)
cc, pp = prof_f0(pk8)
ax[1, 1].plot(cc, pp - pp.max(), "o-", ms=4); shade(ax[1, 1])
ax[1, 1].axhline(-THRESHOLD, color="r", ls=":")
ax[1, 1].set(xlabel=r"$f_0$ [Hz]", title="flat through both windows")
ax[1, 2].scatter(f8, np.log10(e8), s=2, alpha=.2); shade(ax[1, 2])
ax[1, 2].set(xlabel=r"$f_0$ [Hz]", ylabel=r"$\log_{10}\epsilon$")
plt.suptitle("GW170608: the clean-data control", y=1.01)
save("fig_gw170608")

# ---------- fig_forensics (duag.py; recorded battery results) ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].bar(["full\nkernel", "amplitude\nonly", "phase\nonly"], [8.07, 1.63, 11.10], color=["C0", "C1", "C2"])
ax[0].axhline(0, color="k", lw=.5)
ax[0].set(ylabel=r"$\Delta\ln\mathcal{L}$ at cluster maximum", title="KK decomposition (GW151226)")
nulls = [0.74, 4.81, -0.50, -0.84, 0.42, -1.23, -0.27, -0.58, 1.61, 0.50]
ax[1].hist(nulls, bins=10)
ax[1].axvline(5.3, color="r", ls="--", label="on-source (+5.3)")
ax[1].set(xlabel="max spurious kernel gain (adjacent noise)", title="event-local null (n=10)")
ax[1].legend()
save("fig_forensics")

# ---------- joint figures (reader.py [C]) ----------
th_path, lw_path = os.path.join(JOINT, "joint_theta_val.npy"), os.path.join(JOINT, "joint_logw_val.npy")
fb = np.linspace(20, 200, 25); fc = .5 * (fb[:-1] + fb[1:])
D_mpc, U_post, names = [], [], []
for ev in EVENTS:
    k = K[ev].posterior
    e_, f_ = k["eps"].values, k["f0_res"].values
    D_mpc.append(float(np.median(k["luminosity_distance"])))
    U_post.append(float(np.percentile(e_[f_ > 40], 95)))
    names.append(ev.replace("GW", ""))
D_mpc, U_post = np.array(D_mpc), np.array(U_post)

if os.path.exists(th_path) and os.path.exists(lw_path):
    TH = np.load(th_path); lw_ = np.load(lw_path)
    w = np.exp(lw_ - lw_.max())
    m40 = TH[:, 1] > 40
    ul = wq95(TH[m40, 0], w[m40])
    ess = w.sum() ** 2 / (w * w).sum()
    print(f"JOINT ({NEV} events): eps < {ul:.2e} (f0>40, 95%), ESS={ess:.0f}/{len(TH)}")
    jcurve = [wq95(TH[m, 0], w[m]) if (m := (TH[:, 1] >= lo) & (TH[:, 1] < hi)).sum() > 30 else np.nan
              for lo, hi in zip(fb[:-1], fb[1:])]

    plt.figure(figsize=(10, 4.2))
    plt.semilogy(fc, jcurve, "k-o", ms=4, lw=2)
    plt.xlabel(r"$f_0$ [Hz]"); plt.ylabel(rf"$\epsilon_{{95}}$ (joint, {NEV} events)")
    save("fig_joint_exclusion")

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for ev in EVENTS:
        k = K[ev].posterior
        e_, f_ = k["eps"].values, k["f0_res"].values
        cur = [np.percentile(e_[m], 95) if (m := (f_ >= lo) & (f_ < hi)).sum() > 40 else np.nan
               for lo, hi in zip(fb[:-1], fb[1:])]
        ax[0].semilogy(fc, cur, "-", lw=.8, alpha=.35)
    ax[0].semilogy(fc, jcurve, "k-", lw=2.5, label="joint (weighted)")
    ax[0].set(xlabel=r"$f_0$ [Hz]", ylabel=r"$\epsilon_{95}$", title=f"{NEV} per-event curves + joint")
    ax[0].legend(fontsize=8)
    ax[1].scatter(TH[:, 1], np.log10(TH[:, 0]), c=np.log(w + 1e-300), s=8)
    ax[1].set(xlabel=r"$f_0$ [Hz]", ylabel=r"$\log_{10}\epsilon$", title="joint posterior (sample-weighted)")
    ax[2].loglog(D_mpc, U_post, "v", ms=8)
    annotate_offset(ax[2], D_mpc, U_post, names)
    dd = np.geomspace(0.8 * D_mpc.min(), 1.3 * D_mpc.max(), 50)
    ax[2].loglog(dd, np.median(U_post * D_mpc) / dd, "k:", label=r"fixed-$\tau$ scaling $\propto 1/D$")
    ax[2].set(xlabel=r"$D$ [Mpc]", ylabel=r"$\epsilon_{95}$ ($f_0>40$)",
              title=f"all {NEV} events (sampled posteriors)")
    ax[2].legend(fontsize=8)
    save("fig_combination")

    # distance scaling of the weighted per-event ULs (reader.py [E4])
    ck_files = sorted(glob.glob(os.path.join(JOINT, "joint_dlnL_*_val.npy")))
    UL_w, D_w, n_w = [], [], []
    for f in ck_files:
        ev = os.path.basename(f).split("joint_dlnL_")[1].rsplit("_val.npy", 1)[0]
        if ev not in K:
            continue
        d = np.load(f)
        lw = logsumexp(d, axis=1) - np.log(d.shape[1])
        ww = np.exp(lw - lw.max())
        UL_w.append(wq95(TH[m40, 0], ww[m40]))
        D_w.append(float(np.median(K[ev].posterior["luminosity_distance"])))
        n_w.append(ev.replace("GW", ""))
    if UL_w:
        from matplotlib.ticker import NullFormatter
        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        D_w, UL_w = np.array(D_w), np.array(UL_w)
        (slope, icept), cov = np.polyfit(np.log10(D_w), np.log10(UL_w), 1, cov=True)
        print(f"fitted slope: {slope:+.2f} +/- {np.sqrt(cov[0, 0]):.2f}  "
              f"(pure tau-accumulation predicts -1.00)   [{len(D_w)} events]")
        carriers = {"151226", "170814", "190408_181802", "190814", "200311_115853"}
        keep = np.array([n not in carriers for n in n_w])
        (s2, _icept2), c2 = np.polyfit(np.log10(D_w[keep]), np.log10(UL_w[keep]), 1, cov=True)
        print(f"excluding feature-carriers: slope {s2:+.2f} +/- {np.sqrt(c2[0, 0]):.2f}  "
              f"[{keep.sum()} events]")
        plt.figure(figsize=(7, 5))
        plt.loglog(D_w, UL_w, "v", ms=8)
        annotate_offset(plt.gca(), D_w, UL_w, n_w)
        dd = np.geomspace(0.8 * D_w.min(), 1.3 * D_w.max(), 50)
        plt.loglog(dd, 10 ** icept * dd ** slope, "b--", lw=1, label=f"fit: slope {slope:+.2f}")
        plt.loglog(dd, np.median(UL_w * D_w) / dd, "k:", label=r"$\propto 1/D$")
        plt.loglog(dd, 10 ** _icept2 * dd ** s2, "g--", lw=1,
                label=f"null events only: slope {s2:+.2f}")
        plt.xlabel("D [Mpc]"); plt.ylabel(r"$\epsilon_{95}$ (weighted, $f_0>40$)")
        plt.legend()
        save("fig_distance_scaling")
    else:
        print("[no joint_dlnL_*_val.npy checkpoints -- fig_distance_scaling skipped]")
else:
    print("[joint_theta.npy / joint_logw.npy not in outputs/joint_fit_outputs -- joint figures skipped]")

print("\ndone ->", OUT)