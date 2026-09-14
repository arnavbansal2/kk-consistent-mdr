"""Figure 8: validation ladder on simulated data.
(a) Gate 1a chirp-mass recovery, (b) gate 1b-loud f0 recovery,
(c) gate 1b-loud (Gamma, eps) posterior against the injected values.
Run from the repo root: python figures/fig_validation_ladder.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import bilby

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAL = os.path.join(ROOT, "outputs", "validation_outputs")
OUT = os.path.join(ROOT, "figures", "final")

MC_TRUE = (150.0 * 120.0) ** 0.6 / (150.0 + 120.0) ** 0.2
EPS_TRUE, GAM_TRUE, F0_TRUE = 2.5e-21, 10.0, 60.0

r1a = bilby.result.read_in_result(os.path.join(VAL, "1a_fixed", "1a_fixed_result.json"))
r1b = bilby.result.read_in_result(os.path.join(VAL, "1b_loud_v3", "1b_loud_v3_result.json"))

Mc = r1a.posterior["chirp_mass"].values
eps = r1b.posterior["eps"].values
gam = r1b.posterior["gamma_res"].values
f0 = r1b.posterior["f0_res"].values

# numbers quoted in the text
mc_lo, mc_hi = np.percentile(Mc, [5, 95])
print(f"1a  chirp mass: {np.median(Mc):.1f} +{mc_hi-np.median(Mc):.1f} "
      f"-{np.median(Mc)-mc_lo:.1f} Msun  (truth {MC_TRUE:.1f})")
print(f"1a  lnBF(signal vs noise) = {r1a.log_bayes_factor:.1f}")
f0_lo, f0_hi = np.percentile(f0, [5, 95])
print(f"1b  f0: {np.median(f0):.1f} +{f0_hi-np.median(f0):.1f} "
      f"-{np.median(f0)-f0_lo:.1f} Hz  (truth {F0_TRUE:.0f})")
e_lo, e_hi = np.percentile(eps, [5, 95])
g_lo, g_hi = np.percentile(gam, [5, 95])
print(f"1b  eps 90% CI [{e_lo:.2e}, {e_hi:.2e}]  (truth {EPS_TRUE:.1e})")
print(f"1b  gamma 90% CI [{g_lo:.2f}, {g_hi:.2f}] Hz  (truth {GAM_TRUE:.0f}; prior floor 1 Hz)")

# figure
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))

ax[0].hist(Mc, bins=40, color="steelblue", edgecolor="none", alpha=0.85)
ax[0].axvline(MC_TRUE, color="r", ls="--", lw=1.5, label=f"truth = {MC_TRUE:.1f}")
ax[0].set_xlabel(r"chirp mass [$M_\odot$]")
ax[0].set_ylabel("posterior samples")
ax[0].set_title("(a) Stage 1a: GR recovery")
ax[0].set_xlim(98, 130)
ax[0].legend(fontsize=8)

ax[1].hist(f0, bins=np.linspace(59.0, 61.0, 80), color="steelblue", edgecolor="none", alpha=0.85)
ax[1].axvline(F0_TRUE, color="r", ls="--", lw=1.5, label=f"injected {F0_TRUE:.0f} Hz")
ax[1].set_xlim(59.0, 61.0)
ax[1].set_xlabel(r"$f_0$ [Hz]")
ax[1].set_ylabel("posterior samples")
ax[1].set_title(r"(b) Stage 1b: $f_0$ recovered $<1$ Hz")
ax[1].legend(fontsize=8)

ax[2].scatter(np.log10(gam), np.log10(eps), s=2, alpha=0.25,
              color="steelblue", rasterized=True)
ax[2].scatter(np.log10(GAM_TRUE), np.log10(EPS_TRUE), marker="*", s=150,
              color="r", zorder=5, label="truth")
ax[2].scatter(np.log10(np.median(gam)), np.log10(np.median(eps)), marker="+",
              s=80, color="k", zorder=5, lw=1.5, label="median")
ax[2].set_xlabel(r"$\log_{10}\,\Gamma$ [Hz]")
ax[2].set_ylabel(r"$\log_{10}\,\epsilon$")
ax[2].set_title(r"(c) Stage 1b: $(\Gamma,\epsilon)$ posterior")
ax[2].legend(fontsize=8, loc="upper left")

os.makedirs(OUT, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_validation_ladder.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(OUT, "fig_validation_ladder.png"), dpi=150, bbox_inches="tight")
print(f"saved {OUT}/fig_validation_ladder.pdf/.png")