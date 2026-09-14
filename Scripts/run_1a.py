"""Validation gate 1a: GR injection, GR recovery, fixed extrinsics.

Injection: 150 + 120 Msun (detector frame), non-spinning, d_L = 4000 Mpc,
H1 + L1 at Advanced LIGO design sensitivity, 4 s at 1024 Hz, 20-256 Hz,
IMRPhenomXPHM, noise seed 88170235. Sampled: chirp_mass, mass_ratio,
luminosity_distance, geocent_time, phase. Sampler: dynesty, nlive 500,
dlogz 0.1.

Pass criterion: every sampled parameter's injected value lies within its
90% credible interval.

Output: outputs/validation_outputs/1a_fixed/. If the result already
exists it is loaded and the gate table is printed without resampling.
"""
import os
import numpy as np
import bilby

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outdir = os.path.join(ROOT, "outputs", "validation_outputs", "1a_fixed")
label = "1a_fixed"

duration, sampling_frequency = 4.0, 1024.0
f_min, f_max = 20.0, 256.0

injection = dict(
    mass_1=150.0, mass_2=120.0,
    a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0,
    phi_12=0.0, phi_jl=0.0,
    luminosity_distance=4000.0,
    theta_jn=0.8, psi=2.659, phase=1.3,
    ra=1.375, dec=-1.2108,
    geocent_time=1242442967.4,
)


def run():
    np.random.seed(88170235)

    wf_args = dict(waveform_approximant="IMRPhenomXPHM",
                   reference_frequency=20.0, minimum_frequency=f_min)
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=wf_args)

    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in ifos:
        ifo.minimum_frequency, ifo.maximum_frequency = f_min, f_max
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=injection["geocent_time"] - duration + 2.0)
    ifos.inject_signal(waveform_generator=waveform_generator,
                       parameters=injection)

    priors = bilby.gw.prior.BBHPriorDict()
    priors["mass_1"] = bilby.core.prior.Constraint(minimum=50, maximum=400)
    priors["mass_2"] = bilby.core.prior.Constraint(minimum=50, maximum=400)
    priors["chirp_mass"] = bilby.core.prior.Uniform(80, 160, name="chirp_mass")
    priors["mass_ratio"] = bilby.core.prior.Uniform(0.25, 1.0, name="mass_ratio")
    priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        1000, 10000, name="luminosity_distance")
    priors["geocent_time"] = bilby.core.prior.Uniform(
        injection["geocent_time"] - 0.1, injection["geocent_time"] + 0.1)
    priors["phase"] = bilby.core.prior.Uniform(0, 2 * np.pi, boundary="periodic")
    for key in ["a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
                "theta_jn", "psi", "ra", "dec"]:
        priors[key] = injection[key]

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator,
        priors=priors, phase_marginalization=False,
        time_marginalization=False, distance_marginalization=False)

    print("mass priors going into sampler:")
    for k in ["mass_1", "mass_2", "chirp_mass", "mass_ratio"]:
        print(f"    {k}: {priors[k]}")

    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler="dynesty",
        nlive=500, dlogz=0.1, npool=4,
        injection_parameters=injection, outdir=outdir, label=label)
    result.plot_corner()
    return result


def gate(result):
    truth = dict(
        chirp_mass=(injection["mass_1"] * injection["mass_2"]) ** 0.6
                   / (injection["mass_1"] + injection["mass_2"]) ** 0.2,
        mass_ratio=injection["mass_2"] / injection["mass_1"],
        luminosity_distance=injection["luminosity_distance"],
        geocent_time=injection["geocent_time"], phase=injection["phase"])
    print("\n===== GATE 1a =====")
    for k, v in truth.items():
        lo, hi = np.percentile(result.posterior[k], [5, 95])
        ok = lo <= v <= hi
        print(f"{k:22s} truth={v:12.4f}  90% CI=[{lo:.4f}, {hi:.4f}]  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"lnBF(signal vs noise) = {result.log_bayes_factor:.2f}")


if __name__ == "__main__":
    path = os.path.join(outdir, f"{label}_result.json")
    if os.path.exists(path):
        print(f"[load] {path}")
        result = bilby.result.read_in_result(path)
    else:
        result = run()
    gate(result)