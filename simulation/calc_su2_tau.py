import numpy as np
from lattice import Lattice2D
from fields import initialize_links, initialize_matter
from updates import MetropolisUpdater
from observables import measure
from p1_execute import corrected_stats

def run_su2_fss(L, n_meas=2000, n_therm=1000, seed=42):
    lat = Lattice2D(L)
    links = initialize_links(lat, group="su2", random=True, rng=np.random.default_rng(seed))
    matter = initialize_matter(lat, group="su2", random=True, rng=np.random.default_rng(seed+1))
    
    updater = MetropolisUpdater(
        lat, links, matter, beta_g=8.0, kappa=0.3,
        gauge_group='SU2', eps_link=0.2, eps_matter=2.0,
        seed=seed+2
    )

    print(f"SU(2) L={L} beta=8.0 kappa=0.3: Thermalizing {n_therm}...")
    updater.thermalize(n_therm, tune_every=200)

    print(f"SU(2) L={L}: Measuring {n_meas}...")
    R_ts = np.empty(n_meas)
    for i in range(n_meas):
        updater.sweep()
        obs = measure(links, matter, lat)
        R_ts[i] = obs.R
        
    mu, s_naive, s_corr, tau = corrected_stats(R_ts)
    print(f"L={L} -> R={mu:.5f} +/- {s_corr:.5f} | tau_int={tau:.2f}")

for L in [4, 6, 8]:
    run_su2_fss(L, seed=200 + L)
