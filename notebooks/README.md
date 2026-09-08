# Examples

Five notebooks, in order. They build on each other: notebook 1 is the theory,
2 tours the cosmology module, 3 adds the data and the likelihood, 4 samples it,
and 5 checks whether the sampling converged.

All are stored **already executed**, with outputs and figures, so they can be
read without running anything.

| notebook | what it covers |
| --- | --- |
| [`01_distance_modulus_theory.ipynb`](01_distance_modulus_theory.ipynb) | From the FLRW metric to $\mu(z)$: comoving distance, why distances here are dimensionless, the three curvature branches as one series, $D_A$ and $D_L$, Etherington duality, and the $H_0$–$M_B$ degeneracy. |
| [`02_cosmology_tour.ipynb`](02_cosmology_tour.ipynb) | Every method of `FLRW`: distances, cosmic time, comoving volume, $q(z)$, and which parameter values `is_valid` rejects. |
| [`03_union21_likelihood.ipynb`](03_union21_likelihood.ipynb) | Loading Union2.1, the covariance, the analytic marginalization over $\mathcal{M}$, and the $\chi^2$ surface on a grid. |
| [`04_bayes_sampler.ipynb`](04_bayes_sampler.ipynb) | Metropolis–Hastings: proposal scale and geometry, why a rejection is still a sample, what the Hastings ratio does, then the real posterior by MCMC. |
| [`05_diagnostics.ipynb`](05_diagnostics.ipynb) | Convergence: autocorrelation and effective sample size, split-$\hat{R}$, and the Dunkley power spectrum — each checked against an AR(1) process whose answers are analytic, then applied to the real chains. |

## Running them

The notebooks need the `snia` package importable. Activate the environment
first, then launch from it:

```bash
conda activate t1asuper
jupyter lab examples/
```

In VSCode, pick the `t1asuper` interpreter in the kernel selector — the
generic "Python 3" kernel follows whatever `python` is first on `PATH`, which
may not be the right environment.

Notebooks 3 and 4 need the raw data, which is gitignored:

```bash
bash data/download_union21.sh
```

To re-run everything:

```bash
python -m nbconvert --to notebook --execute --inplace examples/*.ipynb
```

Notebooks 4 and 5 take about a minute each; the others are seconds.

## Results

From notebook 3, mapping $\chi^2$ on a grid:

- best fit $(\Omega_m, \Omega_\Lambda) = (0.293,\ 0.693)$, $\chi^2/\mathrm{dof} = 0.945$
- assuming flatness, $\Omega_m = 0.295$ with a 68% interval $[0.256,\ 0.337]$
  (published Union2.1: $0.295 \pm 0.034$)
- Einstein–de Sitter excluded at $\Delta\chi^2 = 128$
- the fitted zero point gives $H_0 = 69.7$ km/s/Mpc, recovering the $h = 0.7$
  the release assumed

From notebook 4, by MCMC on the same posterior:

- $\Omega_m = 0.280 \pm 0.112$, $\Omega_\Lambda = 0.659 \pm 0.210$
- sampling $\mathcal{M}$ as a third parameter instead of marginalizing it
  analytically gives the same contours, which is the check that the
  marginalization has no sign error

From notebook 5, the convergence report for those chains:

- $\hat{R} = 1.0003$ and $1.0006$ for the two parameters
- $\tau_{\rm int} \approx 7.6$, so 72000 draws are worth about 9500 independent
  ones; MCSE is 0.0012 on $\Omega_m$
- the Dunkley test passes on all four chains for both parameters

## A few things worth noticing

- **The flat line is not assumed.** $\Omega_k = 1 - \Omega_m - \Omega_\Lambda$ is
  derived, and flatness is just a line through the allowed region.
- **A closed universe can make $d_L$ decrease with redshift.** The geometry
  focuses light, so more distant sources can appear brighter. Notebook 2.
- **The antipode rejection is not a corner case.** It covers 1.6% of the prior
  box against 14.3% for the no-Big-Bang wedge: models that nearly stall at a
  turning point keep $E^2 > 0$ while $I$ grows past $\pi/\sqrt{|\Omega_k|}$.
- **You cannot replace $\chi^2$ by $\sum(\Delta\mu/\sigma)^2$.** That gives
  511.8 instead of 545.1, because much of the systematic variance is
  common-mode and does not show up as per-object scatter.
- **Omitting the Hastings ratio produces a chain that looks fine and is wrong**
  — stationary trace, sensible acceptance rate, and a distribution 20% too
  narrow. Notebook 4 shows it side by side with the truth.
- **Chain length is not sample size.** At $\tau_{\rm int} = 19$, 200000 draws
  carry the information of about 10000. Notebook 5.
- **Split-$\hat{R}$ misses a chain that is too wide but correctly centred**
  (1.0004), while the rank-normalized and folded version catches it (1.148).
  Report the second one.

## Not covered yet

The reproducible end-to-end pipeline, `scripts/run_union21.py`, which should
run the whole analysis from the raw files to the final figures and the
convergence report.
