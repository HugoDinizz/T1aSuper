# T1aSuper

Numerical Cosmology course exercise.

Type Ia supernova cosmology: inferring $(\Omega_m, \Omega_\Lambda)$ from the
Union2.1 compilation with a Metropolis–Hastings MCMC sampler,
without imposing flatness.

## Installation

T1aSuper requires Python 3.11 or newer. Conda is not required.

### Recommended: Python virtual environment

Clone the repository and create a local virtual environment:

```bash
git clone https://github.com/HugoDinizz/T1aSuper.git
cd T1aSuper

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

Using `python -m pip` ensures that `pip` is invoked through the Python interpreter of the active virtual environment. This avoids ambiguity when multiple Python installations are present on the system, such as system Python, Conda, or Homebrew Python.

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

The `pip install -e .` step installs the package and its runtime dependencies and links `src/snia/` to the Python interpreter of the virtual environment.

Verify the installation with:

```bash
python -c "import snia; print(snia.__file__)"
python -m pytest
```

The import should resolve to a path under `src/snia/`.

### Conda (optional)

A Conda environment is also provided for users who prefer Conda:

```bash
conda env create -f environment.yml
conda activate t1asuper
python -m pip install -e .
```

The `environment.yml` reproduces the Conda environment used for the project, while the `pyproject.toml` defines the Python package and its dependencies.

## Data

The Union2.1 compilation is not tracked in this repository. Download it with:

```bash
bash data/download_union21.sh
```

This fetches three files into `data/raw/` (gitignored) from the
[Supernova Cosmology Project](https://supernova.lbl.gov/Union/):

| File | Content |
| --- | --- |
| `SCPUnion2.1_mu_vs_z.txt` | 580 SNe: name, $z_{\rm CMB}$, $\mu$, $\sigma_\mu$, $P(\text{low-mass host})$ |
| `SCPUnion2.1_covmat_sys.txt` | $580\times580$ covariance **including systematics** |
| `SCPUnion2.1_covmat_nosys.txt` | $580\times580$ covariance, statistical only |

If the download fails with a Cloudflare 522, the SCP server is
intermittently unreachable. Retry later, or download by hand from the
link above into `data/raw/`.

## Assumptions

- $\Omega_r = 0$. Negligible at $z \lesssim 1.5$.
- Curvature is **not** fixed: $\Omega_k = 1 - \Omega_m - \Omega_\Lambda$ is
  derived, not sampled, and the $S_K$ branches are implemented in full.
- $H_0$ is **not** a free parameter. It is perfectly degenerate with the
  SN absolute magnitude $M_B$; both enter only through the additive
  offset $\mathcal{M} = M_B + 25 + 5\log_{10}(c/H_0,\mathrm{Mpc})$.
- The released $\mu$ values are already standardized with an arbitrary
  fiducial $M_B$, so $\mathcal{M}$ must stay free.
- The covariance diagonal already contains $\sigma_\mu^2$; it is never
  added again.
- Final results use the covariance **with systematics**.
- $\chi^2$ is evaluated by Cholesky solve; $C^{-1}$ is never formed.

## Layout

```text
src/snia/
├── cosmology.py        FLRW: E, chi, D_M, D_A, D_L, mu
├── bayes/              Posterior, Proposal, MetropolisHastings
├── diagnostics/        autocorrelation, split R-hat, Dunkley spectrum
└── data/               Union2.1 loader and likelihood

tests/                  acceptance tests
scripts/                reproducible pipeline
notebooks/              narrative and figures
```

# The supernova zero point

Supernova magnitudes constrain the *shape* of the distance–redshift relation. The theoretical distance modulus is

$$
\mu_{\rm th}(z) = 5\log_{10}\left(\frac{D_L(z)}{\rm Mpc}\right) + 25,
\qquad
D_L(z) = \frac{c}{H_0}(1+z)\,S_K\left[\int_0^z \frac{dz'}{E(z')}\right].
$$

All of the $H_0$ dependence sits in the prefactor $c/H_0$, and all of the cosmological *shape* information sits in the dimensionless quantity

$$
d_L(z;\Omega_m,\Omega_\Lambda) \equiv (1+z)\,S_K\left[\int_0^z \frac{dz'}{E(z')}\right].
$$

Because the logarithm turns the product into a sum,

$$
\mu_{\rm th}(z) = 5\log_{10} d_L(z;\Omega_m,\Omega_\Lambda) + \mathcal{M},
$$

with

$$
\mathcal{M} \equiv 25 + 5\log_{10}\left(\frac{c}{H_0\,{\rm Mpc}}\right) + M_B.
$$

On the data side, converting an observed peak magnitude into a distance modulus requires the absolute magnitude of a fiducial SN Ia, $M_B \approx -19.3$, which is not predicted by theory and must be calibrated externally. The released Union2.1 $\mu$ column already has *some* fiducial $M_B$ subtracted, but that choice is arbitrary and imports an $H_0$ assumption that we do not want. This analysis therefore treats $\mathcal{M}$ as a nuisance parameter and marginalizes it analytically, leaving $(\Omega_m,\Omega_\Lambda)$ as the sampled parameter space.

### How it is marginalized

Define the offset-free residual

$$
\Delta_i = \mu_i^{\rm obs} - 5\log_{10} d_L(z_i;\Omega_m,\Omega_\Lambda),
$$

so the full residual is $\Delta - \mathcal{M}\mathbf{1}$ with $\mathbf{1} = (1,\dots,1)^T$. Expanding the chi-square gives a quadratic in $\mathcal{M}$:

$$
\chi^2(\mathcal{M}) = A - 2\mathcal{M}B + \mathcal{M}^2 E,
$$

$$
A = \Delta^T \mathsf{C}^{-1}\Delta,
\qquad
B = \mathbf{1}^T \mathsf{C}^{-1}\Delta,
\qquad
E = \mathbf{1}^T \mathsf{C}^{-1}\mathbf{1}.
$$

Integrating $\mathcal{M}$ out under a flat prior on $(-\infty,\infty)$ is a Gaussian integral. Completing the square,

$$
\chi^2_{\rm marg} = A - \frac{B^2}{E} + \ln E.
$$

$E$ depends only on the covariance and the vector of ones, so it is **independent of the cosmological parameters**. The $\ln E$ term is therefore an additive constant and is dropped:

$$
\boxed{\chi^2 = A - \frac{B^2}{E}}
$$

The sampled parameter space is therefore two-dimensional, $(\Omega_m,\Omega_\Lambda)$.

Notice that the prior on $\mathcal{M}$ is flat and improper. The resulting posterior on $(\Omega_m,\Omega_\Lambda)$ is nevertheless proper, because the integrand is Gaussian in $\mathcal{M}$ with positive curvature $E > 0$, so the integral converges. What is given up is the Bayesian evidence, which inherits the arbitrary normalization of an improper prior; model comparison via Bayes factors would require a proper prior. Parameter estimation is unaffected.

### Cost

$E$ and $v \equiv \mathsf{C}^{-1}\mathbf{1}$ are parameter-independent and are computed once, outside the likelihood. Each likelihood call then needs one triangular solve, $y = \mathsf{L}^{-1}\Delta$, giving $A = |y|^2$ and $B = v^T \Delta$. That is exactly the cost of the non-marginalized version — the marginalization is free.

References: Amanullah et al. 2010, ApJ 716, 712, Appendix C.