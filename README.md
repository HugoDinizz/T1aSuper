# T1aSuper

Numerical Cosmology course exercise.

Type Ia supernova cosmology: inferring $(\Omega_m, \Omega_\Lambda)$ from the
Union2.1 compilation with a Metropolis–Hastings MCMC sampler,
without imposing flatness.

## Installation

```bash
git clone git@github.com:HugoDinizz/T1aSuper.git
cd T1aSuper
conda env create -f environment.yml
conda activate t1asuper
pip install -e .
```

The last step is always required: it links `src/snia/` to the interpreter
of that environment. It is not recorded in `environment.yml`.

Verify:

```bash
python -c "import snia; print(snia.__file__)"
pytest
```

The import must resolve to a path under `src/snia/`.

## Data

The Union2.1 compilation is not tracked in this repository. Download it with:

```bash
bash data/download_union21.sh
```

This fetches three files into `data/raw/` (gitignored) from the
[Supernova Cosmology Project](https://supernova.lbl.gov/Union/):

| File | Content |
|---|---|
| `SCPUnion2.1_mu_vs_z.txt` | 580 SNe: name, $z_{\rm CMB}$, $\mu$, $\sigma_\mu$, $P(\text{low-mass host})$ |
| `SCPUnion2.1_covmat_sys.txt` | $580\times580$ covariance **including systematics** |
| `SCPUnion2.1_covmat_nosys.txt` | $580\times580$ covariance, statistical only |

The script verifies row and column counts and prints SHA-256 checksums,
which are recorded in its header — compare them to confirm you have the
same bytes.

If the download fails with a Cloudflare 522, the SCP server is
intermittently unreachable. Retry later, or download by hand from the
link above into `data/raw/`.

## Assumptions

- $\Omega_r = 0$. Negligible at $z \lesssim 1.5$.
- Curvature is **not** fixed: $\Omega_k = 1 - \Omega_m - \Omega_\Lambda$ is
  derived, not sampled, and the $S_K$ branches are implemented in full.
- $H_0$ is **not** a free parameter. It is perfectly degenerate with the
  SN absolute magnitude $M_B$; both enter only through the additive
  offset $\mathcal{M} = M_B + 25 + 5\log_{10}(c/H_0\,\mathrm{Mpc})$.
- The released $\mu$ values are already standardized with an arbitrary
  fiducial $M_B$, so $\mathcal{M}$ must stay free.
- The covariance diagonal already contains $\sigma_\mu^2$; it is never
  added again.
- Final results use the covariance **with systematics**.
- $\chi^2$ is evaluated by Cholesky solve; $C^{-1}$ is never formed.

## Layout