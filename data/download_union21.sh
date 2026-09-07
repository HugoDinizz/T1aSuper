#!/usr/bin/env bash
#
# Download the SCP Union2.1 supernova compilation.
#
#   Source : Supernova Cosmology Project, https://supernova.lbl.gov/Union/
#   Paper  : Suzuki et al. 2012, ApJ 746, 85  (arXiv:1105.3470)
#   Content: 580 SNe Ia passing the release cuts, 0.015 <= z <= 1.414
#
# Files fetched into data/raw/ (gitignored):
#   SCPUnion2.1_mu_vs_z.txt      name, z_CMB, mu, sigma_mu, P(low-mass host)
#   SCPUnion2.1_covmat_sys.txt   580x580 covariance INCLUDING systematics
#   SCPUnion2.1_covmat_nosys.txt 580x580 covariance, statistical only
#
# SHA-256 (verified 2026-09-07):
#   98fc24a6eef71ebd0c90e25b69271c0a3a8ffe23e7e3c2fada37b9334b1101eb  SCPUnion2.1_mu_vs_z.txt
#   4b73e2e99365a768ffb52bb02702b2c294c6af90179b5d48126d530381ad6b8f  SCPUnion2.1_covmat_sys.txt
#   6496c6c761313e253ce1fe1b3cc8990f0b1cfdf9970f26fcc660792eac57f3d7  SCPUnion2.1_covmat_nosys.txt
#
# ---------------------------------------------------------------------
# Format notes (verified 2026-09-07)
# ---------------------------------------------------------------------
# mu_vs_z: 5 '#' header lines recording the standardization applied,
#          then 580 tab-separated rows.  Column 0 is the SN name (a
#          string), so read with usecols=(1, 2, 3, 4).
#          Rows are NOT sorted by redshift -- they follow SN name order.
#          To sort, permute z, mu, sigma AND the covariance together:
#              idx = np.argsort(z); C = C[np.ix_(idx, idx)]
#          Sorting one without the others silently corrupts the fit.
#
# covmat : 580x580, no header, tab-separated, with one trailing tab per
#          line plus a trailing blank line.  Read with plain np.loadtxt
#          and do NOT pass delimiter='\t' -- the trailing tab would be
#          parsed as a 581st empty column.
#          Both matrices are symmetric and positive definite;
#          covmat_sys has eigenvalues in [7.5e-3, 1.02], condition ~137,
#          so scipy.linalg.cho_factor works without regularization.
#
# ---------------------------------------------------------------------
# Physics notes
# ---------------------------------------------------------------------
# The mu column is already standardized.  The file header records the
# values that were applied:  alpha = 0.1219 (stretch), beta = 2.4657
# (colour), delta = -0.0363 (host-mass step), and M_B = -19.308 for
# h = 0.7 (with systematics).  That M_B zero point is arbitrary, which
# is why the fit must keep a free additive offset  M = M_B + 25 +
# 5*log10(c/(H0*Mpc)).  Do not introduce H0 as a separate parameter --
# it is perfectly degenerate with M_B.
#
# The diagonal of each covariance already contains the statistical
# variances (sigma_mu^2 from column 3).  Do NOT add sigma_mu^2 on top.
#
# The sigma_mu range, 0.084 to 1.007 mag, is real: the largest values
# are low-z objects dominated by peculiar-velocity error.  Do not cut
# them -- C^-1 already downweights them.
#
# ---------------------------------------------------------------------
# The server requires HTTPS *and* a browser User-Agent; plain HTTP or a
# default curl UA is refused with a Cloudflare 522.
#
# Usage:  bash data/download_union21.sh

set -euo pipefail

BASE="https://supernova.lbl.gov/Union/figures"
UA="Mozilla/5.0"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/raw"

FILES=(
  "SCPUnion2.1_mu_vs_z.txt"
  "SCPUnion2.1_covmat_sys.txt"
  "SCPUnion2.1_covmat_nosys.txt"
)

mkdir -p "$DEST"

for f in "${FILES[@]}"; do
  out="$DEST/$f"
  if [[ -s "$out" ]]; then
    echo "have    $f"
    continue
  fi
  echo "fetch   $f"
  if ! curl -fsSL -A "$UA" --retry 3 --retry-delay 5 \
       --connect-timeout 30 -o "$out.part" "$BASE/$f"; then
    rm -f "$out.part"
    echo
    echo "Download failed for $f."
    echo "The SCP server returns 522 intermittently. Retry in a while, or"
    echo "download by hand from https://supernova.lbl.gov/Union/ into:"
    echo "  $DEST"
    exit 1
  fi
  mv "$out.part" "$out"
done

# count data rows: strip comments, then strip blank lines
count_rows() {
  grep -v '^#' "$1" | grep -cv '^[[:space:]]*$'
}

echo
echo "--- verification ---"

n_sne=$(count_rows "$DEST/SCPUnion2.1_mu_vs_z.txt")
echo "SCPUnion2.1_mu_vs_z.txt : $n_sne rows (expected 580)"

for f in "${FILES[@]:1}"; do
  path="$DEST/$f"
  rows=$(count_rows "$path")
  cols=$(awk '!/^#/ && NF {print NF; exit}' "$path")
  echo "$f : $rows rows x $cols cols (expected 580 x 580)"
done

echo
echo "--- sha256 ---"
if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$DEST"/*.txt
else
  sha256sum "$DEST"/*.txt
fi

echo
echo "Files are in $DEST (gitignored)."
echo "Compare the hashes above against the header of this script."