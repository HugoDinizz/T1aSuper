# Download the SCP Union2.1 supernova compilation.
#
#   Source : Supernova Cosmology Project, https://supernova.lbl.gov/Union/
#
# Files fetched into data/raw/ (gitignored):
#   SCPUnion2.1_mu_vs_z.txt      name, z_CMB, mu, sigma_mu, P(low-mass host)
#   SCPUnion2.1_covmat_sys.txt   580x580 covariance INCLUDING systematics
#   SCPUnion2.1_covmat_nosys.txt 580x580 covariance, statistical only
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