set -euo pipefail
cd "$(dirname "$0")"
for fold in 1 2 3 4 5; do
    python train.py --fold "$fold" "$@"
done
