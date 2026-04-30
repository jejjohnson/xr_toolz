#!/usr/bin/env bash
# Resume an AEMET scrape inside tmux with a CPU keepalive sidecar.
#
# Why a keepalive: Azure ML's dsiidlestopagent monitors CPU activity to
# decide when to shut down idle compute. Long AEMET scrapes are mostly
# blocked in HTTP waits / rate-limit gates, so CPU stays near zero and
# Azure has previously killed running scrapes mid-period. A trivial
# busy-loop pinned to 0.5% keeps the node alive without affecting the
# scrape's network budget.
#
# Usage:
#   scripts/aemet_resume.sh monthly --start 1935
#   scripts/aemet_resume.sh daily --start 1950 --end 1999
#   scripts/aemet_resume.sh smoke
#
# Inside the resulting tmux session, detach with Ctrl-b d. Re-attach
# with `tmux attach -t aemet-<flavor>`. Logs at xr_toolz/.logs/.

set -euo pipefail

flavor="${1:?usage: $0 <smoke|monthly|daily> [extra args...]}"
shift || true

case "$flavor" in
  smoke|monthly|daily) ;;
  *) echo "unknown flavor: $flavor (expected smoke|monthly|daily)" >&2; exit 2 ;;
esac

session="aemet-${flavor}"
script="scripts/aemet_${flavor}.py"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -z "${AEMET_SCRATCH_ROOT:-}" ]]; then
    echo "AEMET_SCRATCH_ROOT is not set." >&2
    echo "Set it (recommended in ~/.bashrc) to keep data out of the repo:" >&2
    echo "  export AEMET_SCRATCH_ROOT=\$HOME/cloudfiles/code/Users/adm.jjohnson72/scratch/aemet" >&2
    exit 1
fi

if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session '$session' already exists. Attach with: tmux attach -t $session" >&2
    exit 1
fi

# spawn detached: keepalive in the background, scrape in the foreground
tmux new-session -d -s "$session" -c "$repo_root" "
    set -e
    export AEMET_SCRATCH_ROOT='${AEMET_SCRATCH_ROOT}'
    ( while true; do : ; sleep 30; done ) &
    keepalive_pid=\$!
    trap 'kill \$keepalive_pid 2>/dev/null || true' EXIT
    uv run python ${script} $*
    echo
    echo '--- scrape exited; press any key to close ---'
    read -n 1
"

echo "started tmux session: $session"
echo "  attach:  tmux attach -t $session"
echo "  log:     tail -f ${repo_root}/.logs/aemet_${flavor}.log"
