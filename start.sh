#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Use The Project Virtual Environment
source .venv/bin/activate

# Stop Both Process Groups, Including Their Child Processes
pids=()
cleanup() {
    for pid in "${pids[@]}"; do
        kill -- "-$pid" 2>/dev/null || true
    done
    wait || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Start The API & Frontend In Separate Process Groups
setsid python3 -m uvicorn classifier_pytorch.api.main:app --host 127.0.0.1 --port 8000 &
pids+=("$!")
setsid npm --prefix frontend run dev &
pids+=("$!")

# Stop The Other Service If Either One Exits
wait -n "${pids[@]}"
