#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME BASH_ENV ENV
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
completed=0
finish() {
    status=$?
    trap - EXIT INT TERM HUP
    if [ "$completed" -ne 1 ] || [ "$status" -ne 0 ]; then
        /opt/venv/bin/python -I /tests/initialize_reward.py || true
        [ "$status" -ne 0 ] || status=2
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
/opt/venv/bin/python -I /tests/initialize_reward.py
timeout 30 /opt/venv/bin/python -I /tests/prepare_verifier.py
/opt/venv/bin/python -I /tests/initialize_reward.py
HOME=/root PIP_CONFIG_FILE=/dev/null timeout 60 /opt/venv/bin/python -I -m pip --isolated install --no-index --find-links=/tests/wheels --only-binary=:all: --disable-pip-version-check --no-cache-dir pytest==8.3.5 iniconfig==2.1.0 packaging==25.0 pluggy==1.5.0
/opt/venv/bin/python -I /tests/ratio_verifier.py --manifest /tests/test-manifest.json --output-dir /logs/verifier
completed=1
