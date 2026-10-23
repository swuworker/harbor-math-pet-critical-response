#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME BASH_ENV ENV
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
completed=0
stage=bootstrap
finish() {
    status=$?
    trap - EXIT INT TERM HUP
    if [ "$completed" -ne 1 ] || [ "$status" -ne 0 ]; then
        printf 'Trusted verifier failed at stage=%s exit=%s\n' "$stage" "$status" >&2
        /opt/venv/bin/python -I /tests/initialize_reward.py || true
        /opt/venv/bin/python -I /tests/record_failure.py "$stage" "$status" || true
        [ "$status" -ne 0 ] || status=2
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
/opt/venv/bin/python -I /tests/initialize_reward.py
stage=prepare
timeout 30 /opt/venv/bin/python -I /tests/prepare_verifier.py
/opt/venv/bin/python -I /tests/initialize_reward.py
stage=test_dependencies
HOME=/root PIP_CONFIG_FILE=/dev/null timeout 60 /opt/venv/bin/python -I -m pip --isolated install --no-index --find-links=/tests/wheels --only-binary=:all: --disable-pip-version-check --no-cache-dir pytest==8.3.5 iniconfig==2.1.0 packaging==25.0 pluggy==1.5.0
stage=behavior_checks
/opt/venv/bin/python -I /tests/ratio_verifier.py --manifest /tests/test-manifest.json --output-dir /logs/verifier
completed=1
