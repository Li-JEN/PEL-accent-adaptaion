#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./run.sh --stage 6
./run_pair_mmd.sh --stage 6
./run_pair_swd.sh --stage 6
