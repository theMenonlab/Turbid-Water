#!/usr/bin/env bash
set -euo pipefail
cd /home/al/Documents/Codex/2026-10-15/task-2
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export TS_SOCKET=/tmp/socket-ts.1000
exec /home/al/miniconda3/envs/hsrmamba/bin/python -B -u /home/al/Documents/NRL/work/experiments/multiframe_paper_sprint_20260906/gpu_guard.py -- /home/al/miniconda3/envs/hsrmamba/bin/python -B -u code/workflow.py --phase full
