#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

/home/hjkim/miniconda3/bin/conda run --no-capture-output -n universal \
  python -m music_task_embed.run_mtg_clap \
  --data-root "${ROOT_DIR}/dataset/music_dataset" \
  --mtg-audio-root "${ROOT_DIR}/dataset/music_dataset/raw_30s/audio-low" \
  --manifest-root "${ROOT_DIR}/music_task_embed/manifests" \
  --output-root "${ROOT_DIR}/music_task_embed/embeddings" \
  --model-name clap \
  --task mtg_genre \
  --device cuda \
  --batch-size 8 \
  --include-layers \
  --local-files-only \
  --skip-prepare \
  --log-every 1 \
  > "${LOG_DIR}/mtg_genre_clap.log" 2>&1



### 아래 두줄 복사해서 한번에 실행
### cd /home/hjkim/selective_auditory/ismir2026
### ./music_task_embed/mtg_run_scripts/run_mtg_genre_clap.sh
