#!/usr/bin/env bash
set -euo pipefail

cd /home/hjkim/selective_auditory
export PYTHONUNBUFFERED=1

GPU_IDS_RAW="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_RAW}"
GPU_COUNT="${#GPU_IDS[@]}"
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  GPU_IDS=("0")
  GPU_COUNT=1
fi
echo "[mtg] using GPUs: ${GPU_IDS[*]}"

EMBED_ROOT="/home/hjkim/selective_auditory/ismir2026/music_task_embed/embeddings"
MANIFEST_ROOT="/home/hjkim/selective_auditory/ismir2026/music_task_embed/manifests"

MODELS=(
  clap
  clap_music
  clap_music_speech
  mulan
  mert
  mert95
  musicgen_small
  musicgen_medium
  musicgen_large
  musicgen_melody_large
)

MANIFEST_DIRS=(
  "${MANIFEST_ROOT}/mtgtop50/music_tagging"
  "${MANIFEST_ROOT}/mtgmoodtheme/emotion_classification"
  "${MANIFEST_ROOT}/mtggenre/genre_classification"
  "${MANIFEST_ROOT}/mtginstrument/instrument_classification"
)

declare -a PIDS=()
declare -a PID_MODELS=()

for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  gpu="${GPU_IDS[$((idx % GPU_COUNT))]}"
  echo "[mtg] start model=${model} gpu=${gpu} at $(date)"
  /home/hjkim/miniconda3/bin/conda run --no-capture-output -n universal \
    python -m ismir2026.music_task_embed.run_extraction_job \
    --manifest-dir "${MANIFEST_DIRS[0]}" \
    --manifest-dir "${MANIFEST_DIRS[1]}" \
    --manifest-dir "${MANIFEST_DIRS[2]}" \
    --manifest-dir "${MANIFEST_DIRS[3]}" \
    --model-name "${model}" \
    --output-root "${EMBED_ROOT}" \
    --device "cuda:${gpu}" \
    --batch-size 8 \
    --include-layers \
    --local-files-only &
  PIDS+=("$!")
  PID_MODELS+=("${model}")

  if (( ${#PIDS[@]} >= GPU_COUNT )); then
    if ! wait -n; then
      echo "[mtg] warning: a job exited with error"
    fi
    new_pids=()
    new_models=()
    for j in "${!PIDS[@]}"; do
      if kill -0 "${PIDS[$j]}" 2>/dev/null; then
        new_pids+=("${PIDS[$j]}")
        new_models+=("${PID_MODELS[$j]}")
      fi
    done
    PIDS=("${new_pids[@]}")
    PID_MODELS=("${new_models[@]}")
  fi
done

for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[mtg] warning: a job exited with error"
  fi
done
echo "[mtg] all models completed at $(date)"
