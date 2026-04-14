# ISMIR2026 Workspace

This directory now separates the work into three clearer packages:

- [natural_sound_align](/home/hjkim/selective_auditory/ismir2026/natural_sound_align): natural-sound model vs human fMRI alignment
- [music_task_embed](/home/hjkim/selective_auditory/ismir2026/music_task_embed): `music_dataset` task manifests and audio embedding extraction
- [music_task_probe](/home/hjkim/selective_auditory/ismir2026/music_task_probe): downstream linear probing over the extracted music-task embeddings

## Directory layout

### `natural_sound_align`

Use this package for the Tuckute-style natural-sound alignment analyses against `NH2015` / `B2021`.

Important locations:

- Code: [natural_sound_align](/home/hjkim/selective_auditory/ismir2026/natural_sound_align)
- Cached analysis artifacts: [embeddings](/home/hjkim/selective_auditory/ismir2026/natural_sound_align/embeddings)
- Compiled summaries: [compiled_results](/home/hjkim/selective_auditory/ismir2026/natural_sound_align/compiled_results)
- Notebooks:
  - [tuckute_foundation_analysis.ipynb](/home/hjkim/selective_auditory/ismir2026/natural_sound_align/tuckute_foundation_analysis.ipynb)
  - [summary_plot.ipynb](/home/hjkim/selective_auditory/ismir2026/natural_sound_align/summary_plot.ipynb)

Naming change:

- old: `foundation_embed`
- new: `natural_sound_align`

Artifact root change:

- old: `natural_sound_align/outputs`
- new: `natural_sound_align/embeddings`

Typical commands:

```bash
conda run -n universal python -m ismir2026.natural_sound_align.cache_summary
conda run -n universal python -m ismir2026.natural_sound_align.pipeline --dataset NH2015
```

### `music_task_embed`

Use this package for `music_dataset` downstream-task manifests and embedding extraction.

Important locations:

- Code: [music_task_embed](/home/hjkim/selective_auditory/ismir2026/music_task_embed)
- Raw task data: [music_dataset](/home/hjkim/selective_auditory/ismir2026/dataset/music_dataset)
- Task manifests: [manifests](/home/hjkim/selective_auditory/ismir2026/music_task_embed/manifests)
- Extracted embeddings: [embeddings](/home/hjkim/selective_auditory/ismir2026/music_task_embed/embeddings)
- Model summary table: [embedding_model_summary.csv](/home/hjkim/selective_auditory/ismir2026/music_task_embed/embedding_model_summary.csv)
- Embedding summary notebook: [embedding_extraction_summary.ipynb](/home/hjkim/selective_auditory/ismir2026/music_task_embed/notebooks/embedding_extraction_summary.ipynb)

Naming change:

- old: `music_dt`
- new: `music_task_embed`

Currently extracted models include:

- `clap`
- `clap_music`
- `clap_music_speech`
- `mulan`
- `mert`
- `mert95`
- `wav2vec2`
- `musicgen_small`
- `musicgen_medium`
- `musicgen_large`
- `musicgen_melody_large`

Typical command:

```bash
conda run -n universal python -m ismir2026.music_task_embed.run_extraction_job \
  --output-root /home/hjkim/selective_auditory/ismir2026/music_task_embed/embeddings \
  --device cuda \
  --cache-dir /home/hjkim/.cache/huggingface \
  --local-files-only \
  --include-layers \
  --batch-size 1 \
  --model-name musicgen_large \
  --manifest-dir /home/hjkim/selective_auditory/ismir2026/music_task_embed/manifests/emo/emotion_regression
```

Resume behavior:

- extraction is resume-safe
- rerunning the same job skips already completed partitions

### `music_task_probe`

Use this package for downstream probing over the extracted `music_task_embed` embeddings.

Important locations:

- Code: [music_task_probe](/home/hjkim/selective_auditory/ismir2026/music_task_probe)
- Canonical probe outputs (all models + layerwise/final summaries): [probe_results_all_models](/home/hjkim/selective_auditory/ismir2026/music_task_probe/probe_results_all_models)
- Notebooks:
  - [probe_results_overview.ipynb](/home/hjkim/selective_auditory/ismir2026/music_task_probe/notebooks/probe_results_overview.ipynb)
  - [results_aggregated.ipynb](/home/hjkim/selective_auditory/ismir2026/music_task_probe/notebooks/results_aggregated.ipynb)
  - [layer_analysis.ipynb](/home/hjkim/selective_auditory/ismir2026/music_task_probe/notebooks/layer_analysis.ipynb)

This package reads:

- task embeddings from [music_task_embed/embeddings](/home/hjkim/selective_auditory/ismir2026/music_task_embed/embeddings)
- alignment summaries from [natural_sound_align/embeddings](/home/hjkim/selective_auditory/ismir2026/natural_sound_align/embeddings)

Typical commands:

```bash
conda run -n universal python -m ismir2026.music_task_probe.run_linear_probes
conda run -n universal python -m ismir2026.music_task_probe.linear_probe --representation-mode final_only
```

## Notes on the rename

The following path changes have been applied through the code, notebooks, and manifest metadata:

- `ismir2026.foundation_embed` -> `ismir2026.natural_sound_align`
- `ismir2026.music_dt` -> `ismir2026.music_task_embed`
- probe-related code and result folders moved from `music_dt` into `music_task_probe`
- natural-sound alignment cache root renamed from `outputs` to `embeddings`

If you reopen old notebooks or scripts, use the new package names and roots above.
