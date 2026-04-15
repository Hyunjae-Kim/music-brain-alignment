# natural_component_align

Layerwise alignment between foundation-model embeddings and the six Norman 2015 response components:

- `lowfreq`
- `highfreq`
- `envsounds`
- `pitch`
- `speech`
- `music`

This package reuses the existing Norman/Tuckute stimulus ordering and precomputed embeddings from `natural_sound_align`, but writes its own outputs under `natural_component_align/results`.

Regression follows Tuckute et al. (2023): raw explained variance (`squared Pearson r`) without noise correction.
Per-component RSA is implemented here as a practical extension using component-derived pairwise absolute-difference RDMs, because Tuckute note that single components do not support their original participant-level RSA procedure.

Example smoke test across all requested models:

```bash
python3 -m natural_component_align.pipeline \
  --models CLAP CLAP_MUSIC CLAP_MUSIC_SPEECH MULAN MERT MERT95 WAV2VEC2 MUSICGEN_SMALL MUSICGEN_MEDIUM MUSICGEN_LARGE MUSICGEN_MELODY_LARGE \
  --skip-final \
  --max-representations 1 \
  --n-splits 2 \
  --n-train 30 \
  --n-test 20 \
  --overwrite
```

Compile cached summaries:

```bash
python3 -m natural_component_align.cache_summary
```
