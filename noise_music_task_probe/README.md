# noise_music_task_probe

`noise_music_task_embed`에서 생성된 조건별 임베딩(`clean + 4단계 noise`)으로 linear probe를 실행합니다.

기본 설정:
- 데이터: `emo/emotion_regression`
- 모델: `mulan`, `musicgen_melody_large`

실행 예시:

```bash
conda run -n universal python -m ismir2026.noise_music_task_probe.run_noise_linear_probes
```

출력:
- `noise_music_task_probe/probe_results/probe_summary.csv`
- `noise_music_task_probe/probe_results/probe_summary_with_alignment.csv`
- `noise_music_task_probe/probe_results/probe_fold_details.csv`

`probe_summary_with_alignment.csv`에는 clean 대비 성능 변화(`delta_from_clean`, `delta_from_clean_pct`)가 포함됩니다.
