# noise_music_task_embed

SNR 단계별 노이즈를 오디오 로드 직후(모델 입력 직전) 주입해서 임베딩을 추출합니다.

기본 설정:
- 대상 모델: `mulan`, `musicgen_melody_large`
- 대상 데이터: `music_task_embed/manifests/emo/emotion_regression`
- 노이즈 레벨: `clean`, `20 dB`, `10 dB`, `0 dB`, `-10 dB`

실행 예시:

```bash
conda run -n universal python -m ismir2026.noise_music_task_embed.run_noise_extraction_job \
  --device cuda \
  --cache-dir /home/hjkim/.cache/huggingface \
  --batch-size 1
```

노이즈 샘플(청취용)만 먼저 만들려면:

```bash
conda run -n universal python -m ismir2026.noise_music_task_embed.run_noise_extraction_job \
  --preview-only
```

출력:
- 조건별 임베딩: `noise_music_task_embed/embeddings/<condition>/...`
- 조건 정의: `noise_music_task_embed/embeddings/noise_conditions.json`
- 청취용 샘플: `noise_music_task_embed/preview_audio/emo/emotion_regression`
