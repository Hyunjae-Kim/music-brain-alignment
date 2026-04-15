from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ismir2026.music_task_embed import encoders as base_encoders

from .noise import NoiseCondition, apply_noise_condition


SUPPORTED_MODEL_NAMES = base_encoders.SUPPORTED_MODEL_NAMES


@dataclass(frozen=True)
class NoiseAugmentConfig:
    condition: NoiseCondition
    seed: int


class NoiseWrappedExtractor:
    def __init__(self, base_extractor: Any, *, noise_config: NoiseAugmentConfig) -> None:
        self._base_extractor = base_extractor
        self.noise_config = noise_config
        self.model_name = getattr(base_extractor, "model_name", "unknown")

    @contextmanager
    def _patched_loader(self) -> Iterator[None]:
        original_loader = base_encoders.load_waveform

        def noisy_loader(path: str | Path, target_sample_rate: int):
            clean = original_loader(path, target_sample_rate=target_sample_rate)
            audio_key = f"{Path(path).resolve()}|sr={int(target_sample_rate)}"
            return apply_noise_condition(
                clean,
                audio_key=audio_key,
                condition=self.noise_config.condition,
                seed=self.noise_config.seed,
            )

        base_encoders.load_waveform = noisy_loader
        try:
            yield
        finally:
            base_encoders.load_waveform = original_loader

    def extract(self, audio_paths: list[str], batch_size: int, include_layers: bool = False) -> base_encoders.ExtractorResult:
        with self._patched_loader():
            result = self._base_extractor.extract(
                audio_paths,
                batch_size=batch_size,
                include_layers=include_layers,
            )

        metadata = dict(result.metadata)
        metadata.update(
            {
                "noise_condition": self.noise_config.condition.condition_id,
                "noise_level_index": int(self.noise_config.condition.level_index),
                "noise_snr_db": self.noise_config.condition.snr_db,
                "noise_seed": int(self.noise_config.seed),
            }
        )
        return base_encoders.ExtractorResult(
            final_name=result.final_name,
            final_matrix=result.final_matrix,
            layer_matrices=result.layer_matrices,
            metadata=metadata,
        )

    def representation_names(self, include_layers: bool) -> list[str]:
        if hasattr(self._base_extractor, "representation_names"):
            return self._base_extractor.representation_names(include_layers)
        return ["final"]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_extractor, name)


def get_base_extractor(
    model_name: str,
    *,
    device: str = "auto",
    cache_dir: str | None = None,
    local_files_only: bool = False,
    clap_variant: str = "audio_embedding",
):
    return base_encoders.get_extractor(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        clap_variant=clap_variant,
    )


def wrap_extractor_with_noise(base_extractor: Any, *, condition: NoiseCondition, seed: int) -> NoiseWrappedExtractor:
    return NoiseWrappedExtractor(
        base_extractor,
        noise_config=NoiseAugmentConfig(condition=condition, seed=int(seed)),
    )
