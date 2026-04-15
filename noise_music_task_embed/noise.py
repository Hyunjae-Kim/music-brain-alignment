from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import numpy as np


DEFAULT_SNR_LEVELS_DB: tuple[float, ...] = (20.0, 10.0, 0.0, -10.0)


@dataclass(frozen=True)
class NoiseCondition:
    condition_id: str
    level_index: int
    snr_db: float | None

    @property
    def is_clean(self) -> bool:
        return self.snr_db is None


def _snr_to_tag(snr_db: float) -> str:
    sign = "p" if snr_db >= 0 else "m"
    magnitude = f"{abs(float(snr_db)):.1f}".replace(".", "p")
    return f"snr_{sign}{magnitude}db"


def parse_snr_levels(value: str | None) -> tuple[float, ...]:
    if value is None or not value.strip():
        return DEFAULT_SNR_LEVELS_DB
    levels: list[float] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        levels.append(float(text))
    if not levels:
        raise ValueError("No valid SNR values were provided.")
    return tuple(levels)


def build_noise_conditions(
    *,
    snr_levels_db: Iterable[float],
    include_clean: bool,
) -> list[NoiseCondition]:
    conditions: list[NoiseCondition] = []
    if include_clean:
        conditions.append(NoiseCondition(condition_id="clean", level_index=0, snr_db=None))
    for index, snr_db in enumerate(snr_levels_db, start=1):
        conditions.append(
            NoiseCondition(
                condition_id=f"noise_l{index}_{_snr_to_tag(float(snr_db))}",
                level_index=index,
                snr_db=float(snr_db),
            )
        )
    if not conditions:
        raise ValueError("At least one condition must be specified.")
    return conditions


def _rng_for_audio(*, seed: int, audio_key: str, condition: NoiseCondition) -> np.random.Generator:
    token = f"{int(seed)}|{audio_key}|{condition.condition_id}|{condition.snr_db}"
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    child_seed = int.from_bytes(digest, byteorder="little", signed=False)
    return np.random.default_rng(child_seed)


def mix_white_noise_at_snr(
    clean_waveform: np.ndarray,
    *,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    clean = np.asarray(clean_waveform, dtype=np.float32)
    if clean.ndim != 1:
        raise ValueError(f"Expected mono waveform, got shape={tuple(clean.shape)}")

    signal_power = float(np.mean(np.square(clean, dtype=np.float64)))
    if not np.isfinite(signal_power) or signal_power <= 1e-12:
        return clean.copy()

    unit_noise = rng.standard_normal(clean.shape[0], dtype=np.float32)
    unit_noise_power = float(np.mean(np.square(unit_noise, dtype=np.float64)))
    if not np.isfinite(unit_noise_power) or unit_noise_power <= 0.0:
        return clean.copy()

    target_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    scale = math.sqrt(max(target_noise_power, 0.0) / unit_noise_power)
    return (clean + unit_noise * np.float32(scale)).astype(np.float32, copy=False)


def apply_noise_condition(
    clean_waveform: np.ndarray,
    *,
    audio_key: str,
    condition: NoiseCondition,
    seed: int,
) -> np.ndarray:
    if condition.is_clean:
        return np.asarray(clean_waveform, dtype=np.float32).copy()
    rng = _rng_for_audio(seed=seed, audio_key=audio_key, condition=condition)
    return mix_white_noise_at_snr(
        clean_waveform,
        snr_db=float(condition.snr_db),
        rng=rng,
    )


def measured_snr_db(clean_waveform: np.ndarray, noisy_waveform: np.ndarray) -> float:
    clean = np.asarray(clean_waveform, dtype=np.float64)
    noisy = np.asarray(noisy_waveform, dtype=np.float64)
    if clean.shape != noisy.shape:
        raise ValueError("clean_waveform and noisy_waveform must share the same shape")
    noise = noisy - clean
    signal_power = float(np.mean(np.square(clean)))
    noise_power = float(np.mean(np.square(noise)))
    if signal_power <= 0.0 or noise_power <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))
