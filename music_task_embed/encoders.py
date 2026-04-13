from __future__ import annotations

from dataclasses import dataclass
import os
import time
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoProcessor,
    ClapModel,
    MusicgenForConditionalGeneration,
    MusicgenMelodyForConditionalGeneration,
    Wav2Vec2FeatureExtractor,
)

from muq import MuQMuLan


DEFAULT_CLAP_MODEL_ID = "laion/clap-htsat-unfused"
DEFAULT_CLAP_MUSIC_MODEL_ID = "laion/larger_clap_music"
DEFAULT_CLAP_MUSIC_SPEECH_MODEL_ID = "laion/larger_clap_music_and_speech"
DEFAULT_MULAN_MODEL_ID = "OpenMuQ/MuQ-MuLan-large"
DEFAULT_MERT_MODEL_ID = "m-a-p/MERT-v1-330M"
DEFAULT_MERT95_MODEL_ID = "m-a-p/MERT-v1-95M"
DEFAULT_MUSICGEN_SMALL_MODEL_ID = "facebook/musicgen-small"
DEFAULT_MUSICGEN_MEDIUM_MODEL_ID = "facebook/musicgen-medium"
DEFAULT_MUSICGEN_LARGE_MODEL_ID = "facebook/musicgen-large"
DEFAULT_MUSICGEN_MELODY_LARGE_MODEL_ID = "facebook/musicgen-melody-large"

SUPPORTED_MODEL_NAMES = (
    "clap",
    "clap_music",
    "clap_music_speech",
    "mulan",
    "mert",
    "mert95",
    "musicgen_small",
    "musicgen_medium",
    "musicgen_large",
    "musicgen_melody_large",
)

_LOG_EVERY = int(os.getenv("EMBED_LOG_EVERY", "0"))


def _format_gpu_usage(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(index) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(index) / (1024 ** 3)
        return f" gpu={index} mem={allocated:.2f}G/{reserved:.2f}G"
    except Exception:
        return " gpu=unknown"


def _maybe_log_progress(
    *,
    model_name: str,
    device: torch.device,
    batch_index: int,
    total_batches: int,
    total_items: int,
    batch_size: int,
    start_time: float,
) -> None:
    if _LOG_EVERY <= 0:
        return
    if batch_index % _LOG_EVERY != 0 and batch_index != total_batches:
        return
    elapsed = max(time.time() - start_time, 1e-6)
    done = min(batch_index * batch_size, total_items)
    rate = done / elapsed
    gpu_text = _format_gpu_usage(device)
    print(
        f"[embed] {model_name} {done}/{total_items} "
        f"(batch {batch_index}/{total_batches}) "
        f"{rate:.2f} items/s{gpu_text}",
        flush=True,
    )


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_waveform(path: str | Path, target_sample_rate: int) -> np.ndarray:
    waveform, sample_rate = sf.read(str(path), dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32, copy=False)
    if sample_rate != target_sample_rate:
        gcd = np.gcd(sample_rate, target_sample_rate)
        waveform = resample_poly(
            waveform,
            target_sample_rate // gcd,
            sample_rate // gcd,
        ).astype(np.float32, copy=False)
    return waveform


@dataclass(frozen=True)
class ExtractorResult:
    final_name: str
    final_matrix: np.ndarray
    layer_matrices: dict[str, np.ndarray]
    metadata: dict[str, str | int | float | None]


class ClapExtractor:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_CLAP_MODEL_ID,
        device: torch.device,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        variant: str = "audio_embedding",
        final_representation_name: str | None = None,
    ) -> None:
        if variant not in {"audio_embedding", "encoder"}:
            raise ValueError("CLAP variant must be 'audio_embedding' or 'encoder'.")
        self.model_name = "clap"
        self.variant = variant
        self.final_representation_name = final_representation_name or variant
        self.device = device
        cache_path = Path(cache_dir) if cache_dir is not None else None
        local_snapshot = self._resolve_local_snapshot(model_id, cache_path)
        load_source = local_snapshot if local_snapshot is not None else model_id
        use_safetensors = not (
            local_snapshot is not None and (local_snapshot / "model.safetensors").exists() is False
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            load_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.model = ClapModel.from_pretrained(
            load_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_safetensors=use_safetensors,
        ).to(device)
        self.model.eval()
        self.sample_rate = int(self.feature_extractor.sampling_rate)
        self.model_id = model_id
        self.hidden_state_count = int(getattr(self.model.audio_model.config, "num_hidden_layers", 0)) + 1

    @staticmethod
    def _candidate_hf_roots(cache_dir: Path | None) -> list[Path]:
        roots: list[Path] = []
        if cache_dir is not None:
            roots.append(cache_dir)
        roots.extend(
            [
                Path.home() / "auto_project" / ".hf_cache",
                Path.home() / ".cache" / "huggingface",
            ]
        )
        unique: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            if root not in seen:
                unique.append(root)
                seen.add(root)
        return unique

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    def _resolve_local_snapshot(self, model_id: str, cache_dir: Path | None) -> Path | None:
        model_dirname = f"models--{model_id.replace('/', '--')}"
        snapshot_rows: list[dict[str, Any]] = []
        for hf_root in self._candidate_hf_roots(cache_dir):
            snapshots_root = hf_root / "hub" / model_dirname / "snapshots"
            if not snapshots_root.exists():
                continue
            for snapshot in sorted(path for path in snapshots_root.iterdir() if path.is_dir()):
                row = {
                    "snapshot": snapshot,
                    "config": snapshot / "config.json",
                    "preprocessor": snapshot / "preprocessor_config.json",
                    "safetensors": snapshot / "model.safetensors",
                    "bin": snapshot / "pytorch_model.bin",
                }
                snapshot_rows.append(row)

        for row in snapshot_rows:
            if row["config"].exists() and row["preprocessor"].exists() and row["safetensors"].exists():
                return row["snapshot"]

        config_path = next((row["config"] for row in snapshot_rows if row["config"].exists()), None)
        preprocessor_path = next((row["preprocessor"] for row in snapshot_rows if row["preprocessor"].exists()), None)
        safetensors_path = next((row["safetensors"] for row in snapshot_rows if row["safetensors"].exists()), None)
        if config_path is not None and preprocessor_path is not None and safetensors_path is not None:
            assembled_root = Path("/tmp") / f"clap_local_{model_id.replace('/', '__')}"
            assembled_root.mkdir(parents=True, exist_ok=True)
            self._link_or_copy(config_path, assembled_root / "config.json")
            self._link_or_copy(preprocessor_path, assembled_root / "preprocessor_config.json")
            self._link_or_copy(safetensors_path, assembled_root / "model.safetensors")
            optional_names = (
                "merges.txt",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.json",
            )
            for name in optional_names:
                optional_path = next((row["snapshot"] / name for row in snapshot_rows if (row["snapshot"] / name).exists()), None)
                if optional_path is not None:
                    self._link_or_copy(optional_path, assembled_root / name)
            if (
                (assembled_root / "config.json").exists()
                and (assembled_root / "preprocessor_config.json").exists()
                and (assembled_root / "model.safetensors").exists()
            ):
                return assembled_root

        for row in snapshot_rows:
            if row["config"].exists() and row["preprocessor"].exists() and row["bin"].exists():
                return row["snapshot"]
        return None

    @staticmethod
    def _pool_hidden(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 4:
            return hidden.mean(dim=(-1, -2))
        if hidden.ndim == 3:
            return hidden.mean(dim=1)
        if hidden.ndim == 2:
            return hidden
        raise ValueError(f"Unexpected CLAP hidden-state shape: {tuple(hidden.shape)}")

    def extract(self, audio_paths: list[str], batch_size: int, include_layers: bool = False) -> ExtractorResult:
        final_rows: list[np.ndarray] = []
        layer_rows: dict[str, list[np.ndarray]] = {}
        total_items = len(audio_paths)
        total_batches = (total_items + batch_size - 1) // batch_size
        start_time = time.time()
        if _LOG_EVERY > 0:
            print(
                f"[embed] start {self.model_name} device={self.device} "
                f"items={total_items} batch={batch_size} include_layers={include_layers}",
                flush=True,
            )

        with torch.inference_mode():
            for batch_index, start in enumerate(range(0, len(audio_paths), batch_size), start=1):
                batch_paths = audio_paths[start : start + batch_size]
                waveforms = [load_waveform(path, self.sample_rate) for path in batch_paths]
                inputs = self.feature_extractor(
                    raw_speech=waveforms,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model.audio_model(
                    input_features=inputs["input_features"],
                    is_longer=inputs.get("is_longer"),
                    output_hidden_states=include_layers,
                    return_dict=True,
                )
                encoder_repr = outputs.pooler_output
                audio_embeddings = self.model.audio_projection(encoder_repr)
                audio_embeddings = audio_embeddings / audio_embeddings.norm(
                    p=2,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)

                chosen = audio_embeddings if self.variant == "audio_embedding" else encoder_repr
                final_rows.append(chosen.detach().cpu().numpy().astype(np.float32, copy=False))

                if include_layers and getattr(outputs, "hidden_states", None) is not None:
                    for layer_index, hidden in enumerate(outputs.hidden_states):
                        pooled = self._pool_hidden(hidden)
                        layer_rows.setdefault(f"layer_{layer_index:02d}", []).append(
                            pooled.detach().cpu().numpy().astype(np.float32, copy=False)
                        )
                _maybe_log_progress(
                    model_name=self.model_name,
                    device=self.device,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    total_items=total_items,
                    batch_size=batch_size,
                    start_time=start_time,
                )

        final_matrix = np.concatenate(final_rows, axis=0)
        return ExtractorResult(
            final_name=self.final_representation_name,
            final_matrix=final_matrix,
            layer_matrices={
                layer_name: np.concatenate(parts, axis=0) for layer_name, parts in layer_rows.items()
            },
            metadata={
                "model_id": self.model_id,
                "sample_rate": self.sample_rate,
                "variant": self.variant,
            },
        )

    def representation_names(self, include_layers: bool) -> list[str]:
        names = [self.final_representation_name]
        if include_layers:
            names.extend(f"layer_{layer_index:02d}" for layer_index in range(self.hidden_state_count))
        return names


class MulanExtractor:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MULAN_MODEL_ID,
        device: torch.device,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = "mulan"
        self.model_id = model_id
        self.device = device
        self.model = MuQMuLan.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = 24_000
        self.hidden_state_count = 13

    @staticmethod
    def _pool_hidden(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 3:
            return hidden.mean(dim=1)
        if hidden.ndim == 2:
            return hidden
        raise ValueError(f"Unexpected MuLAN hidden-state shape: {tuple(hidden.shape)}")

    def _extract_backbone_hidden_states(self, wav: torch.Tensor) -> list[torch.Tensor]:
        audio_module = self.model.mulan_module.audio
        hidden_states = audio_module._forward_pretrained_model(wav)

        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.ndim == 4:
                return [hidden_states[index] for index in range(hidden_states.shape[0])]
            if hidden_states.ndim == 3:
                return [hidden_states]
            raise ValueError(f"Unexpected MuLAN hidden-state tensor shape: {tuple(hidden_states.shape)}")

        if isinstance(hidden_states, (tuple, list)):
            tensors = [hidden for hidden in hidden_states if isinstance(hidden, torch.Tensor)]
            if tensors:
                return tensors

        raise ValueError("MuLAN backbone hidden states could not be resolved.")

    def extract(self, audio_paths: list[str], batch_size: int, include_layers: bool = False) -> ExtractorResult:
        embeddings: list[np.ndarray] = []
        layer_rows: dict[str, list[np.ndarray]] = {}
        use_parallel = self.device.type == "cuda"
        batch_size = max(int(batch_size), 1)
        total_items = len(audio_paths)
        total_batches = (total_items + batch_size - 1) // batch_size
        start_time = time.time()
        if _LOG_EVERY > 0:
            print(
                f"[embed] start {self.model_name} device={self.device} "
                f"items={total_items} batch={batch_size} include_layers={include_layers}",
                flush=True,
            )

        for batch_index, start in enumerate(range(0, len(audio_paths), batch_size), start=1):
            batch_paths = audio_paths[start : start + batch_size]

            if include_layers:
                for path in batch_paths:
                    audio = load_waveform(path, target_sample_rate=self.sample_rate)
                    wav = torch.from_numpy(audio).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        embedding = self.model(wavs=wav, parallel_processing=use_parallel)
                        hidden_states = self._extract_backbone_hidden_states(wav)

                    embeddings.append(embedding.detach().cpu().numpy()[0].astype(np.float32, copy=False))
                    for layer_index, hidden in enumerate(hidden_states):
                        pooled = self._pool_hidden(hidden)
                        layer_rows.setdefault(f"layer_{layer_index:02d}", []).append(
                            pooled.detach().cpu().numpy()[0].astype(np.float32, copy=False)
                        )
                _maybe_log_progress(
                    model_name=self.model_name,
                    device=self.device,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    total_items=total_items,
                    batch_size=batch_size,
                    start_time=start_time,
                )
                continue

            clip_batches: list[torch.Tensor] = []
            clip_counts: list[int] = []
            for path in batch_paths:
                audio = load_waveform(path, target_sample_rate=self.sample_rate)
                clip_tensor = self.model._get_all_clips(torch.from_numpy(audio))
                clip_batches.append(clip_tensor)
                clip_counts.append(int(clip_tensor.shape[0]))

            flat_clips = torch.cat(clip_batches, dim=0).to(self.device)
            with torch.no_grad():
                clip_embeddings = self.model.mulan_module.get_audio_latents(flat_clips)

            offset = 0
            for clip_count in clip_counts:
                pooled = clip_embeddings[offset : offset + clip_count].mean(dim=0)
                embeddings.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
                offset += clip_count
            _maybe_log_progress(
                model_name=self.model_name,
                device=self.device,
                batch_index=batch_index,
                total_batches=total_batches,
                total_items=total_items,
                batch_size=batch_size,
                start_time=start_time,
            )

        return ExtractorResult(
            final_name="final",
            final_matrix=np.stack(embeddings, axis=0),
            layer_matrices={
                layer_name: np.stack(parts, axis=0) for layer_name, parts in layer_rows.items()
            },
            metadata={"model_id": self.model_id, "sample_rate": self.sample_rate},
        )

    def representation_names(self, include_layers: bool) -> list[str]:
        names = ["final"]
        if include_layers:
            names.extend(f"layer_{layer_index:02d}" for layer_index in range(self.hidden_state_count))
        return names


class MertExtractor:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MERT_MODEL_ID,
        device: torch.device,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        final_representation_name: str = "final",
    ) -> None:
        self.model_name = "mert"
        self.model_id = model_id
        self.device = device
        self.final_representation_name = final_representation_name
        cache_path = Path(cache_dir) if cache_dir is not None else None
        os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf_modules_cache")
        local_snapshot = self._resolve_local_snapshot(model_id, cache_path)
        if local_snapshot is not None:
            self.processor, self.model = self._load_local_snapshot(local_snapshot)
        else:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            self.model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = int(self.processor.sampling_rate)
        self.hidden_state_count = int(getattr(self.model.config, "num_hidden_layers", 0)) + 1

    @staticmethod
    def _candidate_hf_roots(cache_dir: Path | None) -> list[Path]:
        roots: list[Path] = []
        if cache_dir is not None:
            roots.append(cache_dir)
        roots.extend(
            [
                Path.home() / "auto_project" / ".hf_cache",
                Path.home() / ".cache" / "huggingface",
            ]
        )
        unique: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            if root not in seen:
                unique.append(root)
                seen.add(root)
        return unique

    def _resolve_local_snapshot(self, model_id: str, cache_dir: Path | None) -> Path | None:
        model_dirname = f"models--{model_id.replace('/', '--')}"
        for hf_root in self._candidate_hf_roots(cache_dir):
            snapshots_root = hf_root / "hub" / model_dirname / "snapshots"
            if not snapshots_root.exists():
                continue
            for snapshot in sorted(path for path in snapshots_root.iterdir() if path.is_dir()):
                if (
                    (snapshot / "config.json").exists()
                    and (snapshot / "preprocessor_config.json").exists()
                    and (snapshot / "pytorch_model.bin").exists()
                ):
                    return snapshot
        return None

    @staticmethod
    def _load_local_snapshot(snapshot: Path) -> tuple[Wav2Vec2FeatureExtractor, torch.nn.Module]:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            str(snapshot),
            trust_remote_code=True,
            local_files_only=True,
        )
        config = AutoConfig.from_pretrained(
            str(snapshot),
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModel.from_config(config, trust_remote_code=True)
        state_dict = torch.load(str(snapshot / "pytorch_model.bin"), map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Unexpected MERT load mismatch: missing={len(missing)} unexpected={len(unexpected)}"
            )
        return processor, model

    def _pool_hidden(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return hidden.mean(dim=1)
        attention_mask = attention_mask.to(hidden.device)
        if hasattr(self.model, "_get_feature_vector_attention_mask"):
            mask = self.model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        else:
            mask = attention_mask
        mask = mask.to(hidden.device).unsqueeze(-1).type_as(hidden)
        masked = hidden * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return masked.sum(dim=1) / denom

    def extract(self, audio_paths: list[str], batch_size: int, include_layers: bool = False) -> ExtractorResult:
        embeddings: list[np.ndarray] = []
        layer_rows: dict[str, list[np.ndarray]] = {}
        total_items = len(audio_paths)
        total_batches = (total_items + batch_size - 1) // batch_size
        start_time = time.time()
        if _LOG_EVERY > 0:
            print(
                f"[embed] start {self.model_name} device={self.device} "
                f"items={total_items} batch={batch_size} include_layers={include_layers}",
                flush=True,
            )

        for batch_index, start in enumerate(range(0, len(audio_paths), batch_size), start=1):
            batch_paths = audio_paths[start : start + batch_size]
            audio_batch = [load_waveform(path, target_sample_rate=self.sample_rate) for path in batch_paths]
            inputs = self.processor(
                audio_batch,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            attention_mask = inputs.get("attention_mask")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True, output_hidden_states=include_layers)

            pooled_last = self._pool_hidden(outputs.last_hidden_state, attention_mask)
            embeddings.append(pooled_last.detach().cpu().numpy().astype(np.float32, copy=False))

            if include_layers:
                for layer_index, hidden in enumerate(outputs.hidden_states or ()):
                    pooled = self._pool_hidden(hidden, attention_mask)
                    layer_rows.setdefault(f"layer_{layer_index:02d}", []).append(
                        pooled.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
            _maybe_log_progress(
                model_name=self.model_name,
                device=self.device,
                batch_index=batch_index,
                total_batches=total_batches,
                total_items=total_items,
                batch_size=batch_size,
                start_time=start_time,
            )

        return ExtractorResult(
            final_name=self.final_representation_name,
            final_matrix=np.concatenate(embeddings, axis=0),
            layer_matrices={
                layer_name: np.concatenate(parts, axis=0) for layer_name, parts in layer_rows.items()
            },
            metadata={"model_id": self.model_id, "sample_rate": self.sample_rate},
        )

    def representation_names(self, include_layers: bool) -> list[str]:
        names = [self.final_representation_name]
        if include_layers:
            names.extend(f"layer_{layer_index:02d}" for layer_index in range(self.hidden_state_count))
        return names


def pool_hidden_state(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim == 3:
        return hidden.mean(dim=1)
    if hidden.ndim == 2:
        return hidden
    raise ValueError(f"Unexpected MusicGen hidden-state shape: {tuple(hidden.shape)}")


def build_unconditional_text_inputs(
    processor: AutoProcessor,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    text_inputs = processor.tokenizer([" "] * batch_size, padding=True, return_tensors="pt")
    return {key: value.to(device) for key, value in text_inputs.items()}


def build_audio_prompt_inputs(
    waveforms: list[np.ndarray],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(waveform) for waveform in waveforms)
    batch_size = len(waveforms)
    input_values = torch.zeros((batch_size, 1, max_length), dtype=dtype)
    padding_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for row_index, waveform in enumerate(waveforms):
        length = len(waveform)
        input_values[row_index, 0, :length] = torch.from_numpy(waveform).to(dtype=dtype)
        padding_mask[row_index, :length] = True
    return input_values.to(device), padding_mask.to(device)


class MusicGenExtractor:
    def __init__(
        self,
        *,
        model_id: str,
        device: torch.device,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = "musicgen"
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        cache_path = Path(cache_dir) if cache_dir is not None else None
        local_snapshot = self._resolve_local_snapshot(model_id, cache_path)
        load_source = str(local_snapshot) if local_snapshot is not None else model_id
        use_safetensors = True
        if local_snapshot is not None:
            has_safetensors = (
                (local_snapshot / "model.safetensors").exists()
                or (local_snapshot / "model.safetensors.index.json").exists()
            )
            use_safetensors = has_safetensors

        config = AutoConfig.from_pretrained(
            load_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.processor = AutoProcessor.from_pretrained(
            load_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        model_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
            "use_safetensors": use_safetensors,
        }
        if device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        if config.model_type == "musicgen_melody":
            self.model = MusicgenMelodyForConditionalGeneration.from_pretrained(
                load_source,
                **model_kwargs,
            )
            self.is_melody = True
        elif config.model_type == "musicgen":
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                load_source,
                **model_kwargs,
            )
            self.is_melody = False
        else:
            raise ValueError(f"Unsupported MusicGen config type: {config.model_type!r}")

        self.model.to(device)
        self.model.eval()

        if hasattr(self.model.audio_encoder, "config") and hasattr(self.model.audio_encoder.config, "chunk_length_s"):
            self.model.audio_encoder.config.chunk_length_s = None
        self.sample_rate = int(self.model.audio_encoder.config.sampling_rate)
        self.hidden_state_count = int(getattr(self.model.decoder.config, "num_hidden_layers", 0)) + 1

    @staticmethod
    def _candidate_hf_roots(cache_dir: Path | None) -> list[Path]:
        roots: list[Path] = []
        if cache_dir is not None:
            roots.append(cache_dir)
        roots.extend(
            [
                Path.home() / "auto_project" / ".hf_cache",
                Path.home() / ".cache" / "huggingface",
            ]
        )
        unique: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            if root not in seen:
                unique.append(root)
                seen.add(root)
        return unique

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    def _resolve_local_snapshot(self, model_id: str, cache_dir: Path | None) -> Path | None:
        model_dirname = f"models--{model_id.replace('/', '--')}"
        snapshot_rows: list[dict[str, Path]] = []
        for hf_root in self._candidate_hf_roots(cache_dir):
            snapshots_root = hf_root / "hub" / model_dirname / "snapshots"
            if not snapshots_root.exists():
                continue
            for snapshot in sorted(path for path in snapshots_root.iterdir() if path.is_dir()):
                snapshot_rows.append(
                    {
                        "snapshot": snapshot,
                        "config": snapshot / "config.json",
                        "generation_config": snapshot / "generation_config.json",
                        "preprocessor": snapshot / "preprocessor_config.json",
                        "special_tokens_map": snapshot / "special_tokens_map.json",
                        "spiece_model": snapshot / "spiece.model",
                        "tokenizer_json": snapshot / "tokenizer.json",
                        "tokenizer_config": snapshot / "tokenizer_config.json",
                        "safetensors": snapshot / "model.safetensors",
                        "safetensors_index": snapshot / "model.safetensors.index.json",
                        "bin": snapshot / "pytorch_model.bin",
                    }
                )

        for row in snapshot_rows:
            if (
                row["config"].exists()
                and row["preprocessor"].exists()
                and (row["safetensors"].exists() or row["safetensors_index"].exists())
            ):
                return row["snapshot"]

        config_path = next((row["config"] for row in snapshot_rows if row["config"].exists()), None)
        preprocessor_path = next((row["preprocessor"] for row in snapshot_rows if row["preprocessor"].exists()), None)
        if config_path is None or preprocessor_path is None:
            return None

        assembled_root = Path("/tmp") / f"musicgen_local_{model_id.replace('/', '__')}"
        assembled_root.mkdir(parents=True, exist_ok=True)
        self._link_or_copy(config_path, assembled_root / "config.json")
        self._link_or_copy(preprocessor_path, assembled_root / "preprocessor_config.json")

        optional_names = (
            ("generation_config", "generation_config.json"),
            ("special_tokens_map", "special_tokens_map.json"),
            ("spiece_model", "spiece.model"),
            ("tokenizer_json", "tokenizer.json"),
            ("tokenizer_config", "tokenizer_config.json"),
        )
        for key, filename in optional_names:
            optional_path = next((row[key] for row in snapshot_rows if row[key].exists()), None)
            if optional_path is not None:
                self._link_or_copy(optional_path, assembled_root / filename)

        safetensors_path = next((row["safetensors"] for row in snapshot_rows if row["safetensors"].exists()), None)
        if safetensors_path is not None:
            self._link_or_copy(safetensors_path, assembled_root / "model.safetensors")
            return assembled_root

        index_row = next((row for row in snapshot_rows if row["safetensors_index"].exists()), None)
        if index_row is not None:
            self._link_or_copy(index_row["safetensors_index"], assembled_root / "model.safetensors.index.json")
            for shard_path in sorted(index_row["snapshot"].glob("*.safetensors")):
                self._link_or_copy(shard_path, assembled_root / shard_path.name)
            return assembled_root

        bin_path = next((row["bin"] for row in snapshot_rows if row["bin"].exists()), None)
        if bin_path is not None:
            self._link_or_copy(bin_path, assembled_root / "pytorch_model.bin")
            return assembled_root

        for row in snapshot_rows:
            if row["config"].exists() and row["preprocessor"].exists() and row["bin"].exists():
                return row["snapshot"]
        return None

    def extract_decoder_hidden_states(self, waveforms: list[np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        input_dtype = self.model.dtype if self.device.type == "cuda" else torch.float32
        input_values, padding_mask = build_audio_prompt_inputs(
            waveforms,
            device=self.device,
            dtype=input_dtype,
        )
        text_inputs = build_unconditional_text_inputs(self.processor, len(waveforms), self.device)

        with torch.inference_mode():
            audio_encoder_outputs = self.model.audio_encoder.encode(
                input_values=input_values,
                padding_mask=padding_mask,
                return_dict=True,
            )
            audio_codes = audio_encoder_outputs.audio_codes
            frames, batch_size, codebooks, seq_len = audio_codes.shape
            if frames != 1:
                raise ValueError(
                    f"Expected a single EnCodec frame, got {frames}. "
                    "MusicGen prompt chunking should be disabled for latent extraction."
                )

            if self.model.config.decoder.audio_channels == 2 and codebooks == self.model.decoder.num_codebooks // 2:
                audio_codes = audio_codes.repeat_interleave(2, dim=2)

            decoder_input_ids = audio_codes[0].reshape(batch_size * self.model.decoder.num_codebooks, seq_len)

            model_inputs: dict[str, torch.Tensor] = {
                "decoder_input_ids": decoder_input_ids,
                "output_hidden_states": True,
                "return_dict": True,
                "use_cache": False,
                **text_inputs,
            }

            if self.is_melody:
                feature_inputs = self.processor.feature_extractor(
                    audio=waveforms,
                    sampling_rate=self.sample_rate,
                    padding=True,
                    return_tensors="pt",
                )
                model_inputs["input_features"] = feature_inputs["input_features"].to(
                    device=self.device,
                    dtype=input_dtype,
                )

            outputs = self.model(**model_inputs)
            hidden_states = getattr(outputs, "decoder_hidden_states", None)
            if hidden_states is None:
                hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise AttributeError("MusicGen outputs did not expose decoder hidden states.")

        final_matrix = pool_hidden_state(hidden_states[-1]).detach().cpu().numpy().astype(np.float32, copy=False)
        layer_matrices = {
            f"layer_{int(layer_index):02d}": pool_hidden_state(hidden).detach().cpu().numpy().astype(np.float32, copy=False)
            for layer_index, hidden in enumerate(hidden_states)
        }
        return final_matrix, layer_matrices

    def extract(self, audio_paths: list[str], batch_size: int, include_layers: bool = False) -> ExtractorResult:
        final_rows: list[np.ndarray] = []
        layer_rows: dict[str, list[np.ndarray]] = {}
        total_items = len(audio_paths)
        total_batches = (total_items + batch_size - 1) // batch_size
        start_time = time.time()
        if _LOG_EVERY > 0:
            print(
                f"[embed] start {self.model_name} device={self.device} "
                f"items={total_items} batch={batch_size} include_layers={include_layers}",
                flush=True,
            )

        for batch_index, start in enumerate(range(0, len(audio_paths), batch_size), start=1):
            batch_paths = audio_paths[start : start + batch_size]
            waveforms = [load_waveform(path, target_sample_rate=self.sample_rate) for path in batch_paths]
            batch_final, batch_layers = self.extract_decoder_hidden_states(waveforms)
            final_rows.append(batch_final)
            if include_layers:
                for layer_name, layer_matrix in batch_layers.items():
                    layer_rows.setdefault(layer_name, []).append(layer_matrix)
            _maybe_log_progress(
                model_name=self.model_name,
                device=self.device,
                batch_index=batch_index,
                total_batches=total_batches,
                total_items=total_items,
                batch_size=batch_size,
                start_time=start_time,
            )

        return ExtractorResult(
            final_name="final",
            final_matrix=np.concatenate(final_rows, axis=0),
            layer_matrices={
                layer_name: np.concatenate(parts, axis=0) for layer_name, parts in layer_rows.items()
            },
            metadata={
                "model_id": self.model_id,
                "sample_rate": self.sample_rate,
                "model_type": "musicgen_melody" if self.is_melody else "musicgen",
            },
        )

    def representation_names(self, include_layers: bool) -> list[str]:
        names = ["final"]
        if include_layers:
            names.extend(f"layer_{layer_index:02d}" for layer_index in range(self.hidden_state_count))
        return names


def get_extractor(
    model_name: str,
    *,
    device: str = "auto",
    cache_dir: str | None = None,
    local_files_only: bool = False,
    clap_variant: str = "audio_embedding",
) -> ClapExtractor | MulanExtractor | MertExtractor | MusicGenExtractor:
    normalized = model_name.strip().lower()
    resolved_device = resolve_device(device)
    if normalized == "clap":
        return ClapExtractor(
            device=resolved_device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            variant=clap_variant,
        )
    if normalized == "clap_music":
        return ClapExtractor(
            model_id=DEFAULT_CLAP_MUSIC_MODEL_ID,
            device=resolved_device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            variant="encoder",
            final_representation_name="final",
        )
    if normalized == "clap_music_speech":
        return ClapExtractor(
            model_id=DEFAULT_CLAP_MUSIC_SPEECH_MODEL_ID,
            device=resolved_device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            variant="encoder",
            final_representation_name="final",
        )
    if normalized == "mulan":
        return MulanExtractor(
            device=resolved_device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    if normalized == "mert":
        return MertExtractor(
            device=resolved_device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    if normalized == "mert95":
        return MertExtractor(
            model_id=DEFAULT_MERT95_MODEL_ID,
            device=resolved_device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    if normalized == "musicgen_small":
        return MusicGenExtractor(
            model_id=DEFAULT_MUSICGEN_SMALL_MODEL_ID,
            device=resolved_device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    if normalized == "musicgen_medium":
        return MusicGenExtractor(
            model_id=DEFAULT_MUSICGEN_MEDIUM_MODEL_ID,
            device=resolved_device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    if normalized == "musicgen_large":
        return MusicGenExtractor(
            model_id=DEFAULT_MUSICGEN_LARGE_MODEL_ID,
            device=resolved_device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    if normalized == "musicgen_melody_large":
        return MusicGenExtractor(
            model_id=DEFAULT_MUSICGEN_MELODY_LARGE_MODEL_ID,
            device=resolved_device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    raise ValueError(f"Unsupported model: {model_name!r}. Expected one of {SUPPORTED_MODEL_NAMES}.")
