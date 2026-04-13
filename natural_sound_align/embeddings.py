from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .data import (
    PACKAGE_ROOT,
    PROJECT_ROOT,
    load_local_norman_metadata,
    load_tuckute_stimulus_metadata,
    normalize_dataset_name,
)


FOUNDATION_OUTPUTS = PROJECT_ROOT / "foundation_test" / "outputs"
SPECTEMP_CACHE = PACKAGE_ROOT / "cache" / "model_actv" / "spectemp" / "natsound_activations.h5"

MODEL_SPECS = {
    "CLAP": {
        "final_path": FOUNDATION_OUTPUTS / "clap_encoder" / "combined_representations.npz",
        "layers_index": FOUNDATION_OUTPUTS / "clap_encoder" / "layers" / "index.json",
        "matrix_key": "encoder_representations",
        "script": PROJECT_ROOT / "foundation_test" / "scripts" / "extract_clap_embeddings.py",
        "output_dir": FOUNDATION_OUTPUTS / "clap_encoder",
        "hf_model_id": "laion/clap-htsat-unfused",
    },
    "CLAP_MUSIC": {
        "final_path": FOUNDATION_OUTPUTS / "clap_music" / "combined_representations.npz",
        "layers_index": FOUNDATION_OUTPUTS / "clap_music" / "layers" / "index.json",
        "matrix_key": "encoder_representations",
        "script": PROJECT_ROOT / "foundation_test" / "scripts" / "extract_clap_embeddings.py",
        "output_dir": FOUNDATION_OUTPUTS / "clap_music",
        "hf_model_id": "laion/larger_clap_music",
    },
    "CLAP_MUSIC_SPEECH": {
        "final_path": FOUNDATION_OUTPUTS / "clap_music_speech" / "combined_representations.npz",
        "layers_index": FOUNDATION_OUTPUTS / "clap_music_speech" / "layers" / "index.json",
        "matrix_key": "encoder_representations",
        "script": PROJECT_ROOT / "foundation_test" / "scripts" / "extract_clap_embeddings.py",
        "output_dir": FOUNDATION_OUTPUTS / "clap_music_speech",
        "hf_model_id": "laion/larger_clap_music_and_speech",
    },
    "MULAN": {
        "final_path": FOUNDATION_OUTPUTS / "mulan" / "combined_representations.npz",
        "layers_index": FOUNDATION_OUTPUTS / "mulan" / "layers" / "index.json",
        "matrix_key": "representations",
        "script": PROJECT_ROOT / "foundation_test" / "scripts" / "extract_foundation_model_embeddings.py",
        "output_dir": FOUNDATION_OUTPUTS / "mulan",
        "model_name_arg": "mulan",
        "hf_model_id": "OpenMuQ/MuQ-MuLan-large",
    },
    "MERT": {
        "final_path": FOUNDATION_OUTPUTS / "mert" / "combined_representations.npz",
        "layers_index": FOUNDATION_OUTPUTS / "mert" / "layers" / "index.json",
        "matrix_key": "representations",
        "script": PROJECT_ROOT / "foundation_test" / "scripts" / "extract_foundation_model_embeddings.py",
        "output_dir": FOUNDATION_OUTPUTS / "mert",
        "model_name_arg": "mert",
        "hf_model_id": "m-a-p/MERT-v1-330M",
    },
    "MERT95": {
        "final_path": FOUNDATION_OUTPUTS / "mert95" / "combined_representations.npz",
        "layers_index": FOUNDATION_OUTPUTS / "mert95" / "layers" / "index.json",
        "matrix_key": "representations",
        "script": PROJECT_ROOT / "foundation_test" / "scripts" / "extract_foundation_model_embeddings.py",
        "output_dir": FOUNDATION_OUTPUTS / "mert95",
        "model_name_arg": "mert",
        "hf_model_id": "m-a-p/MERT-v1-95M",
    },
    "MERT330": {
        "final_path": FOUNDATION_OUTPUTS / "mert" / "combined_representations.npz",
        "layers_index": FOUNDATION_OUTPUTS / "mert" / "layers" / "index.json",
        "matrix_key": "representations",
        "script": PROJECT_ROOT / "foundation_test" / "scripts" / "extract_foundation_model_embeddings.py",
        "output_dir": FOUNDATION_OUTPUTS / "mert",
        "model_name_arg": "mert",
        "hf_model_id": "m-a-p/MERT-v1-330M",
    },
    "MUSICGEN_SMALL": {
        "final_path": FOUNDATION_OUTPUTS / "musicgen_small" / "combined_representations.npz",
        "layers_index": FOUNDATION_OUTPUTS / "musicgen_small" / "layers" / "index.json",
        "matrix_key": "representations",
        "script": PROJECT_ROOT / "foundation_test" / "scripts" / "extract_musicgen_decoder_embeddings.py",
        "output_dir": FOUNDATION_OUTPUTS / "musicgen_small",
        "hf_model_id": "facebook/musicgen-small",
    },
    "MUSICGEN_MEDIUM": {
        "final_path": FOUNDATION_OUTPUTS / "musicgen_medium" / "combined_representations.npz",
        "layers_index": FOUNDATION_OUTPUTS / "musicgen_medium" / "layers" / "index.json",
        "matrix_key": "representations",
        "script": PROJECT_ROOT / "foundation_test" / "scripts" / "extract_musicgen_decoder_embeddings.py",
        "output_dir": FOUNDATION_OUTPUTS / "musicgen_medium",
        "hf_model_id": "facebook/musicgen-medium",
    },
    "MUSICGEN_LARGE": {
        "final_path": FOUNDATION_OUTPUTS / "musicgen_large" / "combined_representations.npz",
        "layers_index": FOUNDATION_OUTPUTS / "musicgen_large" / "layers" / "index.json",
        "matrix_key": "representations",
        "script": PROJECT_ROOT / "foundation_test" / "scripts" / "extract_musicgen_decoder_embeddings.py",
        "output_dir": FOUNDATION_OUTPUTS / "musicgen_large",
        "hf_model_id": "facebook/musicgen-large",
    },
    "MUSICGEN_MELODY_LARGE": {
        "final_path": FOUNDATION_OUTPUTS / "musicgen_melody_large" / "combined_representations.npz",
        "layers_index": FOUNDATION_OUTPUTS / "musicgen_melody_large" / "layers" / "index.json",
        "matrix_key": "representations",
        "script": PROJECT_ROOT / "foundation_test" / "scripts" / "extract_musicgen_decoder_embeddings.py",
        "output_dir": FOUNDATION_OUTPUTS / "musicgen_melody_large",
        "hf_model_id": "facebook/musicgen-melody-large",
    },
    "MUSICGEN_MELODY": {
        "final_path": FOUNDATION_OUTPUTS / "musicgen_melody" / "combined_representations.npz",
        "layers_index": FOUNDATION_OUTPUTS / "musicgen_melody" / "layers" / "index.json",
        "matrix_key": "representations",
        "script": PROJECT_ROOT / "foundation_test" / "scripts" / "extract_musicgen_decoder_embeddings.py",
        "output_dir": FOUNDATION_OUTPUTS / "musicgen_melody",
        "hf_model_id": "facebook/musicgen-melody",
    },
    "MUSICGEN_LARGE_MELODY": {
        "final_path": FOUNDATION_OUTPUTS / "musicgen_melody_large" / "combined_representations.npz",
        "layers_index": FOUNDATION_OUTPUTS / "musicgen_melody_large" / "layers" / "index.json",
        "matrix_key": "representations",
        "script": PROJECT_ROOT / "foundation_test" / "scripts" / "extract_musicgen_decoder_embeddings.py",
        "output_dir": FOUNDATION_OUTPUTS / "musicgen_melody_large",
        "hf_model_id": "facebook/musicgen-melody-large",
    },
    "SPECTEMP": {
        "final_path": SPECTEMP_CACHE,
        "layers_index": None,
        "matrix_key": "avgpool",
        "output_dir": SPECTEMP_CACHE.parent,
    },
}


@dataclass(frozen=True)
class EmbeddingBundle:
    model_name: str
    representation_name: str
    matrix: np.ndarray
    metadata: pd.DataFrame
    source_path: Path
    model_id: str | None
    layer_index: int | None = None


def _normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().upper()
    if normalized not in MODEL_SPECS:
        raise ValueError(f"Unsupported model {model_name!r}. Expected one of {sorted(MODEL_SPECS)}.")
    return normalized


def available_layer_indices(model_name: str) -> list[int]:
    spec = MODEL_SPECS[_normalize_model_name(model_name)]
    index_path = spec["layers_index"]
    if index_path is None or not index_path.exists():
        return []
    payload = json.loads(index_path.read_text())
    return [int(layer["layer_index"]) for layer in payload.get("layers", [])]


def _load_npz_metadata(npz_file: np.lib.npyio.NpzFile) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stimulus_index": npz_file["stimulus_index"].astype(int),
            "stim_name": npz_file["stim_names"].astype(str),
            "category_label": npz_file["category_labels"].astype(str),
            "wav_path": npz_file["wav_paths"].astype(str),
        }
    )


def _resolve_embedding_path(model_name: str, layer_index: int | None) -> tuple[Path, str]:
    spec = MODEL_SPECS[_normalize_model_name(model_name)]
    if layer_index is None:
        return Path(spec["final_path"]), str(spec["matrix_key"])
    layer_dir = Path(spec["output_dir"]) / "layers" / f"layer_{int(layer_index):02d}"
    return layer_dir / "combined_representations.npz", "representations"


def _reindex_to_target_order(
    matrix: np.ndarray,
    metadata: pd.DataFrame,
    *,
    dataset_name: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    target_metadata = load_tuckute_stimulus_metadata(dataset_name)
    current_index = {stim_name: idx for idx, stim_name in enumerate(metadata["stim_name"].tolist())}
    order = [current_index[stim_name] for stim_name in target_metadata["stim_name"].tolist()]
    return matrix[order], target_metadata


def load_model_embedding(
    model_name: str,
    layer_index: int | None = None,
    *,
    dataset_name: str = "NH2015",
) -> EmbeddingBundle:
    normalized_model = _normalize_model_name(model_name)
    normalized_dataset = normalize_dataset_name(dataset_name)
    path, matrix_key = _resolve_embedding_path(normalized_model, layer_index)
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding file not found for {normalized_model} layer={layer_index}: {path}"
        )

    if normalized_model == "SPECTEMP":
        with h5py.File(path, "r") as h5_file:
            matrix = np.asarray(h5_file[matrix_key], dtype=np.float32)
        metadata = load_tuckute_stimulus_metadata(normalized_dataset)
        model_id = "Tuckute_spectemp"
        representation_name = "avgpool"
    else:
        with np.load(path, allow_pickle=True) as npz_file:
            if matrix_key not in npz_file.files:
                raise KeyError(f"Matrix key {matrix_key!r} not found in {path}")
            matrix = np.asarray(npz_file[matrix_key], dtype=np.float32)
            metadata = _load_npz_metadata(npz_file)
            model_id = str(npz_file["model_id"].item()) if "model_id" in npz_file.files else None

        reference_metadata = load_local_norman_metadata()
        if metadata["stim_name"].tolist() != reference_metadata["stim_name"].tolist():
            raise ValueError(f"Stimulus order mismatch between {path} and local Norman metadata.")
        matrix, metadata = _reindex_to_target_order(matrix, metadata, dataset_name=normalized_dataset)
        representation_name = "final" if layer_index is None else f"layer_{int(layer_index):02d}"

    return EmbeddingBundle(
        model_name=normalized_model,
        representation_name=representation_name,
        matrix=matrix,
        metadata=metadata,
        source_path=path,
        model_id=model_id,
        layer_index=layer_index,
    )


def ensure_model_embeddings(
    model_name: str,
    *,
    overwrite: bool = False,
    device: str = "cuda",
    python_executable: str | None = None,
) -> Path:
    normalized = _normalize_model_name(model_name)
    spec = MODEL_SPECS[normalized]
    final_path = Path(spec["final_path"])
    if final_path.exists() and not overwrite:
        return final_path
    if normalized == "SPECTEMP":
        raise FileNotFoundError(
            "SPECTEMP activations are expected at "
            f"{final_path}. Download the Tuckute spectemp h5 file first."
        )

    python_executable = python_executable or sys.executable
    command = [
        python_executable,
        str(spec["script"]),
        "--device",
        device,
        "--output-dir",
        str(spec["output_dir"]),
        "--model-id",
        str(spec["hf_model_id"]),
        "--save-intermediate-layers",
    ]
    if "model_name_arg" in spec:
        command.extend(["--model-name", str(spec["model_name_arg"])])

    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    if not final_path.exists():
        raise FileNotFoundError(f"Expected embedding output was not created: {final_path}")
    return final_path
