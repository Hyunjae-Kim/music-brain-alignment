from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from ismir2026.natural_sound_align.data import load_local_norman_metadata
from ismir2026.music_task_embed.encoders import get_extractor


DEFAULT_IMAGE = "jukemir/representations_jukebox"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent / "foundation_test" / "outputs" / "jukebox"


@dataclass
class ExtractionConfig:
    output_dir: str
    model_id: str
    device: str
    wav_dir: str
    cache_dir: str | None
    batch_size: int
    force_reextract: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Jukebox representations for the Norman 165 clips.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_IMAGE,
        help="Stored metadata identifier for the Jukebox extractor backend.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Execution device hint. Use cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument(
        "--wav-dir",
        type=str,
        default=str((PROJECT_ROOT / "dataset" / "norman2015" / "stimuli" / "wav").resolve()),
        help="Directory containing the Norman stimulus wav files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional cache directory for jukemirlib Jukebox checkpoints.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for jukemirlib extraction. Small values are safer for VRAM.",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Ignore any cached per-file Jukebox features and rerun the Docker extractor.",
    )
    parser.add_argument(
        "--save-intermediate-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save all Jukebox top-prior hidden layers under outputs/jukebox/layers/.",
    )
    return parser


def _save_outputs(
    output_dir: Path,
    final_matrix: np.ndarray,
    *,
    layer_matrices: dict[str, np.ndarray],
    image: str,
    wav_dir: Path,
    backend: str,
    cache_dir: str | None,
    batch_size: int,
) -> None:
    metadata = load_local_norman_metadata(wav_dir=wav_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / "stimulus_metadata.csv", index=False)

    np.savez_compressed(
        output_dir / "combined_representations.npz",
        representations=final_matrix.astype(np.float32, copy=False),
        stim_names=metadata["stim_name"].to_numpy(),
        category_labels=metadata["category_label"].to_numpy(),
        wav_paths=metadata["wav_path"].to_numpy(),
        stimulus_index=metadata["stimulus_index"].to_numpy(),
        representation_dim=np.int32(final_matrix.shape[1]),
        sampling_rate=np.int32(44_100),
        model_id=np.array("openai/jukebox-5b-top-prior", dtype=object),
        source_key=np.array("meanpool_top_prior_hidden_state", dtype=object),
        docker_image=np.array(image, dtype=object),
    )

    layers_dir = output_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    for layer_name, matrix in sorted(layer_matrices.items()):
        layer_dir = layers_dir / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            layer_dir / "combined_representations.npz",
            representations=matrix.astype(np.float32, copy=False),
            stim_names=metadata["stim_name"].to_numpy(),
            category_labels=metadata["category_label"].to_numpy(),
            wav_paths=metadata["wav_path"].to_numpy(),
            stimulus_index=metadata["stimulus_index"].to_numpy(),
            representation_dim=np.int32(matrix.shape[1]),
            sampling_rate=np.int32(44_100),
            model_id=np.array("openai/jukebox-5b-top-prior", dtype=object),
            source_key=np.array("meanpool_top_prior_hidden_state", dtype=object),
            docker_image=np.array(image, dtype=object),
        )
        layer_index = int(layer_name.split("_", 1)[1])
        index_rows.append(
            {
                "layer_index": layer_index,
                "representation_name": layer_name,
                "representation_dim": int(matrix.shape[1]),
                "path": str(layer_dir / "combined_representations.npz"),
            }
        )

    with open(layers_dir / "index.json", "w", encoding="utf-8") as file:
        json.dump({"layers": index_rows}, file, indent=2)

    config_payload = asdict(
        ExtractionConfig(
            output_dir=str(output_dir),
            model_id=image,
            device="docker-managed" if backend == "docker" else "python-managed",
            wav_dir=str(wav_dir),
            cache_dir=cache_dir,
            batch_size=batch_size,
            force_reextract=False,
        )
    )
    config_payload["representation_dim"] = int(final_matrix.shape[1])
    config_payload["source_key"] = "meanpool_top_prior_hidden_state"
    config_payload["backend"] = backend
    config_payload["notes"] = "Jukebox is exported as mean pooled top-prior hidden states for all 72 layers; final duplicates layer_72."
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config_payload, file, indent=2)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    wav_dir = Path(args.wav_dir).resolve()
    final_path = output_dir / "combined_representations.npz"
    if final_path.exists() and not args.force_reextract:
        print(f"Using cached Jukebox embeddings: {final_path}")
        return

    if not wav_dir.exists():
        raise FileNotFoundError(f"Wav directory not found: {wav_dir}")

    extractor = get_extractor(
        "jukebox",
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=False,
    )
    metadata = load_local_norman_metadata(wav_dir=wav_dir)
    result = extractor.extract(
        metadata["wav_path"].astype(str).tolist(),
        batch_size=max(1, args.batch_size),
        include_layers=bool(args.save_intermediate_layers),
    )

    _save_outputs(
        output_dir,
        result.final_matrix,
        layer_matrices=result.layer_matrices,
        image=args.model_id,
        wav_dir=wav_dir,
        backend="jukemirlib",
        cache_dir=args.cache_dir,
        batch_size=max(1, args.batch_size),
    )
    print(f"Saved: {final_path}")


if __name__ == "__main__":
    main()
