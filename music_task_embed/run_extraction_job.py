from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ismir2026.music_task_embed.encoders import ClapExtractor, SUPPORTED_MODEL_NAMES, get_extractor, resolve_device
from ismir2026.music_task_embed.storage import (
    has_complete_partition_job,
    iter_partitions,
    load_manifest_bundle,
    save_partition_embeddings,
)


def _chunk_root(
    *,
    output_root: str | Path,
    bundle,
    model_name: str,
    partition_name: str,
) -> Path:
    return (
        Path(output_root)
        / bundle.dataset_name.lower()
        / bundle.task_name.lower()
        / model_name.lower()
        / "_partial"
        / partition_name
    )


def _chunk_path(
    *,
    output_root: str | Path,
    bundle,
    model_name: str,
    partition_name: str,
    chunk_index: int,
) -> Path:
    return _chunk_root(
        output_root=output_root,
        bundle=bundle,
        model_name=model_name,
        partition_name=partition_name,
    ) / f"chunk_{chunk_index:04d}.npz"


def _save_chunk_payload(
    *,
    chunk_path: Path,
    partition_df: pd.DataFrame,
    model_name: str,
    final_name: str,
    final_matrix: np.ndarray,
    layer_matrices: dict[str, np.ndarray],
    extractor_metadata: dict[str, object] | None,
) -> None:
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "sample_id": partition_df["sample_id"].astype(str).to_numpy(),
        "audio_path": partition_df["audio_path"].astype(str).to_numpy(),
        "model_name": np.array(model_name, dtype=object),
        "final_name": np.array(final_name, dtype=object),
        "extractor_metadata": np.array(json.dumps(extractor_metadata or {}, sort_keys=True), dtype=object),
        f"embeddings__{final_name}": np.asarray(final_matrix, dtype=np.float32),
    }
    for layer_name, layer_matrix in sorted(layer_matrices.items()):
        payload[f"embeddings__{layer_name}"] = np.asarray(layer_matrix, dtype=np.float32)
    np.savez_compressed(chunk_path, **payload)


def _chunk_complete(
    *,
    chunk_path: Path,
    partition_df: pd.DataFrame,
    model_name: str,
    representation_names: list[str],
) -> bool:
    if not chunk_path.exists():
        return False
    try:
        with np.load(chunk_path, allow_pickle=True) as payload:
            sample_id = np.asarray(payload["sample_id"]).astype(str)
            saved_model_name = str(payload["model_name"].item())
            saved_final_name = str(payload["final_name"].item())
            available_keys = set(payload.files)
            shapes = {
                name: tuple(np.asarray(payload[f"embeddings__{name}"]).shape)
                for name in representation_names
                if f"embeddings__{name}" in available_keys
            }
    except Exception:
        return False

    if saved_model_name.lower() != model_name.lower():
        return False
    if saved_final_name != representation_names[0]:
        return False
    expected_sample_ids = partition_df["sample_id"].astype(str).to_numpy()
    if not np.array_equal(sample_id, expected_sample_ids):
        return False
    for representation_name in representation_names:
        key = f"embeddings__{representation_name}"
        if key not in available_keys:
            return False
        if shapes[representation_name][0] != len(partition_df):
            return False
    return True


def _assemble_from_chunks(
    *,
    bundle,
    partition_name: str,
    partition_df: pd.DataFrame,
    model_name: str,
    representation_names: Iterable[str],
    output_root: str | Path,
    extractor_metadata: dict[str, object] | None,
    total_chunks: int,
) -> None:
    representation_names = list(representation_names)
    chunk_paths = [
        _chunk_path(
            output_root=output_root,
            bundle=bundle,
            model_name=model_name,
            partition_name=partition_name,
            chunk_index=chunk_index,
        )
        for chunk_index in range(total_chunks)
    ]
    chunk_payloads = [np.load(path, allow_pickle=True) for path in chunk_paths]
    try:
        for representation_name in representation_names:
            key = f"embeddings__{representation_name}"
            embeddings = np.concatenate(
                [np.asarray(payload[key], dtype=np.float32) for payload in chunk_payloads],
                axis=0,
            )
            output_path = save_partition_embeddings(
                bundle=bundle,
                partition_name=partition_name,
                partition_df=partition_df,
                embeddings=embeddings,
                model_name=model_name,
                representation_name=representation_name,
                output_root=output_root,
                extractor_metadata=extractor_metadata,
            )
            print(output_path)
    finally:
        for payload in chunk_payloads:
            payload.close()


def _run_checkpointed_partition(
    *,
    extractor,
    bundle,
    partition_name: str,
    partition_df: pd.DataFrame,
    model_name: str,
    output_root: str | Path,
    batch_size: int,
    include_layers: bool,
) -> None:
    representation_names = list(extractor.representation_names(include_layers))
    total_items = len(partition_df)
    total_chunks = (total_items + batch_size - 1) // batch_size
    last_metadata: dict[str, object] | None = None

    for chunk_index, start in enumerate(range(0, total_items, batch_size)):
        chunk_df = partition_df.iloc[start : start + batch_size].reset_index(drop=True)
        chunk_path = _chunk_path(
            output_root=output_root,
            bundle=bundle,
            model_name=model_name,
            partition_name=partition_name,
            chunk_index=chunk_index,
        )
        if _chunk_complete(
            chunk_path=chunk_path,
            partition_df=chunk_df,
            model_name=model_name,
            representation_names=representation_names,
        ):
            print(
                f"SKIP-CHUNK {bundle.dataset_name}/{bundle.task_name} {model_name} "
                f"{partition_name} chunk={chunk_index + 1}/{total_chunks}",
                flush=True,
            )
            continue

        chunk_audio_paths = chunk_df["audio_path"].astype(str).tolist()
        result = extractor.extract(
            chunk_audio_paths,
            batch_size=min(batch_size, len(chunk_audio_paths)),
            include_layers=include_layers,
        )
        last_metadata = result.metadata
        _save_chunk_payload(
            chunk_path=chunk_path,
            partition_df=chunk_df,
            model_name=model_name,
            final_name=result.final_name,
            final_matrix=result.final_matrix,
            layer_matrices=result.layer_matrices,
            extractor_metadata=result.metadata,
        )
        print(
            f"SAVED-CHUNK {bundle.dataset_name}/{bundle.task_name} {model_name} "
            f"{partition_name} chunk={chunk_index + 1}/{total_chunks}",
            flush=True,
        )

    if last_metadata is None:
        first_chunk_path = _chunk_path(
            output_root=output_root,
            bundle=bundle,
            model_name=model_name,
            partition_name=partition_name,
            chunk_index=0,
        )
        with np.load(first_chunk_path, allow_pickle=True) as payload:
            last_metadata = json.loads(str(payload["extractor_metadata"].item()))

    _assemble_from_chunks(
        bundle=bundle,
        partition_name=partition_name,
        partition_df=partition_df,
        model_name=model_name,
        representation_names=representation_names,
        output_root=output_root,
        extractor_metadata=last_metadata,
        total_chunks=total_chunks,
    )
    shutil.rmtree(
        _chunk_root(
            output_root=output_root,
            bundle=bundle,
            model_name=model_name,
            partition_name=partition_name,
        ),
        ignore_errors=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run embedding extraction jobs for one or more manifests.")
    parser.add_argument("--manifest-dir", action="append", required=True)
    parser.add_argument("--model-name", choices=SUPPORTED_MODEL_NAMES, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--include-layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--partition-column", choices=["split", "fold"], default=None)
    parser.add_argument("--clap-variant", choices=["audio_embedding", "encoder"], default="audio_embedding")
    parser.add_argument("--clap-model-id", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.model_name == "clap" and args.clap_model_id:
        extractor = ClapExtractor(
            model_id=args.clap_model_id,
            device=resolve_device(args.device),
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            variant=args.clap_variant,
        )
    else:
        extractor = get_extractor(
            args.model_name,
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            clap_variant=args.clap_variant,
        )
    representation_names = (
        extractor.representation_names(args.include_layers)
        if hasattr(extractor, "representation_names")
        else None
    )

    for manifest_dir in args.manifest_dir:
        bundle = load_manifest_bundle(manifest_dir)
        for partition_name, partition_df in iter_partitions(
            bundle.manifest,
            partition_column=args.partition_column,
        ):
            if representation_names is not None and has_complete_partition_job(
                bundle=bundle,
                partition_name=partition_name,
                partition_df=partition_df,
                model_name=args.model_name,
                representation_names=representation_names,
                output_root=args.output_root,
            ):
                print(
                    f"SKIP {bundle.dataset_name}/{bundle.task_name} "
                    f"{args.model_name} {partition_name}"
                )
                continue
            print(
                f"START {bundle.dataset_name}/{bundle.task_name} "
                f"{args.model_name} {partition_name} n={len(partition_df)}",
                flush=True,
            )
            if args.model_name == "jukebox":
                _run_checkpointed_partition(
                    extractor=extractor,
                    bundle=bundle,
                    partition_name=partition_name,
                    partition_df=partition_df,
                    model_name=args.model_name,
                    output_root=args.output_root,
                    batch_size=args.batch_size,
                    include_layers=args.include_layers,
                )
            else:
                result = extractor.extract(
                    partition_df["audio_path"].astype(str).tolist(),
                    batch_size=args.batch_size,
                    include_layers=args.include_layers,
                )
                output_path = save_partition_embeddings(
                    bundle=bundle,
                    partition_name=partition_name,
                    partition_df=partition_df,
                    embeddings=result.final_matrix,
                    model_name=args.model_name,
                    representation_name=result.final_name,
                    output_root=args.output_root,
                    extractor_metadata=result.metadata,
                )
                print(output_path)
                for layer_name, layer_matrix in sorted(result.layer_matrices.items()):
                    layer_path = save_partition_embeddings(
                        bundle=bundle,
                        partition_name=partition_name,
                        partition_df=partition_df,
                        embeddings=layer_matrix,
                        model_name=args.model_name,
                        representation_name=layer_name,
                        output_root=args.output_root,
                        extractor_metadata=result.metadata,
                    )
                    print(layer_path)
            print(
                f"DONE {bundle.dataset_name}/{bundle.task_name} "
                f"{args.model_name} {partition_name}",
                flush=True,
            )


if __name__ == "__main__":
    main()
