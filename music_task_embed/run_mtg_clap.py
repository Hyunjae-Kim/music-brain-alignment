from __future__ import annotations

import argparse
from pathlib import Path

from .download_prepare import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MTG_AUDIO_ROOT,
    DEFAULT_MANIFEST_ROOT,
    prepare_mtg_manifests,
)
from . import encoders
from .encoders import SUPPORTED_MODEL_NAMES, get_extractor
from .storage import (
    has_complete_partition_job,
    iter_partitions,
    load_manifest_bundle,
    save_partition_embeddings,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "embeddings"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare MTG manifests and extract embeddings for MTG tasks."
    )
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--manifest-root", type=str, default=str(DEFAULT_MANIFEST_ROOT))
    parser.add_argument("--mtg-audio-root", type=str, default=str(DEFAULT_MTG_AUDIO_ROOT))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-name", choices=SUPPORTED_MODEL_NAMES, default="clap")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--include-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Log progress every N batches (0 disables batch logging).",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=["mtg_top50", "mtg_mood", "mtg_genre", "mtg_instrument"],
        default=None,
        help="Optional MTG task filter. Repeat to include multiple tasks.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip manifest preparation (expects manifests already exist).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    data_root = Path(args.data_root)
    manifest_root = Path(args.manifest_root)
    audio_root = Path(args.mtg_audio_root)
    output_root = Path(args.output_root)

    data_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        prepare_mtg_manifests(
            data_root=data_root,
            manifest_root=manifest_root,
            audio_root=audio_root,
        )

    task_to_manifest = {
        "mtg_top50": manifest_root / "mtgtop50" / "music_tagging",
        "mtg_mood": manifest_root / "mtgmoodtheme" / "emotion_classification",
        "mtg_genre": manifest_root / "mtggenre" / "genre_classification",
        "mtg_instrument": manifest_root / "mtginstrument" / "instrument_classification",
    }
    selected_tasks = args.task or list(task_to_manifest.keys())
    manifest_dirs = [task_to_manifest[task] for task in selected_tasks]
    for manifest_dir in manifest_dirs:
        if not (manifest_dir / "manifest.csv").exists():
            raise FileNotFoundError(f"Missing manifest.csv at {manifest_dir}")

    if args.log_every and args.log_every > 0:
        encoders._LOG_EVERY = int(args.log_every)

    model_name = args.model_name
    extractor = get_extractor(
        model_name,
        device=args.device,
        cache_dir=None,
        local_files_only=args.local_files_only,
    )
    representation_names = (
        extractor.representation_names(args.include_layers)
        if hasattr(extractor, "representation_names")
        else None
    )

    for manifest_dir in manifest_dirs:
        bundle = load_manifest_bundle(manifest_dir)
        manifest = bundle.manifest
        if args.limit is not None:
            manifest = manifest.iloc[: args.limit].reset_index(drop=True)

        for partition_name, partition_df in iter_partitions(manifest):
            if representation_names is not None and has_complete_partition_job(
                bundle=bundle,
                partition_name=partition_name,
                partition_df=partition_df,
                model_name=model_name,
                representation_names=representation_names,
                output_root=output_root,
            ):
                print(f"SKIP {bundle.dataset_name}/{bundle.task_name} {model_name} {partition_name}")
                continue

            print(
                f"START {bundle.dataset_name}/{bundle.task_name} "
                f"{model_name} {partition_name} n={len(partition_df)}",
                flush=True,
            )
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
                model_name=model_name,
                representation_name=result.final_name,
                output_root=output_root,
                extractor_metadata=result.metadata,
            )
            print(output_path)
            for layer_name, layer_matrix in sorted(result.layer_matrices.items()):
                layer_path = save_partition_embeddings(
                    bundle=bundle,
                    partition_name=partition_name,
                    partition_df=partition_df,
                    embeddings=layer_matrix,
                    model_name=model_name,
                    representation_name=layer_name,
                    output_root=output_root,
                    extractor_metadata=result.metadata,
                )
                print(layer_path)
            print(
                f"DONE {bundle.dataset_name}/{bundle.task_name} "
                f"{model_name} {partition_name}",
                flush=True,
            )


if __name__ == "__main__":
    main()
