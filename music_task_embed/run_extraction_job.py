from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
