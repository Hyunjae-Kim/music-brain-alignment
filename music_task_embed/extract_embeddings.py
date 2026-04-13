from __future__ import annotations

import argparse
from pathlib import Path

from .encoders import SUPPORTED_MODEL_NAMES, get_extractor
from .storage import (
    has_complete_partition_job,
    iter_partitions,
    load_manifest_bundle,
    save_partition_embeddings,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST_ROOT = PACKAGE_ROOT / "manifests"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "embeddings"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract CLAP / MuLAN / MERT embeddings for local MIR manifests.")
    parser.add_argument("--manifest-dir", type=str, required=True, help="Directory containing manifest.csv and task_metadata.json.")
    parser.add_argument("--model-name", choices=SUPPORTED_MODEL_NAMES, required=True)
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--include-layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--partition-column", choices=["split", "fold"], default=None)
    parser.add_argument("--clap-variant", choices=["audio_embedding", "encoder"], default="audio_embedding")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick smoke tests.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = load_manifest_bundle(args.manifest_dir)
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

    manifest = bundle.manifest
    if args.limit is not None:
        manifest = manifest.iloc[: args.limit].reset_index(drop=True)

    saved_paths: list[Path] = []
    for partition_name, partition_df in iter_partitions(manifest, partition_column=args.partition_column):
        if representation_names is not None and has_complete_partition_job(
            bundle=bundle,
            partition_name=partition_name,
            partition_df=partition_df,
            model_name=args.model_name,
            representation_names=representation_names,
            output_root=args.output_root,
        ):
            print(f"SKIP {bundle.dataset_name}/{bundle.task_name} {args.model_name} {partition_name}")
            continue
        result = extractor.extract(
            partition_df["audio_path"].astype(str).tolist(),
            batch_size=args.batch_size,
            include_layers=args.include_layers,
        )
        saved_paths.append(
            save_partition_embeddings(
                bundle=bundle,
                partition_name=partition_name,
                partition_df=partition_df,
                embeddings=result.final_matrix,
                model_name=args.model_name,
                representation_name=result.final_name,
                output_root=args.output_root,
                extractor_metadata=result.metadata,
            )
        )
        for layer_name, layer_matrix in sorted(result.layer_matrices.items()):
            saved_paths.append(
                save_partition_embeddings(
                    bundle=bundle,
                    partition_name=partition_name,
                    partition_df=partition_df,
                    embeddings=layer_matrix,
                    model_name=args.model_name,
                    representation_name=layer_name,
                    output_root=args.output_root,
                    extractor_metadata=result.metadata,
                )
            )

    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
