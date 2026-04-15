from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ismir2026.music_task_embed.storage import (  # noqa: E402
    has_complete_partition_embeddings,
    iter_partitions,
    load_manifest_bundle,
    save_partition_embeddings,
)
from ismir2026.noise_music_task_embed.encoders import (  # noqa: E402
    SUPPORTED_MODEL_NAMES,
    get_base_extractor,
    wrap_extractor_with_noise,
)
from ismir2026.noise_music_task_embed.noise import (  # noqa: E402
    build_noise_conditions,
    parse_snr_levels,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMO_MANIFEST_DIR = WORKSPACE_ROOT / "music_task_embed" / "manifests" / "emo" / "emotion_regression"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "noise_music_task_embed" / "embeddings"
DEFAULT_MODEL_LAYER = (
    "mulan:layer_08",
    "musicgen_melody_large:layer_35",
)
DEFAULT_CONDITIONS = (
    "noise_l1_snr_p20p0db",
    "noise_l2_snr_p10p0db",
    "noise_l3_snr_p0p0db",
)


def _parse_model_layer(entries: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"--model-layer must be 'model:layer', got: {entry}")
        model_name, layer_name = entry.split(":", 1)
        model_name = model_name.strip().lower()
        layer_name = layer_name.strip()
        if model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(f"Unsupported model '{model_name}'.")
        if not layer_name.startswith("layer_"):
            raise ValueError(f"Layer name must start with 'layer_': {layer_name}")
        parsed.append((model_name, layer_name))
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract only selected hidden layers for noise-robustness probing.",
    )
    parser.add_argument("--manifest-dir", action="append", default=[str(DEFAULT_EMO_MANIFEST_DIR)])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--partition-column", choices=["split", "fold"], default=None)
    parser.add_argument("--noise-seed", type=int, default=1337)
    parser.add_argument("--clap-variant", choices=["audio_embedding", "encoder"], default="audio_embedding")

    parser.add_argument(
        "--model-layer",
        action="append",
        default=None,
        help="Repeatable. Format: model_name:layer_xx",
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=None,
        help="Noise condition id to run (repeatable).",
    )
    parser.add_argument("--include-clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise-snrs-db", type=str, default="20,10,0,-10")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model_layers = _parse_model_layer(args.model_layer or list(DEFAULT_MODEL_LAYER))
    manifest_dirs = [Path(path) for path in args.manifest_dir]
    bundles = [load_manifest_bundle(path) for path in manifest_dirs]

    all_conditions = build_noise_conditions(
        snr_levels_db=parse_snr_levels(args.noise_snrs_db),
        include_clean=bool(args.include_clean),
    )
    selected_ids = set(args.condition or list(DEFAULT_CONDITIONS))
    if args.include_clean:
        selected_ids.add("clean")
    conditions = [condition for condition in all_conditions if condition.condition_id in selected_ids]
    if not conditions:
        raise ValueError("No valid conditions selected.")

    for model_name, target_layer in model_layers:
        base_extractor = get_base_extractor(
            model_name,
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            clap_variant=args.clap_variant,
        )

        for condition in conditions:
            extractor = wrap_extractor_with_noise(
                base_extractor,
                condition=condition,
                seed=int(args.noise_seed),
            )
            condition_output_root = Path(args.output_root) / condition.condition_id

            for bundle in bundles:
                for partition_name, partition_df in iter_partitions(bundle.manifest, partition_column=args.partition_column):
                    if has_complete_partition_embeddings(
                        bundle=bundle,
                        partition_name=partition_name,
                        partition_df=partition_df,
                        model_name=model_name,
                        representation_name=target_layer,
                        output_root=condition_output_root,
                    ):
                        print(
                            f"SKIP {condition.condition_id} {bundle.dataset_name}/{bundle.task_name} "
                            f"{model_name} {target_layer} {partition_name}",
                            flush=True,
                        )
                        continue

                    print(
                        f"START {condition.condition_id} {bundle.dataset_name}/{bundle.task_name} "
                        f"{model_name} {target_layer} {partition_name} n={len(partition_df)}",
                        flush=True,
                    )
                    result = extractor.extract(
                        partition_df["audio_path"].astype(str).tolist(),
                        batch_size=int(args.batch_size),
                        include_layers=True,
                    )
                    if target_layer not in result.layer_matrices:
                        available = ", ".join(sorted(result.layer_matrices.keys())[:5])
                        raise KeyError(
                            f"Layer {target_layer} not found for {model_name}. Available prefixes: {available}"
                        )
                    layer_matrix = result.layer_matrices[target_layer]
                    metadata = dict(result.metadata)
                    metadata["selected_layer_only"] = target_layer

                    out_path = save_partition_embeddings(
                        bundle=bundle,
                        partition_name=partition_name,
                        partition_df=partition_df,
                        embeddings=layer_matrix,
                        model_name=model_name,
                        representation_name=target_layer,
                        output_root=condition_output_root,
                        extractor_metadata=metadata,
                    )
                    print(out_path, flush=True)
                    print(
                        f"DONE {condition.condition_id} {bundle.dataset_name}/{bundle.task_name} "
                        f"{model_name} {target_layer} {partition_name}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
