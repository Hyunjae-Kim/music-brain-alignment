from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ismir2026.music_task_embed.storage import (  # noqa: E402
    has_complete_partition_job,
    iter_partitions,
    load_manifest_bundle,
    save_partition_embeddings,
)
from ismir2026.music_task_embed import encoders as base_encoders  # noqa: E402

from ismir2026.noise_music_task_embed.encoders import (  # noqa: E402
    SUPPORTED_MODEL_NAMES,
    get_base_extractor,
    wrap_extractor_with_noise,
)
from ismir2026.noise_music_task_embed.noise import (  # noqa: E402
    apply_noise_condition,
    build_noise_conditions,
    measured_snr_db,
    parse_snr_levels,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMO_MANIFEST_DIR = WORKSPACE_ROOT / "music_task_embed" / "manifests" / "emo" / "emotion_regression"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "noise_music_task_embed" / "embeddings"
DEFAULT_PREVIEW_ROOT = WORKSPACE_ROOT / "noise_music_task_embed" / "preview_audio"
DEFAULT_MODELS = ("mulan", "musicgen_melody_large")


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _safe_filename(text: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(text))
    return safe.strip("_") or "sample"


def _select_preview_rows(manifest: pd.DataFrame, *, n_samples: int, preferred_split: str) -> pd.DataFrame:
    candidates = manifest.copy()
    candidates["split"] = candidates["split"].astype(str)
    if preferred_split != "all" and (candidates["split"] == preferred_split).any():
        candidates = candidates[candidates["split"] == preferred_split].reset_index(drop=True)
    ordered = candidates.sort_values(["split", "fold", "sample_id"], kind="stable")
    return ordered.drop_duplicates(subset=["sample_id"], keep="first").head(int(n_samples)).reset_index(drop=True)


def _write_condition_manifest(*, output_root: Path, args: argparse.Namespace, conditions: list) -> None:
    payload = {
        "noise_seed": int(args.noise_seed),
        "include_clean": bool(args.include_clean),
        "snr_levels_db": [condition.snr_db for condition in conditions if condition.snr_db is not None],
        "conditions": [
            {
                "condition_id": condition.condition_id,
                "level_index": int(condition.level_index),
                "snr_db": condition.snr_db,
            }
            for condition in conditions
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "noise_conditions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_preview_audio(
    *,
    bundle,
    conditions,
    preview_root: Path,
    preview_sample_rate: int,
    preview_count: int,
    preferred_split: str,
    noise_seed: int,
) -> Path:
    selected = _select_preview_rows(
        bundle.manifest,
        n_samples=preview_count,
        preferred_split=preferred_split,
    )
    dataset_root = preview_root / bundle.dataset_name.lower() / bundle.task_name.lower()
    dataset_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for row_index, row in selected.iterrows():
        sample_id = str(row["sample_id"])
        source_path = str(row["audio_path"])
        split_name = str(row["split"])
        safe_id = _safe_filename(sample_id)

        clean_waveform = base_encoders.load_waveform(source_path, target_sample_rate=preview_sample_rate)
        audio_key = f"{Path(source_path).resolve()}|sr={int(preview_sample_rate)}"

        for condition in conditions:
            noisy_waveform = clean_waveform.copy()
            if not condition.is_clean:
                noisy_waveform = apply_noise_condition(
                    noisy_waveform,
                    audio_key=audio_key,
                    condition=condition,
                    seed=noise_seed,
                )

            condition_dir = dataset_root / condition.condition_id
            condition_dir.mkdir(parents=True, exist_ok=True)
            output_wav = condition_dir / f"{row_index + 1:02d}_{safe_id}.wav"
            sf.write(str(output_wav), noisy_waveform, samplerate=int(preview_sample_rate))

            records.append(
                {
                    "dataset_name": bundle.dataset_name,
                    "task_name": bundle.task_name,
                    "sample_id": sample_id,
                    "split": split_name,
                    "source_audio_path": source_path,
                    "preview_sample_rate": int(preview_sample_rate),
                    "noise_condition": condition.condition_id,
                    "noise_level_index": int(condition.level_index),
                    "target_snr_db": condition.snr_db,
                    "measured_snr_db": (
                        float("inf")
                        if condition.is_clean
                        else float(measured_snr_db(clean_waveform, noisy_waveform))
                    ),
                    "output_wav": str(output_wav),
                }
            )

    metadata_path = dataset_root / "preview_manifest.csv"
    pd.DataFrame(records).to_csv(metadata_path, index=False)
    return metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract downstream music embeddings with on-the-fly SNR-controlled noise augmentation.",
    )
    parser.add_argument(
        "--manifest-dir",
        action="append",
        default=None,
        help="Directory containing manifest.csv and task_metadata.json (can be repeated).",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        choices=SUPPORTED_MODEL_NAMES,
        default=None,
        help="Model to run (repeatable). Defaults to mulan + musicgen_melody_large.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--include-layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--partition-column", choices=["split", "fold"], default=None)
    parser.add_argument("--clap-variant", choices=["audio_embedding", "encoder"], default="audio_embedding")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--noise-snrs-db",
        type=str,
        default="20,10,0,-10",
        help="Comma-separated SNR levels in dB for noise levels 1..N.",
    )
    parser.add_argument("--include-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise-seed", type=int, default=1337)
    parser.add_argument(
        "--only-condition",
        action="append",
        default=None,
        help="Optional condition_id filter (repeatable).",
    )

    parser.add_argument("--save-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--preview-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, generate preview audio/metadata only and skip embedding extraction.",
    )
    parser.add_argument("--preview-root", type=Path, default=DEFAULT_PREVIEW_ROOT)
    parser.add_argument("--preview-per-dataset", type=int, default=2)
    parser.add_argument("--preview-sample-rate", type=int, default=32000)
    parser.add_argument("--preview-split", choices=["train", "valid", "test", "all"], default="test")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    manifest_dirs = [Path(path) for path in (args.manifest_dir or [DEFAULT_EMO_MANIFEST_DIR])]
    model_names = _dedupe_keep_order(args.model_name or list(DEFAULT_MODELS))

    snr_levels = parse_snr_levels(args.noise_snrs_db)
    conditions = build_noise_conditions(snr_levels_db=snr_levels, include_clean=bool(args.include_clean))
    if args.only_condition:
        allow = set(args.only_condition)
        conditions = [condition for condition in conditions if condition.condition_id in allow]
        if not conditions:
            raise ValueError(f"No conditions matched --only-condition={sorted(allow)}")

    bundles = [load_manifest_bundle(manifest_dir) for manifest_dir in manifest_dirs]

    _write_condition_manifest(output_root=args.output_root, args=args, conditions=conditions)

    if args.save_preview:
        for bundle in bundles:
            metadata_path = _save_preview_audio(
                bundle=bundle,
                conditions=conditions,
                preview_root=args.preview_root,
                preview_sample_rate=int(args.preview_sample_rate),
                preview_count=int(args.preview_per_dataset),
                preferred_split=args.preview_split,
                noise_seed=int(args.noise_seed),
            )
            print(f"PREVIEW {metadata_path}", flush=True)

    if args.preview_only:
        print("PREVIEW_ONLY complete. Skipping embedding extraction.", flush=True)
        return

    for model_name in model_names:
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
            representation_names = (
                extractor.representation_names(args.include_layers)
                if hasattr(extractor, "representation_names")
                else None
            )
            condition_output_root = Path(args.output_root) / condition.condition_id

            for bundle in bundles:
                manifest = bundle.manifest
                if args.limit is not None:
                    manifest = manifest.iloc[: int(args.limit)].reset_index(drop=True)

                for partition_name, partition_df in iter_partitions(
                    manifest,
                    partition_column=args.partition_column,
                ):
                    if representation_names is not None and has_complete_partition_job(
                        bundle=bundle,
                        partition_name=partition_name,
                        partition_df=partition_df,
                        model_name=model_name,
                        representation_names=representation_names,
                        output_root=condition_output_root,
                    ):
                        print(
                            f"SKIP {condition.condition_id} "
                            f"{bundle.dataset_name}/{bundle.task_name} {model_name} {partition_name}",
                            flush=True,
                        )
                        continue

                    print(
                        f"START {condition.condition_id} "
                        f"{bundle.dataset_name}/{bundle.task_name} {model_name} {partition_name} "
                        f"n={len(partition_df)}",
                        flush=True,
                    )
                    result = extractor.extract(
                        partition_df["audio_path"].astype(str).tolist(),
                        batch_size=int(args.batch_size),
                        include_layers=bool(args.include_layers),
                    )
                    output_path = save_partition_embeddings(
                        bundle=bundle,
                        partition_name=partition_name,
                        partition_df=partition_df,
                        embeddings=result.final_matrix,
                        model_name=model_name,
                        representation_name=result.final_name,
                        output_root=condition_output_root,
                        extractor_metadata=result.metadata,
                    )
                    print(output_path, flush=True)

                    for layer_name, layer_matrix in sorted(result.layer_matrices.items()):
                        layer_path = save_partition_embeddings(
                            bundle=bundle,
                            partition_name=partition_name,
                            partition_df=partition_df,
                            embeddings=layer_matrix,
                            model_name=model_name,
                            representation_name=layer_name,
                            output_root=condition_output_root,
                            extractor_metadata=result.metadata,
                        )
                        print(layer_path, flush=True)

                    print(
                        f"DONE {condition.condition_id} "
                        f"{bundle.dataset_name}/{bundle.task_name} {model_name} {partition_name}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
