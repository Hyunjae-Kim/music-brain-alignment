from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from .storage import ManifestBundle, write_manifest_bundle


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "CIM-bench"
DEFAULT_MANIFEST_ROOT = PACKAGE_ROOT / "manifests"

NSYNTH_PATTERN = re.compile(
    r"^(?P<family>[^_]+)_(?P<source>[^_]+)_(?P<pitch>\d+)-(?P<velocity>\d+)-(?P<quality>\d+)$"
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _audio_rows(audio_root: Path, pattern: str) -> list[Path]:
    return sorted(audio_root.rglob(pattern))


def _normalize_split(split: str) -> str:
    split = split.strip().lower()
    alias_map = {
        "dev": "valid",
        "validation": "valid",
        "val": "valid",
        "train+": "train",
        "valid+": "valid",
        "validation+": "valid",
        "test+": "test",
    }
    return alias_map.get(split, split)


def _with_audio_stem_suffix(audio_relpath: str, stem_suffix: str) -> str:
    if not stem_suffix:
        return audio_relpath
    relpath = Path(audio_relpath)
    return relpath.with_name(f"{relpath.stem}{stem_suffix}{relpath.suffix}").as_posix()


def _load_cmi_split_map(jsonl_dir: Path, file_names: list[str]) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for file_name in file_names:
        path = jsonl_dir / file_name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                audio_id = str(payload["audioid"]).strip()
                splits = payload.get("split", [])
                split_name = _normalize_split(str(splits[0])) if splits else _normalize_split(path.stem)
                split_map[audio_id] = split_name
    return split_map


def build_vocalset_manifest(
    *,
    audio_root: str | Path = DEFAULT_DATA_ROOT / "VocalSet" / "audio",
    output_root: str | Path = DEFAULT_MANIFEST_ROOT,
    split_jsonl_dir: str | Path | None = None,
) -> Path:
    audio_root = Path(audio_root)
    if not audio_root.exists():
        raise FileNotFoundError(f"VocalSet audio root not found: {audio_root}")

    split_map: dict[str, str] = {}
    if split_jsonl_dir is not None:
        split_map = _load_cmi_split_map(
            Path(split_jsonl_dir),
            ["VocalSet_1train.jsonl", "VocalSet_valid.jsonl", "VocalSet_test.jsonl"],
        )

    rows: list[dict[str, Any]] = []
    for audio_path in _audio_rows(audio_root, "*.wav"):
        relpath = audio_path.relative_to(audio_root).as_posix()
        technique = audio_path.parent.name
        rows.append(
            {
                "sample_id": audio_path.stem,
                "audio_relpath": relpath,
                "audio_path": str(audio_path.resolve()),
                "split": split_map.get(relpath, "unspecified"),
                "fold": "",
                "target_json": _json_dumps(technique),
                "label_text": technique,
                "technique": technique,
            }
        )

    manifest = pd.DataFrame(rows)
    bundle = ManifestBundle(
        dataset_name="VocalSet",
        task_name="singing_technique",
        target_type="single_label_classification",
        manifest=manifest,
        label_vocabulary=tuple(sorted(manifest["label_text"].unique().tolist())),
        metadata={
            "audio_root": str(audio_root.resolve()),
            "split_source": str(split_jsonl_dir) if split_jsonl_dir is not None else None,
        },
    )
    return write_manifest_bundle(bundle, Path(output_root))


def build_nsynth_manifests(
    *,
    nsynth_root: str | Path = DEFAULT_DATA_ROOT / "NSynth",
    output_root: str | Path = DEFAULT_MANIFEST_ROOT,
) -> list[Path]:
    nsynth_root = Path(nsynth_root)
    if not nsynth_root.exists():
        raise FileNotFoundError(f"NSynth root not found: {nsynth_root}")

    split_dirs = [
        ("train", nsynth_root / "nsynth-train" / "audio"),
        ("valid", nsynth_root / "nsynth-valid" / "audio"),
        ("test", nsynth_root / "nsynth-test" / "audio"),
    ]

    rows: list[dict[str, Any]] = []
    for split_name, audio_dir in split_dirs:
        if not audio_dir.exists():
            continue
        for audio_path in _audio_rows(audio_dir, "*.wav"):
            match = NSYNTH_PATTERN.match(audio_path.stem)
            if match is None:
                raise ValueError(f"Could not parse NSynth file name: {audio_path.name}")
            payload = match.groupdict()
            rows.append(
                {
                    "sample_id": audio_path.stem,
                    "audio_relpath": audio_path.relative_to(nsynth_root).as_posix(),
                    "audio_path": str(audio_path.resolve()),
                    "split": split_name,
                    "fold": "",
                    "instrument_family": payload["family"],
                    "instrument_source": payload["source"],
                    "pitch_midi": int(payload["pitch"]),
                    "velocity": int(payload["velocity"]),
                    "quality": int(payload["quality"]),
                }
            )

    if not rows:
        raise FileNotFoundError(f"No NSynth wav files were found under {nsynth_root}")

    base_df = pd.DataFrame(rows).sort_values(["split", "sample_id"], kind="stable").reset_index(drop=True)

    instrument_df = base_df.copy()
    instrument_df["target_json"] = instrument_df["instrument_family"].map(_json_dumps)
    instrument_df["label_text"] = instrument_df["instrument_family"]

    pitch_df = base_df.copy()
    pitch_df["target_json"] = pitch_df["pitch_midi"].map(_json_dumps)
    pitch_df["label_text"] = pitch_df["pitch_midi"].astype(str)

    output_paths = [
        write_manifest_bundle(
            ManifestBundle(
                dataset_name="NSynth",
                task_name="instrument_classification",
                target_type="single_label_classification",
                manifest=instrument_df,
                label_vocabulary=tuple(sorted(instrument_df["instrument_family"].unique().tolist())),
                metadata={"audio_root": str(nsynth_root.resolve())},
            ),
            Path(output_root),
        ),
        write_manifest_bundle(
            ManifestBundle(
                dataset_name="NSynth",
                task_name="pitch_estimation",
                target_type="single_label_classification",
                manifest=pitch_df,
                label_vocabulary=tuple(str(int(value)) for value in sorted(pitch_df["pitch_midi"].unique().tolist())),
                metadata={"audio_root": str(nsynth_root.resolve())},
            ),
            Path(output_root),
        ),
    ]
    return output_paths


def build_gtzan_manifest(
    *,
    audio_root: str | Path = DEFAULT_DATA_ROOT / "GTZAN" / "Data" / "genres_original",
    output_root: str | Path = DEFAULT_MANIFEST_ROOT,
    num_folds: int = 10,
) -> Path:
    audio_root = Path(audio_root)
    if not audio_root.exists():
        raise FileNotFoundError(f"GTZAN audio root not found: {audio_root}")
    if num_folds < 2:
        raise ValueError("GTZAN num_folds must be at least 2.")

    label_to_paths: dict[str, list[Path]] = defaultdict(list)
    for audio_path in _audio_rows(audio_root, "*.wav"):
        label_to_paths[audio_path.parent.name].append(audio_path)

    rows: list[dict[str, Any]] = []
    for label, paths in sorted(label_to_paths.items()):
        for index, audio_path in enumerate(sorted(paths)):
            rows.append(
                {
                    "sample_id": audio_path.stem,
                    "audio_relpath": audio_path.relative_to(audio_root).as_posix(),
                    "audio_path": str(audio_path.resolve()),
                    "split": "",
                    "fold": f"{index % num_folds:02d}",
                    "target_json": _json_dumps(label),
                    "label_text": label,
                    "genre": label,
                }
            )

    manifest = pd.DataFrame(rows).sort_values(["fold", "label_text", "sample_id"], kind="stable")
    bundle = ManifestBundle(
        dataset_name="GTZAN",
        task_name="genre_classification",
        target_type="single_label_classification",
        manifest=manifest.reset_index(drop=True),
        label_vocabulary=tuple(sorted(manifest["label_text"].unique().tolist())),
        metadata={
            "audio_root": str(audio_root.resolve()),
            "split_strategy": "deterministic_stratified_kfold",
            "num_folds": int(num_folds),
        },
    )
    return write_manifest_bundle(bundle, Path(output_root))


def build_cmi_jsonl_manifest(
    *,
    dataset_name: str,
    task_name: str,
    audio_root: str | Path,
    jsonl_paths: list[str | Path],
    output_root: str | Path = DEFAULT_MANIFEST_ROOT,
    multi_label_separator: str = ", ",
    audio_stem_suffix: str = "",
) -> Path:
    audio_root = Path(audio_root)
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")
    if not jsonl_paths:
        raise ValueError("At least one jsonl path is required.")

    rows: list[dict[str, Any]] = []
    observed_labels: set[str] = set()
    target_type = "single_label_classification"

    for jsonl_path in [Path(path) for path in jsonl_paths]:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                audio_relpath = _with_audio_stem_suffix(str(payload["audioid"]).strip(), audio_stem_suffix)
                output_value = payload.get("output")
                split_values = payload.get("split", [])
                split_name = _normalize_split(str(split_values[0])) if split_values else _normalize_split(jsonl_path.stem)

                labels: list[str]
                if isinstance(output_value, (int, float)):
                    labels = [str(output_value)]
                    serialized_target = _json_dumps(float(output_value))
                    target_type = "regression"
                    label_text = str(output_value)
                else:
                    output_text = str(output_value).strip()
                    if multi_label_separator in output_text:
                        labels = [item.strip() for item in output_text.split(multi_label_separator) if item.strip()]
                        serialized_target = _json_dumps(labels)
                        target_type = "multi_label_classification"
                        label_text = multi_label_separator.join(labels)
                    else:
                        labels = [output_text]
                        serialized_target = _json_dumps(output_text)
                        label_text = output_text

                observed_labels.update(label for label in labels if label)
                audio_path = audio_root / audio_relpath
                if not audio_path.exists():
                    raise FileNotFoundError(f"Missing audio file referenced by manifest: {audio_path}")
                rows.append(
                    {
                        "sample_id": str(payload.get("uuid") or Path(audio_relpath).stem),
                        "audio_relpath": audio_relpath,
                        "audio_path": str(audio_path.resolve()),
                        "split": split_name,
                        "fold": "",
                        "target_json": serialized_target,
                        "label_text": label_text,
                    }
                )

    manifest = pd.DataFrame(rows).sort_values(["split", "sample_id"], kind="stable").reset_index(drop=True)
    bundle = ManifestBundle(
        dataset_name=dataset_name,
        task_name=task_name,
        target_type=target_type,
        manifest=manifest,
        label_vocabulary=tuple(sorted(observed_labels)) if target_type != "regression" else (),
        metadata={
            "audio_root": str(audio_root.resolve()),
            "jsonl_paths": [str(Path(path).resolve()) for path in jsonl_paths],
            "multi_label_separator": multi_label_separator,
            "audio_stem_suffix": audio_stem_suffix,
        },
    )
    return write_manifest_bundle(bundle, Path(output_root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build local manifests for downstream MIR embedding extraction.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    vocalset = subparsers.add_parser("vocalset")
    vocalset.add_argument("--audio-root", type=str, default=str(DEFAULT_DATA_ROOT / "VocalSet" / "audio"))
    vocalset.add_argument("--output-root", type=str, default=str(DEFAULT_MANIFEST_ROOT))
    vocalset.add_argument("--split-jsonl-dir", type=str, default=None)

    nsynth = subparsers.add_parser("nsynth")
    nsynth.add_argument("--nsynth-root", type=str, default=str(DEFAULT_DATA_ROOT / "NSynth"))
    nsynth.add_argument("--output-root", type=str, default=str(DEFAULT_MANIFEST_ROOT))

    gtzan = subparsers.add_parser("gtzan")
    gtzan.add_argument("--audio-root", type=str, default=str(DEFAULT_DATA_ROOT / "GTZAN" / "Data" / "genres_original"))
    gtzan.add_argument("--output-root", type=str, default=str(DEFAULT_MANIFEST_ROOT))
    gtzan.add_argument("--num-folds", type=int, default=10)

    cmi_jsonl = subparsers.add_parser("cmi-jsonl")
    cmi_jsonl.add_argument("--dataset-name", type=str, required=True)
    cmi_jsonl.add_argument("--task-name", type=str, required=True)
    cmi_jsonl.add_argument("--audio-root", type=str, required=True)
    cmi_jsonl.add_argument("--jsonl", type=str, action="append", required=True)
    cmi_jsonl.add_argument("--output-root", type=str, default=str(DEFAULT_MANIFEST_ROOT))
    cmi_jsonl.add_argument("--multi-label-separator", type=str, default=", ")
    cmi_jsonl.add_argument("--audio-stem-suffix", type=str, default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "vocalset":
        path = build_vocalset_manifest(
            audio_root=args.audio_root,
            output_root=args.output_root,
            split_jsonl_dir=args.split_jsonl_dir,
        )
        print(path)
        return
    if args.command == "nsynth":
        for path in build_nsynth_manifests(nsynth_root=args.nsynth_root, output_root=args.output_root):
            print(path)
        return
    if args.command == "gtzan":
        path = build_gtzan_manifest(
            audio_root=args.audio_root,
            output_root=args.output_root,
            num_folds=args.num_folds,
        )
        print(path)
        return
    if args.command == "cmi-jsonl":
        path = build_cmi_jsonl_manifest(
            dataset_name=args.dataset_name,
            task_name=args.task_name,
            audio_root=args.audio_root,
            jsonl_paths=args.jsonl,
            output_root=args.output_root,
            multi_label_separator=args.multi_label_separator,
            audio_stem_suffix=args.audio_stem_suffix,
        )
        print(path)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
