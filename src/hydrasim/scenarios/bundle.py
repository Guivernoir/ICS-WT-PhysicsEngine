"""Deterministic lab bundle export for HydraSim scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .loader import load_scenario_json
from .pcap import render_pcap_bytes
from .profiles import get_scenario
from .render import render_markdown_summary, render_transcript_csv


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(identifier: str, custom_json: Path | None):
    if custom_json is not None:
        return load_scenario_json(custom_json)
    return get_scenario(identifier)


def _capture_notes(scenario_id: str) -> str:
    return (
        f"# HydraSim Lab Bundle Capture Notes: {scenario_id}\n\n"
        "This bundle is deterministic synthetic simulator output.\n\n"
        "## Contents\n\n"
        "- `transcript.csv`: scenario transaction truth.\n"
        "- `summary.md`: human-readable scenario profile.\n"
        "- `scenario.pcap`: transcript-derived classic Ethernet PCAP.\n"
        "- `manifest.json`: source and generation metadata.\n"
        "- `checksums.sha256`: SHA-256 hashes for generated artifacts.\n\n"
        "## Boundaries\n\n"
        "- This is not an operational capture.\n"
        "- This is not customer data.\n"
        "- This does not prove field safety or operational behavior.\n"
        "- Passive observers should not transmit traffic into the scenario.\n"
    )


def export_lab_bundle(
    scenario_id: str,
    output_dir: str | Path,
    custom_json: str | Path | None = None,
) -> tuple[Path, ...]:
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"bundle output already exists: {target}")
    target.mkdir(parents=True)

    custom_path = Path(custom_json) if custom_json is not None else None
    scenario = _load(scenario_id, custom_path)
    transcript = render_transcript_csv(scenario).encode("utf-8")
    summary = render_markdown_summary(scenario).encode("utf-8")
    pcap = render_pcap_bytes(scenario)
    notes = _capture_notes(scenario.scenario_id).encode("utf-8")

    artifacts = {
        "transcript.csv": transcript,
        "summary.md": summary,
        "scenario.pcap": pcap,
        "capture-notes.md": notes,
    }
    manifest = {
        "scenario_id": scenario.scenario_id,
        "name": scenario.name,
        "source_class": "SyntheticScenario",
        "custom_json": str(custom_path) if custom_path is not None else None,
        "transaction_count": len(scenario.transactions),
        "node_count": len(scenario.nodes),
        "field_point_count": len(scenario.field_points),
        "pcap_kind": "transcript-derived classic Ethernet PCAP",
        "limitations": list(scenario.limitations),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    artifacts["manifest.json"] = manifest_bytes

    checksum_lines = []
    written: list[Path] = []
    for name, data in artifacts.items():
        path = target / name
        path.write_bytes(data)
        written.append(path)
        checksum_lines.append(f"{_sha256(data)}  {name}")
    checksum_data = ("\n".join(checksum_lines) + "\n").encode("utf-8")
    checksum_path = target / "checksums.sha256"
    checksum_path.write_bytes(checksum_data)
    written.append(checksum_path)
    return tuple(written)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a HydraSim lab bundle")
    parser.add_argument("scenario", help="built-in scenario ID, alias, or custom label")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--custom-json", type=Path)
    args = parser.parse_args(argv)

    try:
        export_lab_bundle(args.scenario, args.output_dir, custom_json=args.custom_json)
    except FileExistsError as exc:
        parser.error(str(exc))
    print(f"bundle written: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
