"""Command-line interface for staged HydraSim simulation profiles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .bundle import export_hs_bundle
from .catalog import AREA_CHOICES, CONTROL_SYSTEMS, MEDIA, STAGES, TOPOLOGIES
from .orchestration import (
    build_live_orchestration_plan,
    render_live_plan,
    run_live_orchestration,
)
from .pcap import render_hs_pcap_bytes
from .profiles import get_profile, profile_ids
from .release import (
    build_reference_water_plant_cfd_release_candidate,
    render_release_candidate_json,
    render_release_candidate_markdown,
)
from .render import render_summary_markdown, render_transcript_csv
from .runtime import build_runtime_artifact
from .scenarios import scenario_ids
from .validator import validate_profile


def _write_text_or_print(text: str, output: Path | None) -> None:
    if output is None:
        print(text, end="")
    else:
        output.write_text(text, encoding="utf-8")


def _write_bytes(data: bytes, output: Path | None) -> None:
    if output is None:
        raise ValueError("binary pcap output requires --output")
    output.write_bytes(data)


def _list_profiles() -> int:
    for profile_id in profile_ids():
        profile = get_profile(profile_id)
        print(f"{profile.profile_id}\t{profile.preset}\t{profile.name}")
    return 0


def _validate_profile(profile_id: str) -> int:
    profile = get_profile(profile_id)
    errors = validate_profile(profile)
    if errors:
        for error in errors:
            print(f"error: {error}")
        return 1
    print(f"valid: {profile.profile_id}")
    return 0


def _run(args) -> int:
    artifact = build_runtime_artifact(
        args.profile,
        args.scenario,
        args.area,
        args.stage,
        control_system=args.control_system,
        topology=args.topology,
        media=args.media,
    )
    if args.format == "csv":
        _write_text_or_print(render_transcript_csv(artifact), args.output)
    elif args.format == "markdown":
        _write_text_or_print(render_summary_markdown(artifact), args.output)
    else:
        _write_bytes(render_hs_pcap_bytes(artifact), args.output)
    return 0


def _launch_live(args) -> int:
    plan = build_live_orchestration_plan(
        args.profile,
        args.scenario,
        args.area,
        args.stage,
        host=args.host,
        base_port=args.base_port,
        duration_seconds=args.duration,
        startup_delay_seconds=args.startup_delay,
        log_dir=args.log_dir,
    )
    if args.dry_run:
        _write_text_or_print(render_live_plan(plan), args.output)
        return 0
    results = run_live_orchestration(plan)
    for result in results:
        print(
            f"{result.node_id}\tport={result.port}\t"
            f"return_code={result.return_code}\t{result.status}"
        )
    return 0


def _release_candidate(args) -> int:
    release = build_reference_water_plant_cfd_release_candidate(
        scenario_id=args.scenario,
        selected_area=args.area,
    )
    if args.format == "json":
        _write_text_or_print(render_release_candidate_json(release), args.output)
    else:
        _write_text_or_print(render_release_candidate_markdown(release), args.output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run HydraSim plant profiles")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-profiles", help="list built-in plant profiles")

    validate = sub.add_parser("validate-profile", help="validate a built-in profile")
    validate.add_argument("profile", choices=profile_ids())

    run = sub.add_parser("run", help="render a staged runtime artifact")
    run.add_argument("profile", choices=profile_ids())
    run.add_argument("--scenario", choices=scenario_ids(), default="HS-WTP-002")
    run.add_argument("--area", choices=AREA_CHOICES, default="disinfection")
    run.add_argument("--stage", choices=STAGES, default="offline-export")
    run.add_argument(
        "--format", choices=("markdown", "csv", "pcap"), default="markdown"
    )
    run.add_argument("--output", type=Path)
    run.add_argument("--control-system", choices=CONTROL_SYSTEMS, default=None)
    run.add_argument("--media", choices=MEDIA, default=None)
    run.add_argument("--topology", choices=TOPOLOGIES, default=None)

    bundle = sub.add_parser(
        "export-bundle", help="export deterministic HydraSim lab bundle"
    )
    bundle.add_argument("profile", choices=profile_ids())
    bundle.add_argument("scenario", choices=scenario_ids())
    bundle.add_argument("output_dir", type=Path)
    bundle.add_argument("--area", choices=AREA_CHOICES, default="all")
    bundle.add_argument("--stage", choices=STAGES, default="offline-export")
    bundle.add_argument("--control-system", choices=CONTROL_SYSTEMS, default=None)
    bundle.add_argument("--media", choices=MEDIA, default=None)
    bundle.add_argument("--topology", choices=TOPOLOGIES, default=None)

    live = sub.add_parser(
        "launch-live",
        help="launch selected-area synthetic Modbus endpoint processes",
    )
    live.add_argument("profile", choices=profile_ids())
    live.add_argument("--scenario", choices=scenario_ids(), default="HS-WTP-002")
    live.add_argument(
        "--area",
        choices=tuple(item for item in AREA_CHOICES if item != "all"),
        default="disinfection",
    )
    live.add_argument(
        "--stage",
        choices=tuple(item for item in STAGES if item != "offline-export"),
        default="full-cell",
    )
    live.add_argument("--host", default="127.0.0.1")
    live.add_argument("--base-port", type=int, default=5520)
    live.add_argument("--duration", type=float, default=30.0)
    live.add_argument("--startup-delay", type=float, default=1.0)
    live.add_argument("--log-dir", type=Path)
    live.add_argument("--dry-run", action="store_true")
    live.add_argument("--output", type=Path)

    release = sub.add_parser(
        "release-candidate",
        help="render the HS-35 Reference Water Plant CFD release-candidate checklist",
    )
    release.add_argument("--scenario", choices=scenario_ids(), default="HS-WTP-002")
    release.add_argument(
        "--area",
        choices=tuple(item for item in AREA_CHOICES if item != "all"),
        default="disinfection",
    )
    release.add_argument("--format", choices=("markdown", "json"), default="markdown")
    release.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "list-profiles":
            return _list_profiles()
        if args.command == "validate-profile":
            return _validate_profile(args.profile)
        if args.command == "run":
            return _run(args)
        if args.command == "export-bundle":
            export_hs_bundle(
                args.profile,
                args.scenario,
                args.area,
                args.stage,
                args.output_dir,
                control_system=args.control_system,
                topology=args.topology,
                media=args.media,
            )
            print(f"bundle written: {args.output_dir}")
            return 0
        if args.command == "launch-live":
            return _launch_live(args)
        if args.command == "release-candidate":
            return _release_candidate(args)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
