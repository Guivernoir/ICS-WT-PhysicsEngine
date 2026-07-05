"""Run built-in or custom HydraSim scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .loader import load_scenario_json
from .live import run_live_scenario, run_live_scenario_targets
from .pcap import render_pcap_bytes
from .profiles import get_scenario
from .render import render_markdown_summary, render_transcript_csv


def _load_scenario(identifier: str, custom_json: Path | None):
    if custom_json is not None:
        return load_scenario_json(custom_json)
    return get_scenario(identifier)


def _write_or_print(text: str, output: Path | None) -> None:
    if output is None:
        print(text, end="")
    else:
        output.write_text(text, encoding="utf-8")


def _write_bytes(data: bytes, output: Path | None) -> None:
    if output is None:
        raise ValueError("binary pcap output requires --output")
    output.write_bytes(data)


def _parse_target(value: str) -> tuple[str, tuple[str, int]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("target must be server_id=host:port")
    server_id, endpoint = value.split("=", 1)
    if ":" not in endpoint:
        raise argparse.ArgumentTypeError("target must be server_id=host:port")
    host, port_text = endpoint.rsplit(":", 1)
    try:
        port = int(port_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target port must be an integer") from exc
    if not server_id or not host or not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("target must be server_id=host:port")
    return server_id, (host, port)


def _run_live(args) -> str:
    scenario = _load_scenario(args.scenario, args.custom_json)
    if args.target:
        run_live_scenario_targets(
            scenario.scenario_id,
            dict(args.target),
            unit_id=args.unit_id,
            time_scale=args.time_scale,
            custom_json=args.custom_json,
        )
        return "live scenario completed\n"

    server_ids = {transaction.server_id for transaction in scenario.transactions}
    if len(server_ids) > 1:
        needed = ", ".join(sorted(server_ids))
        raise RuntimeError(
            "multi-endpoint live scenarios require --target mappings for: " + needed
        )
    run_live_scenario(
        scenario.scenario_id,
        args.host,
        args.port,
        unit_id=args.unit_id,
        time_scale=args.time_scale,
        custom_json=args.custom_json,
    )
    return "live scenario completed\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or export HydraSim scenarios")
    parser.add_argument("scenario", help="built-in scenario ID, alias, or custom label")
    parser.add_argument("--custom-json", type=Path, help="load scenario from JSON")
    parser.add_argument(
        "--mode",
        choices=("transcript", "live"),
        default="transcript",
        help="export a transcript or drive a live Modbus endpoint",
    )
    parser.add_argument("--format", choices=("csv", "markdown", "pcap"), default="csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5020)
    parser.add_argument(
        "--target",
        action="append",
        type=_parse_target,
        help="live target mapping as server_id=host:port; repeat for multi-node scenarios",
    )
    parser.add_argument("--unit-id", type=int, default=1)
    parser.add_argument(
        "--time-scale",
        type=float,
        default=0.0,
        help="0 runs immediately; positive values replay relative scenario timing",
    )
    args = parser.parse_args(argv)

    if args.mode == "live":
        try:
            rendered = _run_live(args)
        except RuntimeError as exc:
            parser.error(str(exc))
    else:
        scenario = _load_scenario(args.scenario, args.custom_json)
        if args.format == "pcap":
            try:
                _write_bytes(render_pcap_bytes(scenario), args.output)
            except ValueError as exc:
                parser.error(str(exc))
            return 0
        elif args.format == "markdown":
            rendered = render_markdown_summary(scenario)
        else:
            rendered = render_transcript_csv(scenario)

    _write_or_print(rendered, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
