"""Command-line export for deterministic Modbus MVP scenario profiles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .profiles import get_scenario, scenario_ids
from .render import render_markdown_summary, render_transcript_csv


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export HydraSim MVP Modbus scenarios")
    parser.add_argument("scenario", choices=scenario_ids())
    parser.add_argument("--format", choices=("csv", "markdown"), default="csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    scenario = get_scenario(args.scenario)
    if args.format == "csv":
        rendered = render_transcript_csv(scenario)
    else:
        rendered = render_markdown_summary(scenario)

    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
