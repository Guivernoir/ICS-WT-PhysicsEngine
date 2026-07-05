# Contributing

HydraSim accepts changes that improve the simulator, tests, examples, package
metadata, CI policy, or public README surface without exposing private planning
notes.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,modbus]"
```

Run the simulator without Modbus:

```bash
python -m hydrasim --no-modbus --duration 10 --dt 1
```

## Quality Gate

Run the full gate before proposing changes:

```bash
python -m pip check
python -m black --check src tests tools
python -m ruff check src tools tests
python -m mypy src/hydrasim
python -m compileall -q src tests tools
python tools/check_project_quality.py
python -m unittest discover -s tests -v
```

The project quality checker enforces a hard 500-line limit for every Python code
file. Split modules before crossing that limit.

## Code Standards

- Keep public package names under `hydrasim` and command names under `hs-*`.
- Prefer small modules with explicit boundaries over large legacy catch-all
  files.
- Keep tests close to the behavior being changed.
- Use deterministic exports for transcripts, bundles, PCAPs, and evidence
  surfaces.
- Keep CFD/digital-twin claims explicitly synthetic unless real calibration and
  external validation evidence exists.

## Documentation Rules

Public documentation belongs in `README.md`, `CONTRIBUTING.md`,
`CHANGELOG.md`, `SECURITY.md`, examples, package metadata, and concise inline
module docs. Internal planning notes belong under ignored `.private/docs`.

Do not add public attack playbooks, third-party plant details, credentials,
production captures, or operational guidance for systems you do not own.

## Pull Request Checklist

- The full local quality gate passes.
- Public README commands still match the implemented CLI.
- New or changed files use HydraSim, HS, or `hydrasim` naming.
- Security-sensitive reports follow [SECURITY.md](SECURITY.md).
- The changelog is updated for user-visible behavior, metadata, CI, or policy
  changes.
