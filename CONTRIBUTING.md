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

Install the HMI toolchain:

```bash
cd hmi
npm ci
```

The Rust runtime uses stable Rust:

```bash
cd runtime
cargo check --locked
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
python -m coverage run -m unittest discover -s tests -v
python -m coverage report
python -m coverage xml
cd hmi
npm audit
npm run lint
npm run check
npm run test
npm run build
cd ../runtime
cargo fmt --check
cargo check --locked
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

The project quality checker enforces a hard 500-line limit for Python, Rust,
Svelte, TypeScript, JavaScript, and CSS code files. Coverage reporting enforces
the configured project floor in `pyproject.toml`. Split modules or components
before crossing the line limit, and add tests before reducing coverage.

## Code Standards

- Keep public package names under `hydrasim` and command names under `hs-*`.
- Keep HMI code under `hmi`, Rust runtime code under `runtime`, and Python
  simulation code under `src/hydrasim`.
- Keep browser code behind the HMI API boundary; process, PCS, and protocol
  logic belongs in Rust or Python, not in Svelte.
- Keep Python focused on physical simulation. New network, media, PCS, or HMI
  backend behavior belongs in Rust.
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
- HMI changes pass `npm audit`, `npm run lint`, `npm run check`,
  `npm run test`, and `npm run build`.
- Rust runtime changes pass rustfmt, locked cargo check/test, and clippy with
  warnings denied.
- Security-sensitive reports follow [SECURITY.md](SECURITY.md).
- The changelog is updated for user-visible behavior, metadata, CI, or policy
  changes.

## Release Checklist

- The quality workflow is green on `main`.
- `CHANGELOG.md` has a release entry for the version.
- `pyproject.toml` contains the intended version.
- A signed or otherwise intentional tag named `vX.Y.Z` is pushed.
- The release workflow builds and checks the distribution artifacts.
- PyPI publishing is triggered only after the `pypi` trusted publisher
  environment is configured and the manual workflow input is enabled on a tag.
