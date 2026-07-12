# Changelog

All notable user-visible changes to HydraSim should be recorded here.

HydraSim follows semantic versioning intent:

- Patch versions: compatible fixes and metadata corrections.
- Minor versions: compatible features, scenarios, commands, or evidence surfaces.
- Major versions: breaking public API, command, package, or artifact changes.

## Unreleased

### Added

- Public GitHub surface with README badges, package metadata, MIT license,
  security policy, contributor guide, changelog, and custom scenario example.
- CI quality gate for Python 3.11, 3.12, 3.13, and 3.14.
- Hard project quality policy that fails when any Python code file exceeds 500
  lines.
- Strict Mypy checking of untyped function bodies.
- Dependabot coverage for GitHub Actions and Python packaging metadata.
- Coverage-enforced unit test execution with XML artifacts in CI.
- Release workflow that builds and validates wheel/source distributions on
  `v*` tags.
- Guarded PyPI trusted-publishing job for manual release dispatches on tags.
- SvelteKit HMI scaffold with simulation dashboard, alarms, trends, scenarios,
  bounded operator setpoints, npm lockfile, and frontend unit tests.
- HMI CI gate for npm audit, ESLint, Prettier, Svelte type checking, Vitest, and
  static production builds.
- Rust runtime that exposes HTTP endpoints for snapshots, bounded setpoint
  writes, coil writes, PCS interlocks, a Rust-owned Modbus TCP server, and a
  stdio Python simulation-worker boundary.
- Rust CI gate for formatting, locked dependency checks, tests, and clippy with
  warnings denied.

### Changed

- Renamed the public package and commands to HydraSim, `hydrasim`, and `hs-*`.
- Moved internal planning notes out of the public docs surface and into ignored
  private files.
- Split large legacy modules into smaller HydraSim modules.
- Modernized Python package metadata with SPDX license metadata, project URLs,
  typed package marker, and explicit tool configuration.
- Expanded CI from a partial project check to the full local quality gate,
  including package coverage and distribution-build readiness.
- Expanded the 500-line quality policy from Python-only checks to Python,
  Rust, Svelte, TypeScript, JavaScript, and CSS code files.
- Documented and enforced the Svelte/Rust/Python separation of concerns for the
  simulation-only HMI.

### Removed

- Public legacy docs directory.
- Public references to old project/package naming in checked source, tests,
  tools, examples, and metadata.
