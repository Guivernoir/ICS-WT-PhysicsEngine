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

### Changed

- Renamed the public package and commands to HydraSim, `hydrasim`, and `hs-*`.
- Moved internal planning notes out of the public docs surface and into ignored
  private files.
- Split large legacy modules into smaller HydraSim modules.
- Modernized Python package metadata with SPDX license metadata, project URLs,
  typed package marker, and explicit tool configuration.
- Expanded CI from a partial project check to the full local quality gate.

### Removed

- Public legacy docs directory.
- Public references to old project/package naming in checked source, tests,
  tools, examples, and metadata.
