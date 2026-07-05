# Security Policy

## Supported Scope

HydraSim is a simulation and test harness. It is not certified design authority,
commissioning evidence, safety validation, or real-plant validation.

Security reports should relate to the HydraSim codebase, HMI frontend, package
metadata, generated artifacts, local simulator behavior, or CI/release process.

Do not use HydraSim to test systems you do not own or administer. Do not submit
third-party plant details, credentials, production packet captures, or exploit
steps against real infrastructure.

The SvelteKit HMI is a simulation-only operator interface. It must not connect
directly to Modbus endpoints from the browser, embed credentials, or present
synthetic telemetry as production plant state.

## Supported Versions

| Version | Supported |
| --- | --- |
| Current `main` branch | Yes |
| Latest tagged release | Yes |
| Older tagged releases | Best effort |
| Forks or modified deployments | No |

## Reporting

Report vulnerabilities privately through GitHub Security Advisories for this
repository. If private advisory reporting is unavailable, open a minimal public
issue asking for a private contact path and do not include exploit details,
operational targets, credentials, or sensitive plant information.

Please include:

- Affected HydraSim version or commit.
- Reproduction steps against a local HydraSim instance.
- Expected impact and any known workaround.
- Whether third-party systems or sensitive data were involved.

## Disclosure

Reports are triaged for reproducibility, affected versions, and user impact.
Security fixes should land with tests or quality-gate coverage whenever
practical. Public disclosure should wait until a fix or mitigation is available,
unless the issue is already public or actively exploited.
