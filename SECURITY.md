# Security

flashback is an episodic-memory backend that stores conversation data, NLP
embeddings, and (depending on deployment) personally identifying information
from the chats it remembers. The defaults assume single-operator, single-host
deployment behind a trusted network boundary (tailnet, localhost, internal LAN).
Operators who plan to expose flashback more broadly should read this document
and the design notes under `docs/TENANCY.md` first.

## Threat model snapshot

**What flashback defends against today:**

- **Default network exposure** — server binds to `127.0.0.1`. Public exposure
  is opt-in via reverse proxy or `docker-compose` port-publish change.
- **Crate supply chain** — `cargo-deny` enforces a license allowlist + a
  registry allowlist (`crates.io` only) on every PR; unknown git sources
  are rejected.
- **Known CVE deps** — `cargo audit` (RustSec advisory DB) runs on every
  PR and weekly on `main`.
- **Static analysis** — `cargo clippy -D warnings` plus `semgrep --config auto`
  on every PR; warnings are errors.
- **Secret scanning** — both `trufflehog --only-verified` and `gitleaks`
  run on the full git history every PR.
- **Workflow correctness** — `actionlint` lints every workflow file before
  merge.

**What flashback does NOT defend against:**

- **Operator-machine compromise.** If the host running flashback is rooted,
  all stored memories + embeddings + master key are readable.
- **Model-provider compromise.** When remote LLMs are configured (e.g. via
  `PROVIDER_REMOTE_API_BASE`), the provider sees the full extraction /
  distillation prompt and response. Don't send memory content you wouldn't
  send to that provider directly.
- **Prompt injection inside memory content.** If you remember adversarial
  text and later distill it with an LLM, the LLM can be steered by it.
  Treat distilled summaries as data, not as instructions.
- **At-rest encryption of the DB.** Memories sit in SQLite/Postgres files
  unencrypted by default — rely on disk-level encryption + OS access
  controls.
- **Multi-tenant isolation.** `docs/TENANCY.md` is exploratory; multi-user
  visibility/permissions are not yet implemented.

## Security tooling (CI + repo hygiene)

What runs on every push / PR:

- **CI** (`.github/workflows/ci.yml`) — `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test`, release build, all on stable Rust.
- **Sonar** (`.github/workflows/build.yml`) — SonarCloud scan over `crates/**` with `cargo llvm-cov` lcov coverage.
- **Security** (`.github/workflows/security.yml`) — runs all of:
  - `actionlint` workflow lint
  - `trufflehog --only-verified` against full git history
  - `gitleaks detect` against full git history (SARIF uploaded as artifact)
  - `rustsec/audit-check` (cargo-audit) on Cargo.lock
  - `cargo deny check` (licenses + sources + advisories + bans) per `deny.toml`
  - `semgrep --config auto --error` (pinned image tag)
- **SBOM** (`.github/workflows/sbom.yml`) — on every published release, generates CycloneDX + SPDX SBOMs via `anchore/sbom-action` (syft) and attaches them to the release.
- **Dependabot** — weekly grouped cargo + github-actions + docker update PRs.

What's enabled at the GitHub repo level:

- Dependabot vulnerability alerts: **on**
- Dependabot automated security updates: **on**
- All GitHub Actions pinned to commit SHAs with trailing `# vX.Y.Z` comment.
- `CODEOWNERS` routes review on security-sensitive paths (`/.github/`, `/SECURITY.md`, `/deny.toml`, `/migrations/`, `/crates/server/src/auth/`).

Local pre-push convenience:

- `./scripts/scan-secrets.sh` — runs `gitleaks` against the working tree (install gitleaks first). Optional pre-commit hook: `ln -s ../../scripts/scan-secrets.sh .git/hooks/pre-commit`.

## Deferred until repo goes public

These rely on GitHub Advanced Security (GHAS) or only meaningfully apply to
public repos; intentionally not configured today. Re-evaluate at the
public-flip:

- **CodeQL code scanning** — free on public, GHAS-only on private.
- **OpenSSF Scorecard** — publishes results publicly; not useful on private.
- **`actions/dependency-review-action`** on PRs — requires Dependency Graph API, GHAS-only on private.
- **Secret scanning + push protection** — GitHub-native; GHAS-only on private.
- **Branch protection rules** on `main` — `gh api .../branches/main/protection` requires Pro on private repos. Until then treat as social-contract enforced.
- **Private vulnerability reporting** — org-level toggle; not configured.

## Reporting

Found something? Email `security@horizon-digital.dev`. No bounty program
today; just a thanks and credit if you want it. Please don't open a public
issue for vulnerabilities.
