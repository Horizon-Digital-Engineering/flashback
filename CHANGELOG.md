# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and Flashback adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-23

Initial public release. Flashback is a self-contained Rust microservice that
gives any LLM dynamic, episodic memory: a four-tier hierarchy with append-with-
supersede history, real-time within-conversation ingest, and hybrid retrieval
over a temporal graph backed by `pgvector`.

### What works in 0.1.0

**Memory model**
- Four-tier hierarchy: core (always-injected) / working (TTL'd) / episodic /
  semantic.
- Append-with-supersede — old memories never deleted; superseded rows stay in
  the lineage chain for `/lineage` queries.
- Default retrieval returns the terminal node; lineage traversal exposes the
  full evolution.

**Ingest**
- `POST /memory/ingest` accepts raw text or structured user/assistant turn
  pairs.
- Pluggable `AiProvider` for extraction:
  - `heuristic` — rule-based, in-process, zero network. Default.
  - `remote` — any OpenAI-compatible HTTP endpoint (OpenRouter, Anthropic,
    OpenAI, local Ollama, etc.).
  - `embedded` — LLM running in-process via `mistralrs` (air-gapped / single-
    box deploys).
- Per-role model tiering: separate extract (fast, ~2s budget) and distill
  (background, minutes-OK) models — see `docs/MODEL-TIERING.md`.

**Retrieval**
- `POST /memory/search` — hybrid: vector cosine + BM25 keyword + recency +
  project-match + entity overlap.
- `answer` mode (relevance-weighted) and `manager` mode (situational-
  awareness-weighted).
- `POST /context/assemble` — structured 5-layer prompt: procedural / active-
  project / retrieved-memories / document-chunks / recent-conversation.

**State objects**
- Typed mutable state (`/state/{kind}/{key}`) with op-based patches and
  full supersede history. `todo_list` is the first shipped kind.

**Auth**
- Bearer-token, sha256-hashed at rest, scoped per user.
- Plaintext shown once at mint; `flashback token mint --user=<user>
  --name=<label>`.
- `--dev` / `FLASHBACK_DEV_MODE=1` bypasses auth for local development;
  banner-warned on every startup.

**MCP transport**
- Streamable-HTTP MCP server on `:8082/mcp`, wraps the REST API as typed
  tools.
- Wire into Claude Desktop / Cursor / Claude Code by pasting the URL + bearer
  into the client config — see README.

**Storage**
- Postgres + `pgvector` 0.4.2 (with SQLx 0.9 support).
- `sqlx::migrate!` baked-in migrations; `AUTO_MIGRATE=1` runs them on first
  boot.
- `fastembed-rs` for embeddings; ONNX model cached at
  `/opt/flashback/fastembed-cache` (pre-fetched at Docker build time).

**Deploy**
- `docker compose up` — Postgres + sidecar + REST + MCP, all wired.
- DigitalOcean App Platform spec at `.do/app.yaml` — one-click deploy.
- `deploy/install.sh` for fresh-VPS bootstrap.

**Consolidation**
- Background worker promotes working → episodic and distills episodic →
  semantic on configurable intervals (daily + weekly defaults).
- Per-user scoping; results logged in `consolidation_runs`.

### Security posture (shipped with 0.1.0)

CI / repo hygiene running on every push + PR:
- `cargo fmt --check`, `cargo clippy`, `cargo test`, release build (`ci.yml`)
- SonarCloud scan with `cargo llvm-cov` coverage (`build.yml`)
- `actionlint`, `trufflehog --only-verified`, `gitleaks detect`,
  `cargo deny check` (advisories + bans + licenses + sources),
  `semgrep --config auto`, `actions/dependency-review-action` on PRs
  (`security.yml`)
- GitHub CodeQL with the `security-and-quality` query suite for Rust
  (`codeql.yml`)
- OpenSSF Scorecard, weekly + on push, publishes public score
  (`scorecard.yml`)
- CycloneDX + SPDX SBOMs generated and attached on every release
  (`sbom.yml`)
- Dependabot with grouped major + minor-and-patch, capped at 5 PRs per
  ecosystem per week.

Repo-level:
- Secret scanning + push protection on (GitHub-native).
- Dependabot vulnerability alerts + automated security updates on.
- All GitHub Actions SHA-pinned (except `ossf/scorecard-action`, which the
  Scorecard webapp requires as a tag pin — documented inline).
- CODEOWNERS, PR template with security checklist, bug + security-contact
  issue templates.

### Not in 0.1.0 (designs documented; implementation deferred)

- **Multi-tenant isolation.** `docs/TENANCY.md` is exploratory — visibility
  scoping, group memberships, per-tenant consolidation are designed but not
  shipped. Today every memory belongs to one `user_id` and admin endpoints
  see all of a user's memories.
- **Cognitive modes.** `docs/MODES.md` is exploratory — per-project default
  modes, caller overrides, LLM auto-classification all designed but not
  shipped.
- **Branch protection on `main`.** Solo-dev project today; deliberate
  trade-off until contributors join.
- **Custom secret-scanning patterns** (non-provider + validity-checks).
  Require an org-level toggle on Horizon-Digital-Engineering; basic
  GitHub-provider scanning is on.
- **Private vulnerability reporting.** Org-level toggle; not configured.

### License

Business Source License 1.1. The Licensed Work is © 2026 Horizon Digital
Engineering LLC. Non-production use is freely permitted. Production use is
permitted except for offering Flashback as a hosted or managed service that
competes with one offered by the Licensor. License auto-converts to Apache
License 2.0 on **2030-05-23** (four years from this release).

See [LICENSE](./LICENSE) for the full text.

[0.1.0]: https://github.com/Horizon-Digital-Engineering/flashback/releases/tag/v0.1.0
