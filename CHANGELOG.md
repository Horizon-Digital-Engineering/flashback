# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and Flashback adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] — 2026-09-05

### Fixed
- The systemd installer aborted with "reason: unbound variable" on its
  skip-build path — a checkout whose only change since the last build was
  docs had nothing newer than the binary, and the one branch that never set
  `reason` hit `set -u`. This is the first version whose installer can
  upgrade a box end to end.

## [0.2.0] — 2026-09-04

Breaking. The raw layer was cut back to what writers actually send, the schema
is redeclared from scratch, and a database from 0.1.0 does not migrate forward
— rebuild it. Everything derived is rebuilt from raw, so nothing derived is
lost by doing so.

### Changed
- **Raw holds arrived data only.** `mode` moved to `derived_record_mode`,
  server-inferred `supersedes` to `derived_superseded`, and `prev_id` to
  `derived_link` — a resolution re-run on every rebuild, so a late-arriving
  parent gets linked instead of leaving a permanent gap. `importance` is gone:
  a writer's judgement a rebuild could never disagree with.
- **`project_id` → `topic_id`, `container_id` → `thread_id`.** A topic is where
  a thread is filed — a filter, not a wall. A thread is one conversation.
- **The tamper chain runs along arrival order**, so every record protects every
  record that arrived after it. It used to chain along the causal link, which
  meant an imported corpus produced thousands of one-record chains protecting
  nothing.
- **Token roles: `service` vs `operator`.** A service token reaches the
  REST/MCP API only; an operator token reaches `/admin` only. The middleware
  enforces the wall in both directions, and the admin login refuses service
  tokens.
- **The admin UI is the operator plane** — it shows every user's records,
  curated nodes, references, catalog, proposals, and map, instead of only the
  logged-in token's own rows.
- Schema is declared from scratch in `migrations/` (one file per subsystem, all
  `CREATE`, no `ALTER` archaeology).
- The server and the MCP wrapper bind loopback unless told otherwise.

### Fixed
- **Registers leaked between users.** The built-in clone matched every row in
  the table rather than the template user's, so each caller received a copy of
  every other user's registers, names and descriptions included.
- **The mind map returned 500 on every request**, having outlived two columns
  it still selected.
- **Streamed replies lost characters at network chunk boundaries** — a chunk
  ending mid-character had its bytes replaced before the next could complete
  it, in the reply shown and in the record written from it.
- **One unrecognised word discarded a whole extraction.** A model answering a
  synonym for an enum value failed the parse outright and took the topic and
  entities with it.
- **The MCP wrapper granted any origin.** The API server carries no CORS layer
  on purpose; the wrapper fronts the same memory and the same token.
- **An unknown remote-provider name resolved to a default**, sending the
  configured API key to a vendor the operator never named. Unrecognised
  provider and backend values are now refused at startup.
- **The admin session cookie is `Secure`** when the request arrived over TLS.
- **Bracketed IPv6 hosts parsed wrongly**, so the cloud metadata address the
  SSRF check names by hand was the one spelling it could not see.
- Sandbox tables enforce the foreign keys their production counterparts do.
- `RowNotFound` answers 404 rather than 500; unique and foreign-key violations
  answer 409 and 400.
- Service tokens can no longer be minted as the reserved wildcard user.
- Two cross-site scripting holes in admin rendering.

### Added
- Fuzz targets over the parsers that read model output, run weekly and on
  demand.
- `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
- A release check that refuses a tag disagreeing with the workspace version.

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

[0.2.1]: https://github.com/Horizon-Digital-Engineering/flashback/releases/tag/v0.2.1
[0.2.0]: https://github.com/Horizon-Digital-Engineering/flashback/releases/tag/v0.2.0
[0.1.0]: https://github.com/Horizon-Digital-Engineering/flashback/releases/tag/v0.1.0
