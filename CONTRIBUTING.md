# Contributing to Flashback

Flashback is a **public** repository. Everything you push — code, commit
messages, docs — is public and permanent.

## Status & contributions

Flashback is built and shipped by Horizon Digital Engineering, and it moves
fast: `main` changes frequently, the storage schema is still being cut back,
and the APIs are not stable yet. It's public so you can read it, run it, and
fork it — **not** as an open call for contributions. Unsolicited PRs may sit
unreviewed or be closed without merge, especially when they collide with
in-flight work. Found a bug or want a change? **Open an issue first** so we can
tell you whether it fits before you spend time on a PR.

The conventions below are for us and anyone working from a fork.

## The raw layer is append-only

This is the one rule the whole design rests on. `raw_records` holds what a
writer sent us and nothing we worked out ourselves. It is enforced by database
triggers, not by convention: an `UPDATE`, a `DELETE` or a `TRUNCATE` raises.

Two things follow, and both have bitten before:

- **A value that gets recomputed does not belong in raw.** If a rebuild could
  ever produce a different answer, it is derived — put it in a `derived_*`
  table where a wrong answer is a re-derivation instead of a permanent hole.
- **A migration that changes stored bytes changes the hash chain.** Every row
  is chained over arrival order, so altering one invalidates every row after
  it. Schema changes to `raw_records` mean a rebuild, not an in-place fix.

The sandbox (`playground` schema) is append-only for the same reason. A
rehearsal with different storage semantics is not a rehearsal.

## Building and testing

```bash
cargo build --workspace
cargo test --workspace          # needs DATABASE_URL and a Postgres role with CREATEDB
cargo fmt --all
cargo clippy --workspace --all-targets
```

Database-backed tests use `#[sqlx::test]`, which creates a throwaway database
per test. Point `DATABASE_URL` at a Postgres with pgvector available.

### Coverage

```bash
cargo llvm-cov --workspace --summary-only
```

New code should arrive with tests. The bar is not a percentage — it is that a
test asserts a **property** (nothing crosses a user boundary, the output is
finite, nothing executable escapes into HTML) rather than that a function
returns what it currently returns. Every defect found in the last coverage pass
was in code that had a comment claiming it was correct.

### Fuzzing

The parsers that eat model output are fuzzed:

```bash
cargo +nightly fuzz run parse_extraction -- -max_total_time=60
```

Targets live in `fuzz/fuzz_targets`. They run nightly in CI and are
deliberately not a pull-request check — a stochastic run must never block
somebody else's merge. If you add a parser that reads bytes we didn't write,
add a target for it.

## Commit messages

Plain English, imperative, describing what changed and why. The subject line
says what the change does, not which file it touched:

```
Stop one unrecognised word discarding a whole extraction

Extraction's intent and operation fields parsed strictly, so a model that
answered "note" where the enum has "update" failed the parse outright — the
topic and the entities went with it.
```

- Explain **why** in the body when it isn't obvious from the diff.
- No tool names, no generated-by trailers, no co-author trailers.
- A commit that fixes a bug should say what the bug did to somebody, not just
  which line moved.

## Pull requests

One PR per unit of work, even when it covers several concerns — commit per
concern inside it. Keep the body a description of the change and its
consequences, not a changelog of the branch.

CI must be green before merge: format, clippy, tests, CodeQL, SonarQube,
cargo-deny, semgrep, and both secret scanners.

## Security

Don't file security issues publicly. See [SECURITY.md](SECURITY.md) — report
to `security@horizon-digital.dev`.

Two things the scanners will catch, so save yourself the round trip:

- **Never commit a credential**, including a realistic-looking fake one. Test
  fixtures that assign a long literal to something named `api_key` trip the
  secret scanners; name the binding for what it is and keep the value short.
- **Never commit an env file.** `.env.example` is the one that belongs in the
  repo, and it carries placeholders only.

## Public-repo hygiene

Flashback is public. Nothing in this repository may name private
infrastructure — internal hostnames, tailnet addresses, private repository
names, or the names of unrelated internal projects. That applies to code,
comments, commit messages, test fixtures and documentation equally.

Design notes, deployment records and anything else operational live in a
private repository, not here.
