## Summary

<!-- Why this change exists. One or two sentences. -->

## Changes

<!-- What changed, at a level a reviewer can scan in 30s. -->

## Verification

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [ ] `cargo test --workspace --all-features`
- [ ] Manually exercised the change end-to-end (MCP tool call / HTTP endpoint / docker compose up)

## Security checklist

- [ ] No new secret material committed (check `git diff` for tokens/keys/PEMs)
- [ ] No new license outside the deny.toml allowlist (CI will fail otherwise)
- [ ] No new unbounded `unsafe` blocks; if any, the `// SAFETY:` comment justifies them
- [ ] Any new auth / tenancy / visibility path has tests covering the deny case, not just the allow case

## Migrations / deploy notes

<!-- Schema migrations? New env var? Restart required? Leave blank if none. -->
