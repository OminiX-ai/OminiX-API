# Releasing OminiX-API

## Prerequisites

```bash
cargo install cargo-release git-cliff
```

## Release a new version

```bash
# Preview what will happen
cargo release patch --dry-run

# Release (bumps version, generates changelog, commits, tags, pushes)
cargo release patch             # 0.1.0 → 0.1.1
cargo release minor             # 0.1.0 → 0.2.0
cargo release major             # 0.1.0 → 1.0.0
```

## What happens

1. `cargo release patch` bumps version in `Cargo.toml`
2. Runs `git-cliff` to update `CHANGELOG.md` from conventional commits
3. Commits: `release: v0.1.1`
4. Tags: `v0.1.1`
5. Pushes commit + tag to `ominix/main`
6. Tag push triggers `.github/workflows/release.yml`:
   - Checks out both OminiX-API and OminiX-MLX (sibling layout)
   - Builds release binary on Apple Silicon
   - Packages as `ominix-api-0.1.1-darwin-aarch64.tar.gz`
   - Creates GitHub release with changelog

## Commit message format

Use [conventional commits](https://www.conventionalcommits.org/) for automatic changelog grouping:

```
feat: add new endpoint          → Features
fix: handle empty input         → Bug Fixes
perf: reduce memory allocation  → Performance
refactor: simplify handler      → Refactor
docs: update README             → Documentation
ci: fix release workflow        → CI
```

## Notes

- `publish = false` — this crate is not published to crates.io (path deps to OminiX-MLX)
- The release workflow needs `MLX_REPO_TOKEN` secret if OminiX-MLX is private
- Always `--dry-run` first to verify
