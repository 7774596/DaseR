# Contributing to DaseR

## Branch Naming

Create a branch for every change. Never commit directly to `master`.

```
feat/<topic>       # new feature        e.g. feat/storage-eviction
fix/<topic>        # bug fix            e.g. fix/ring-buffer-wrap
refactor/<topic>   # refactor           e.g. refactor/ipc-client
chore/<topic>      # tooling / config   e.g. chore/ci-setup
test/<topic>       # tests only         e.g. test/connector-coverage
docs/<topic>       # documentation      e.g. docs/architecture
```

## Commit Format

This project uses [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <short description>
```

**Types:** `feat` | `fix` | `refactor` | `chore` | `test` | `docs`

**Scopes:** `scaffold` | `storage` | `server` | `connector` | `tests` | `ci` | `docs`

**Examples:**
```
feat(connector): add async IPC client for worker role
fix(storage): correct ring buffer wrap-around offset calculation
test(e2e): add cold-read latency assertion
chore(ci): add ruff pre-commit hook
```

## Merging to master

- All branches are merged via **squash merge** — the entire branch becomes one commit on `master`.
- The PR title is the squash commit message and must follow the commit format above.
- Force-pushing to `master` is prohibited.

## Pull Request Checklist

Before opening a PR, verify:

- [ ] `pre-commit run --all-files` passes
- [ ] All new/modified functions have type hints and docstrings
- [ ] New features and bug fixes include tests
- [ ] Tests use `IOUringBackend` or mock — no GDS hardware required
- [ ] No cross-layer imports (connector must not import server internals)
- [ ] No `threading` + `asyncio` mixing
- [ ] No LMCache imports in `daser/`

## Reporting Issues

Use the issue templates in `.github/ISSUE_TEMPLATE/`. For bugs, include a minimal reproduction. For features, describe the motivation and how it fits the architecture.

## Code Style

See [CLAUDE.md](CLAUDE.md) for detailed coding conventions, architecture rules, and the full code review checklist.
