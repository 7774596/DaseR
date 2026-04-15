## Summary

What does this PR do? (1-3 bullet points)

-

## Motivation

Why is this change needed?

## Test plan

How was this tested?

- [ ] Unit tests added / updated
- [ ] `pytest tests/ -q` passes
- [ ] `pre-commit run --all-files` passes

## Checklist

### Correctness
- [ ] Code does what the PR description claims
- [ ] Edge cases handled
- [ ] No regressions — existing tests still pass

### Architecture
- [ ] No cross-layer direct imports
- [ ] No `threading` + `asyncio` mixing
- [ ] No LMCache imports in `daser/`

### Style & Standards
- [ ] `pre-commit run --all-files` passes
- [ ] Type hints and docstrings on all new/modified functions
- [ ] License header on all new Python files

### Testing
- [ ] Tests use `IOUringBackend` or mock — no GDS hardware required
- [ ] Tests target public interface, not internals
