# Git and Collaboration Workflow

## Branch Naming

```text
feature/eda-profile
feature/normalization
feature/blocking-baseline
feature/pairwise-features
feature/matching-model
feature/evaluation
docs/experiment-log
fix/output-validation
```

## Commit Format

Use concise, meaningful commits:

```text
feat(blocking): add token-based candidate generation
feat(features): add address similarity features
fix(output): validate candidate_pairs schema
docs: record blocking experiment B3
test(normalization): cover unicode and null values
```

## Pull Request Requirements

Every PR should contain:

1. What changed?
2. Why was it needed?
3. Files/modules changed.
4. How it was tested.
5. Runtime or memory impact, if relevant.
6. Known limitations.
7. Any follow-up task.

## Integration Rules

- Do not commit directly to `main`.
- Keep PRs small enough to review.
- Resolve merge conflicts with the module owner.
- Run tests before requesting review.
- Do not merge code that changes output schemas without notifying all members.
- Update documentation in the same PR or create a linked documentation task.
