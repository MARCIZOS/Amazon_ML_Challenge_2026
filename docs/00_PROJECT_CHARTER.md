# Project Charter

## 1. Problem Statement

Develop an entity-resolution system that identifies which records from the source datasets correspond to entities in the reference dataset. The pipeline must generate candidate pairs efficiently rather than comparing every record against every other record.

## 2. Goals

- Produce a reproducible end-to-end pipeline.
- Generate a high-recall candidate set with controlled size.
- Build transparent and testable matching features.
- Compare multiple blocking and matching strategies.
- Produce valid submission files.
- Maintain an auditable record of decisions, experiments, and AI-agent contributions.

## 3. Non-Goals

- Using SageMaker solely for appearance or prestige.
- Adding external business databases or APIs unless the rules explicitly permit them.
- Building an unnecessarily complex model before a reliable baseline exists.
- Optimizing only the public leaderboard while neglecting candidate generation quality and reproducibility.

## 4. Success Criteria

The project is considered technically ready when:

- The complete pipeline runs from raw input to final output.
- Candidate generation is deterministic and documented.
- Candidate statistics are reported.
- Matching results pass schema and consistency checks.
- The final document explains preprocessing, blocking, features, model, thresholds, aggregation, and limitations.
- Another team member can reproduce the result from the repository.
