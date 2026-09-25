# Project Charter

## Problem

Develop an entity-resolution system that maps records from source datasets to entities in a reference dataset while generating candidate pairs efficiently.

## Goals

- Build a reproducible end-to-end pipeline.
- Maintain high candidate recall with controlled candidate-set size.
- Develop transparent and testable matching features.
- Compare blocking and matching strategies.
- Produce valid submission artifacts.
- Preserve implementation context for human contributors and AI agents.

## Non-goals

- Using SageMaker without a demonstrated need.
- Using prohibited external data or APIs.
- Building complex models before a reliable baseline exists.
- Optimizing only leaderboard performance while ignoring candidate generation and reproducibility.

## Success criteria

- Full pipeline runs from input to output.
- Candidate generation is scalable and documented.
- Candidate statistics are recorded.
- Output schemas are validated.
- Final methodology reflects the actual implementation.
