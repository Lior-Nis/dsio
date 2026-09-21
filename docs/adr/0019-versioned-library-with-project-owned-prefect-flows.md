---
status: accepted
date: 2026-09-18
supersedes: ADR-0017, ADR-0018
---

# DSio is a versioned library orchestrated by project-owned Prefect flows

DSio is published and consumed as a pinned Python package rather than cloned into each project, and each consuming project defines its execution DAG directly as a Prefect flow. This replaces the fixed fold-as-process runner and clone-and-merge distribution model: they simplified one repository but could not provide one evolving, importable component library for unrelated projects. Prefect owns orchestration, MLflow owns evidence, and DSio owns only their reproducibility boundary and reusable ML components; there is no DSio graph model or orchestration CLI. Each Prefect flow execution creates a new immutable parent MLflow Run, while retries remain child Attempts of that parent and cross-execution reuse references verified immutable Evidence.
