---
status: accepted
date: 2026-09-18
supersedes: ADR-0017, ADR-0018
amended_by: ADR-0020
---

# DSio is a versioned library orchestrated by project-owned Prefect flows

DSio is published and consumed as a pinned Python package rather than cloned into each project, and each consuming project defines its execution DAG directly as a Prefect flow. This replaces the fixed fold-as-process runner and clone-and-merge distribution model: they simplified one repository but could not provide one evolving, importable component library for unrelated projects. Prefect owns orchestration, MLflow owns evidence, and DSio owns only their reproducibility boundary and reusable ML components; there is no DSio graph model or orchestration CLI. ADR-0020 amends the original MLflow parent/child hierarchy: attempts are visible top-level Runs linked by Prefect identity rather than children of an empty flow Run.
