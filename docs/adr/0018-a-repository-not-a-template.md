# 18. This is a repository, not a template

Status: accepted (2026-08-20)
Supersedes: the Copier-template distribution model described in `copier.yml` and the root
README.
Implemented: yes, by Plan 2a. The `project → dsio` direction is now enforced by nothing —
there is one package, so the direction this ADR described no longer has two sides, and the
import-linter contract that stood in for it has been retired.

## Context

dsio was distributed as a Copier template. The argument for it, stated in `copier.yml` and
repeated in the README, was propagation: *"a plain fork can only merge; a Copier project
records its answers and re-applies template changes on top of local edits, which is the
difference between spine improvements reaching every project and rotting in whichever repo
they were written."*

The propagation goal is right. The premise about git is not.

A fork "can only merge" was never a property of git — it was a consequence of every project
renaming its package. `copier.yml` asks for `module_name`, and `src/{{ module_name }}/` bends
around the answer, so two projects generated from the same template have paths that do not
line up and a merge conflicts on everything. Remove the rename and `git merge upstream/main`
does exactly what `copier update` did.

The template also cost more than the rename. Because `pyproject.toml.jinja` owns that name,
the repository has no root `pyproject.toml` — so **the repo cannot run itself**. That single
fact produced a chain of consequences: `lib/dsio/` had to be a separate distribution, treated
as read-only by convention, which justified a broad CLI and wide `__init__.py` façades so
consumers could inspect a package they were not allowed to edit. Two of those façades
re-exported 41 and 30 symbols.

The cost of the façades was not only maintenance. A package façade means a consumer can reach
a module without any `grep "from dsio.data.cache import"` ever finding it — a blind spot that
hid a consumer in four of the five batches of the deletion work, each time costing a rework
round. The template's shape was, indirectly, making the codebase harder to change safely.

CI is the sharpest evidence. The workflow ran `uv sync --locked` at the repository root and
had never once passed, because there is no root project for uv to find. It had been written
for the layout of a *generated* project. Nobody noticed, because the repo had no remote.

## Decision

The workspace collapses to a single distribution rooted at `src/dsio/`, with one
`pyproject.toml`. No Copier, no Jinja, no `.copier-answers.yml`. The package name is fixed, so
there is nothing left to template.

Projects clone the repository, add their components in the same tree, and pull improvements
with `git remote add upstream …` followed by `git merge upstream/main`.

The repository runs itself: `uv sync --extra cpu && uv run pytest` is green in a fresh
clone. An accelerator extra is mandatory, not optional: a bare `uv sync` deliberately
installs no torch, so a bare `uv run pytest` fails loudly at collection rather than
silently pulling several gigabytes of CUDA wheel.

## Consequences

Propagation is preserved and gets *better*: no answers file, no rendered-path `_exclude`
rules, and none of their traps — `copier.yml` needed an anchored `/runs` because a bare `runs`
also matched `lib/dsio/src/dsio/runs/` and silently dropped the run ledger from every generated
project.

Conflicts become ordinary git conflicts, in the files both sides actually edited.

One guarantee is genuinely weakened, and it should be named rather than glossed. The direction
`project → dsio`, never the reverse, was enforced by *packaging*: the spine had no way to name
a project, because it was a separate distribution that did not depend on one. In a single
package that becomes a lint rule — an import-linter contract — rather than a structural
impossibility. A contract catches the mistake before it lands; packaging made it unthinkable.
That is a real downgrade, accepted because the alternative is a repository that cannot run its
own test suite.

(As implemented, this went further than accepted here — see the status line at the top of
this ADR: the contract was retired rather than kept, because in one package the direction
has no second side left to enforce.)

The second consequence is social rather than technical. A clone that never adds the upstream
remote is a fork, and its fixes rot exactly as the original README warned. The template
enforced the relationship mechanically; a remote is a habit. The mitigation is that the habit
is one command, documented at the top of the README, rather than a tool that must be installed
and understood first.
