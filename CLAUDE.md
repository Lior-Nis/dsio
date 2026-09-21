# DSio

DSio is a versioned ML/DL experimentation library rooted at `src/dsio/`. Consumer projects
pin it as a dependency and define their workflows directly with Prefect.

DSio owns reusable pipeline components, not project DAGs, deployment, scheduling, or project
lifecycle. The current boundary is defined by
`docs/adr/0019-versioned-library-with-project-owned-prefect-flows.md` and
`docs/superpowers/specs/2026-09-18-generic-experiment-spine.md`.

## Company context

Business context for this project — what Nix is, who the client is, what the
current bet is — lives in the company knowledgebase at `~/Projects/nix`.

- Venture note: `ventures/DSio.md`
- Company: `company/Nix.md`, `company/Current Bet.md`

Do not duplicate that context here, and do not write project research there —
see `~/Projects/nix/_meta/project-kb-boundary.md`.
