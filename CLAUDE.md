# DSio

DSio is a reproducible ML/DL experimentation spine: one package rooted at `src/dsio/`,
with one root `pyproject.toml`, that runs its own test suite from a fresh clone.

It is **not** a Copier template — that model was reversed by
`docs/adr/0018-a-repository-not-a-template.md`. Projects clone the repository and pull
spine improvements with `git fetch upstream && git merge upstream/main`.

## Company context

Business context for this project — what Nix is, who the client is, what the
current bet is — lives in the company knowledgebase at `~/Projects/nix`.

- Venture note: `ventures/DSio.md`
- Company: `company/Nix.md`, `company/Current Bet.md`

Do not duplicate that context here, and do not write project research there —
see `~/Projects/nix/_meta/project-kb-boundary.md`.
