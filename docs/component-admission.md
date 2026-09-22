# Component admission

Reusable models, objectives, transforms, metrics, splitters, samplers, collators, and
related components enter DSIO through one path. This is a source-review process, not a
runtime plugin system.

## Enter experimental

1. Put the implementation under `dsio.experimental` with generic terminology and
   serializable configuration.
2. Select it through its named `module:qualname` import path, or add it to an existing
   DSIO-owned closed dispatcher. Never expose project-side registration.
3. Run `require_admissible_component` with the consumer project package names. The audit
   rejects private dependencies, private DSIO modules, project-name branches, and registry
   mutation. To stay mechanically decidable, experimental source may not use star imports,
   dynamic import/code-evaluation APIs, closed-dispatcher references, or define a runtime
   registration surface. Supplied consumer names are forbidden throughout the source;
   built-in ambiguous terms such as `pulse` are rejected only when project identity is also
   referenced.
4. Add focused unit tests, an integration or property test through the stable DSIO
   contract, deterministic replay evidence, and provenance assertions.
5. Record one real downstream confirmation, limitations, and an adversarial agent review
   in the PR. A human maintainer must explicitly approve entry.

Failing any item keeps the component outside DSIO. Static checks are necessary, not a
substitute for review: genericity and scientific validity require judgment.

## Promote to stable

Promotion moves the implementation to its responsibility-named stable package. The PR
must provide all experimental evidence plus:

- an unrelated second downstream use;
- compatibility tests for the public import path, configuration, determinism, and
  provenance behavior;
- focused unit and integration/property tests in the published distribution;
- migration notes and a semantic-versioning classification;
- a fresh adversarial agent review; and
- explicit human approval saying the component is stable.

There is no automatic promotion. A missing or failed item leaves the component in
`dsio.experimental`.

## Compatibility

- Compatible fixes to stable behavior are patch releases.
- Additive stable APIs are minor releases.
- Incompatible stable behavior requires a major release. When feasible, first deprecate
  it in a minor release with migration notes, then remove it in the next major release.
- Experimental APIs may change in a minor release, but only through this admission and
  review process.

Projects pin a released DSIO version. They are never silently redirected from one stable
component or behavior to another.
