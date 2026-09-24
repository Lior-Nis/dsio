---
status: accepted
date: 2026-09-24
depends-on: ADR-0021
---

# Exports resolve components from training evidence

A checkpoint contains parameter state, not the constructor contract needed to restore its
model or the input preprocessor needed for inference. Repeating those constructors in an
export task lets training and export drift independently, especially when fitted constructor
parameters are passed through an ad hoc task result.

Training provenance therefore records complete component configurations for `model` and
`preprocessor`: an importable reference plus canonical constructor parameters.
`require_checkpoint_lineage()` validates the checkpoint Run and may return named component
configurations only when the current consumer Git state, dependency lock, and installed DSIO
package match the recorded execution evidence. The exporter resolves those configurations and
then builds the predictor. It does not redeclare the model or preprocessor constructor.

This is deliberately not a component registry or a second model store. MLflow remains the
evidence store, project packages still own project-specific component code, and DSIO supplies
only validation and resolution. A bare component reference remains valid descriptive
provenance, but it is insufficient for checkpoint reconstruction. Older evidence without
versioned execution context remains readable, while export reconstruction fails closed until
the recorded code and environment are restored.

## Consequences

Fitted values such as Bike Sharing's scaler statistics are part of immutable training evidence
instead of task-to-task plumbing. Exporting an older checkpoint requires checking out the
recorded consumer code and dependency lock; DSIO detects a mismatch rather than silently
loading state into today's class. Model weights and MLflow model packaging remain unchanged.
