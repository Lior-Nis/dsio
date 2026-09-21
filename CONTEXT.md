# DSio

DSio is the shared vocabulary for reproducible machine-learning experiments assembled by independent projects. It defines the evidence and reusable-component boundaries without owning any project's product lifecycle.

## Experimentation

**Experiment**:
One identified execution of a project-declared workflow that produces reproducible evidence.
_Avoid_: Job, pipeline run

**Experiment Node**:
One independently retryable unit within an Experiment's dependency graph.
_Avoid_: Stage, phase

**Attempt**:
One execution try for an Experiment Node. Failed Attempts remain part of the Experiment's history.
_Avoid_: Run

**Execution Identity**:
The stable identity of an Experiment Node derived from every input capable of changing its output.
_Avoid_: Config hash, cache key

**Evidence**:
Immutable metrics, predictions, datasets, models, or other outputs used to evaluate an Experiment.
_Avoid_: Result

**Evidence Role**:
The declared purpose for which Evidence may be used, such as discovery or confirmation.
_Avoid_: Split type

## Data

**Dataset Identity**:
The immutable identity of the source data consumed by an Experiment Node.
_Avoid_: Dataset path

**Split Manifest**:
A versioned, immutable declaration of named membership roles over one Dataset Identity.
_Avoid_: Fold file, index list

**Partition**:
One named set of role memberships from a Split Manifest consumed by an Experiment Node.
_Avoid_: Fold

**Sample Identity**:
The stable identifier that relates an input sample to its predictions and Evidence.
_Avoid_: Row position, batch index

## Training and inference

**Training System**:
The single DSio-owned composition boundary through which all supported model training occurs.
_Avoid_: Runner, task kind

**Component**:
A named, importable unit injected into the Training System, such as a network, objective, optimizer factory, predictor, transform, or dataset factory.
_Avoid_: Plugin

**Predictor**:
An immutable model artifact containing only the behavior required to turn valid inputs into schema-bound predictions.
_Avoid_: Training checkpoint

**Prediction Schema**:
The versioned contract relating Sample Identities to a Predictor's output fields.
_Avoid_: Result model

## Component maturity

**Experimental Component**:
A reviewed DSio Component proven by one real downstream Experiment but not yet covered by a compatibility promise.
_Avoid_: Project component

**Stable Component**:
A DSio Component proven by an unrelated second use and protected by compatibility tests.
_Avoid_: Blessed component
