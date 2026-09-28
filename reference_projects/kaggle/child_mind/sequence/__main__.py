"""Reproducible command-line entrypoint for the raw-sequence CMI experiment."""

from __future__ import annotations

import argparse
import json

from reference_projects.kaggle.child_mind.sequence.flow import child_mind_sequence_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("workspace")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--window-length", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--existing-store-path")
    parser.add_argument("--no-calibrate", action="store_true")
    arguments = parser.parse_args()
    result = child_mind_sequence_flow(
        arguments.data_dir,
        arguments.workspace,
        seed=arguments.seed,
        window_length=arguments.window_length,
        max_epochs=arguments.max_epochs,
        accelerator=arguments.accelerator,
        calibrate=not arguments.no_calibrate,
        existing_store_path=arguments.existing_store_path,
    )
    summary = {
        name: value
        for name, value in result.items()
        if name not in {"assignments", "split_groups"}
    }
    print("DSIO_CMI_SEQUENCE_RESULT=" + json.dumps(summary, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
