"""The torch-facing Dataset over the memory-mapped store.

:mod:`dsio.dataset.dataset` turns a fold's positions into windows read from the mmap on
demand, and collates them into the batches Lightning trains on. See that module for the
paradigm dsio commits to: which pretext objective a dataset serves is a constructor
argument, not a runtime flag.
"""
