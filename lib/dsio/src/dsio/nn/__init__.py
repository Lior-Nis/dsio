"""Torch components, assembled into one chain and trained by one step.

Importing this package registers dsio's built-in components. torch and lightning are an
optional extra, so nothing outside :mod:`dsio.nn` and the torch runner imports it.
"""

from dsio.nn import components  # noqa: F401 - side effect: registers built-in components
