"""Contract suites a project runs against its own implementations.

dsio's hardest-won knowledge is written down as prose in the modules it protects:
``readers.py`` explains why a reader is opened per process, ``examples.py`` explains why a
subset must keep its parent's digest. Prose protects the implementations in this package
and nothing else. The moment a project writes its own reader or its own ``Examples`` — the
whole point of both being protocols — that knowledge is a docstring they may not have read.

These suites are that knowledge as executable checks. Import one, point it at your own
implementation, and the bug dsio already paid for fails your test suite instead of
someone's overnight run.

Test-only, and an import contract keeps it that way: nothing in the spine may import this
package, so nothing here can drift into the training path.
"""

from dsio.testing.examples_contract import ContractViolation, check_examples_contract
from dsio.testing.reader_contract import check_reader_contract

__all__ = ["ContractViolation", "check_examples_contract", "check_reader_contract"]
