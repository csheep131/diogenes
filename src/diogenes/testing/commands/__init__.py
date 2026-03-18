"""CLI commands for Diogenes testing."""

from diogenes.testing.commands.quick import quick_test
from diogenes.testing.commands.batch import batch_test
from diogenes.testing.commands.compare import compare_models
from diogenes.testing.commands.report import generate_report

__all__ = [
    "quick_test",
    "batch_test",
    "compare_models",
    "generate_report",
]
