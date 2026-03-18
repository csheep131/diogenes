"""Diogenes Model Testing Framework.

Comprehensive testing tools for evaluating Diogenes model performance,
epistemic mode detection, and reliability metrics.
"""

from diogenes.testing.core.storage import TestStorage, TestResult
from diogenes.testing.core.runner import TestRunner, TestConfig
from diogenes.testing.commands.quick import quick_test
from diogenes.testing.commands.batch import batch_test
from diogenes.testing.commands.compare import compare_models

__version__ = "0.1.0"

__all__ = [
    "TestStorage",
    "TestResult",
    "TestRunner",
    "TestConfig",
    "quick_test",
    "batch_test",
    "compare_models",
]
