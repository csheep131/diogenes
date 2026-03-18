"""Test suites for Diogenes model evaluation.

Available suites:
- epistemic_modes.json: Tests for all 7 epistemic response modes
- hallucination.json: Tests for hallucination detection
- calibration.json: Tests for confidence calibration
"""

from pathlib import Path

# Get the directory containing the suite files
SUITES_DIR = Path(__file__).parent

# Suite file paths
EPISTEMIC_MODES_SUITE = SUITES_DIR / "epistemic_modes.json"
HALLUCINATION_SUITE = SUITES_DIR / "hallucination.json"
CALIBRATION_SUITE = SUITES_DIR / "calibration.json"

__all__ = [
    "SUITES_DIR",
    "EPISTEMIC_MODES_SUITE",
    "HALLUCINATION_SUITE",
    "CALIBRATION_SUITE",
]
