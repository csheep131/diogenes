"""Test runner and orchestrator for Diogenes model testing."""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from diogenes.inference import DiogenesInference, InferenceResult
from diogenes.model import DiogenesModel, EpistemicMode

from diogenes.testing.core.storage import TestResult, TestStorage


logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Configuration for test execution."""

    # Model settings
    model_name: str = "Qwen/Qwen3-0.6B"
    model_path: Optional[str] = None
    use_4bit: bool = False

    # Inference settings
    temperature: float = 0.7
    max_length: int = 512
    top_p: float = 0.9
    do_sample: bool = True

    # Test settings
    return_logprobs: bool = True
    compute_latency: bool = True

    # Storage settings
    storage_path: Optional[str] = None
    storage_backend: str = "jsonl"  # 'jsonl' or 'sqlite'

    # Parallel execution
    max_workers: int = 4

    @classmethod
    def from_yaml(cls, path: str) -> "TestConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            TestConfig instance
        """
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Merge with defaults
        testing_config = config.get("testing", {})
        inference_config = config.get("inference", {})
        model_config = config.get("model", {})

        return cls(
            model_name=model_config.get("name", "Qwen/Qwen3-0.6B"),
            model_path=model_config.get("cache_dir"),
            use_4bit=model_config.get("use_4bit", False),
            temperature=inference_config.get("temperature", 0.7),
            max_length=inference_config.get("max_length", 512),
            top_p=inference_config.get("top_p", 0.9),
            do_sample=inference_config.get("do_sample", True),
            storage_path=testing_config.get("storage_path"),
            storage_backend=testing_config.get("storage_backend", "jsonl"),
            max_workers=testing_config.get("max_workers", 4),
        )


@dataclass
class TestSuite:
    """A collection of test cases."""

    name: str
    description: str
    test_cases: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str) -> "TestSuite":
        """Load test suite from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            TestSuite instance
        """
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            name=data.get("name", "unknown"),
            description=data.get("description", ""),
            test_cases=data.get("test_cases", []),
            metadata=data.get("metadata", {}),
        )


class TestRunner:
    """Orchestrates test execution for Diogenes models.

    Features:
    - Single and batch test execution
    - Parallel processing
    - Integration with existing inference.py
    - Result storage and retrieval
    """

    def __init__(
        self,
        model: Optional[DiogenesModel] = None,
        config: Optional[TestConfig] = None,
    ):
        """Initialize test runner.

        Args:
            model: Pre-loaded DiogenesModel (optional)
            config: Test configuration
        """
        self.config = config or TestConfig()
        self.model = model
        self.inference: Optional[DiogenesInference] = None
        self.storage: Optional[TestStorage] = None

        if self.config.storage_path:
            self.storage = TestStorage(
                self.config.storage_path,
                self.config.storage_backend,
            )

    def load_model(self) -> None:
        """Load the model if not already loaded."""
        if self.model is None:
            logger.info(f"Loading model: {self.config.model_name}")
            self.model = DiogenesModel.from_pretrained(
                model_name_or_path=self.config.model_name,
                use_4bit=self.config.use_4bit,
                cache_dir=self.config.model_path,
            )

        self.inference = DiogenesInference(
            model=self.model,
            default_max_length=self.config.max_length,
            default_temperature=self.config.temperature,
        )
        logger.info("Model loaded and inference engine initialized")

    def run_single(
        self,
        prompt: str,
        expected_response: Optional[str] = None,
        expected_mode: Optional[str] = None,
        suite_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        save_result: bool = True,
    ) -> TestResult:
        """Run a single test case.

        Args:
            prompt: Input prompt
            expected_response: Expected response (for evaluation)
            expected_mode: Expected epistemic mode
            suite_name: Name of the test suite
            tags: Additional tags
            save_result: Whether to save the result

        Returns:
            TestResult object
        """
        if self.inference is None:
            self.load_model()

        test_id = str(uuid.uuid4())
        start_time = time.time()

        # Run inference
        result: InferenceResult = self.inference.generate(
            prompt=prompt,
            max_length=self.config.max_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            return_logprobs=self.config.return_logprobs,
        )

        latency_ms = (time.time() - start_time) * 1000 if self.config.compute_latency else 0.0

        # Evaluate correctness if ground truth provided
        is_correct = None
        if expected_response is not None:
            is_correct = result.text.strip() == expected_response.strip()

        # Create test result
        test_result = TestResult(
            test_id=test_id,
            prompt=prompt,
            response=result.text,
            epistemic_mode=result.epistemic_mode.value,
            confidence=result.confidence,
            tokens=result.tokens,
            logprobs=result.logprobs,
            expected_response=expected_response,
            expected_mode=expected_mode,
            is_correct=is_correct,
            model_name=self.config.model_name,
            model_path=self.config.model_path,
            temperature=self.config.temperature,
            max_length=self.config.max_length,
            suite_name=suite_name,
            tags=tags or [],
            latency_ms=latency_ms,
            token_count=len(result.tokens),
        )

        # Save result
        if save_result and self.storage:
            self.storage.save(test_result)

        return test_result

    def run_batch(
        self,
        prompts: list[str],
        expected_responses: Optional[list[str]] = None,
        expected_modes: Optional[list[str]] = None,
        suite_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        parallel: bool = False,
        save_results: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[TestResult]:
        """Run batch tests.

        Args:
            prompts: List of input prompts
            expected_responses: List of expected responses
            expected_modes: List of expected epistemic modes
            suite_name: Name of the test suite
            tags: Additional tags
            parallel: Use parallel execution
            save_results: Whether to save results
            progress_callback: Callback for progress updates (current, total)

        Returns:
            List of TestResult objects
        """
        if self.inference is None:
            self.load_model()

        n = len(prompts)
        results: list[TestResult] = []

        if parallel and self.config.max_workers > 1:
            results = self._run_batch_parallel(
                prompts=prompts,
                expected_responses=expected_responses,
                expected_modes=expected_modes,
                suite_name=suite_name,
                tags=tags,
                progress_callback=progress_callback,
            )
        else:
            results = self._run_batch_sequential(
                prompts=prompts,
                expected_responses=expected_responses,
                expected_modes=expected_modes,
                suite_name=suite_name,
                tags=tags,
                progress_callback=progress_callback,
            )

        # Save results
        if save_results and self.storage:
            for result in results:
                self.storage.save(result)

        return results

    def _run_batch_sequential(
        self,
        prompts: list[str],
        expected_responses: Optional[list[str]] = None,
        expected_modes: Optional[list[str]] = None,
        suite_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[TestResult]:
        """Run batch tests sequentially."""
        results = []
        n = len(prompts)

        for i, prompt in enumerate(prompts):
            result = self.run_single(
                prompt=prompt,
                expected_response=expected_responses[i] if expected_responses else None,
                expected_mode=expected_modes[i] if expected_modes else None,
                suite_name=suite_name,
                tags=tags,
                save_result=False,  # Save all at once
            )
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, n)

        return results

    def _run_batch_parallel(
        self,
        prompts: list[str],
        expected_responses: Optional[list[str]] = None,
        expected_modes: Optional[list[str]] = None,
        suite_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[TestResult]:
        """Run batch tests in parallel."""
        results = [None] * len(prompts)
        completed = 0
        n = len(prompts)

        def run_test(idx: int) -> tuple[int, TestResult]:
            return idx, self.run_single(
                prompt=prompts[idx],
                expected_response=expected_responses[idx] if expected_responses else None,
                expected_mode=expected_modes[idx] if expected_modes else None,
                suite_name=suite_name,
                tags=tags,
                save_result=False,
            )

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(run_test, i): i for i in range(n)}

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed += 1

                if progress_callback:
                    progress_callback(completed, n)

        return [r for r in results if r is not None]

    def run_suite(
        self,
        suite: TestSuite,
        parallel: bool = False,
        save_results: bool = True,
    ) -> list[TestResult]:
        """Run a complete test suite.

        Args:
            suite: TestSuite to run
            parallel: Use parallel execution
            save_results: Whether to save results

        Returns:
            List of TestResult objects
        """
        logger.info(f"Running test suite: {suite.name}")

        prompts = []
        expected_responses = []
        expected_modes = []

        for case in suite.test_cases:
            prompts.append(case.get("prompt", ""))
            expected_responses.append(case.get("expected_response"))
            expected_modes.append(case.get("expected_mode"))

        return self.run_batch(
            prompts=prompts,
            expected_responses=expected_responses or None,
            expected_modes=expected_modes or None,
            suite_name=suite.name,
            tags=suite.metadata.get("tags"),
            parallel=parallel,
            save_results=save_results,
        )

    def get_storage(self) -> Optional[TestStorage]:
        """Get the storage backend."""
        return self.storage

    def close(self) -> None:
        """Close resources."""
        if self.storage:
            self.storage.close()


def create_runner_from_config(config_path: str) -> TestRunner:
    """Create a TestRunner from a configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configured TestRunner
    """
    config = TestConfig.from_yaml(config_path)
    return TestRunner(config=config)
