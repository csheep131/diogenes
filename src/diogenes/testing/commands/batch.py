"""Batch testing command for running multiple tests."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from diogenes.model import DiogenesModel

from diogenes.testing.core.storage import TestResult, TestStorage
from diogenes.testing.core.runner import TestRunner, TestConfig, TestSuite


logger = logging.getLogger(__name__)
console = Console()


def batch_test(
    prompts: Optional[list[str]] = None,
    prompts_file: Optional[str] = None,
    suite_file: Optional[str] = None,
    model: Optional[DiogenesModel] = None,
    model_name: str = "Qwen/Qwen3-0.6B",
    model_path: Optional[str] = None,
    use_4bit: bool = False,
    temperature: float = 0.7,
    max_length: int = 512,
    top_p: float = 0.9,
    storage_path: Optional[str] = None,
    storage_backend: str = "jsonl",
    parallel: bool = True,
    max_workers: int = 4,
    suite_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> list[TestResult]:
    """Run batch tests on multiple prompts.

    Args:
        prompts: List of prompts to test
        prompts_file: Path to file containing prompts (one per line or JSON)
        suite_file: Path to test suite JSON file
        model: Pre-loaded model
        model_name: Model name for loading
        model_path: Path to local model
        use_4bit: Use 4-bit quantization
        temperature: Sampling temperature
        max_length: Maximum generation length
        top_p: Nucleus sampling parameter
        storage_path: Path to store results
        storage_backend: Storage backend ('jsonl' or 'sqlite')
        parallel: Use parallel execution
        max_workers: Maximum number of workers for parallel execution
        suite_name: Name for the test suite
        tags: Tags to apply to all tests

    Returns:
        List of TestResult objects
    """
    # Load prompts
    if suite_file:
        return _run_from_suite(
            suite_file=suite_file,
            model=model,
            model_name=model_name,
            model_path=model_path,
            use_4bit=use_4bit,
            temperature=temperature,
            max_length=max_length,
            top_p=top_p,
            storage_path=storage_path,
            storage_backend=storage_backend,
            parallel=parallel,
            max_workers=max_workers,
        )

    if prompts_file:
        prompts = _load_prompts(prompts_file)

    if not prompts:
        console.print("[red]No prompts provided. Use --prompts, --prompts-file, or --suite-file[/red]")
        return []

    # Create runner
    config = TestConfig(
        model_name=model_name,
        model_path=model_path,
        use_4bit=use_4bit,
        temperature=temperature,
        max_length=max_length,
        top_p=top_p,
        storage_path=storage_path,
        storage_backend=storage_backend,
        max_workers=max_workers,
    )

    runner = TestRunner(model=model, config=config)

    try:
        # Run batch tests with progress bar
        results = []

        def progress_callback(current: int, total: int) -> None:
            progress.update(task_id, completed=current)

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "[green]Running tests...",
                total=len(prompts),
            )

            results = runner.run_batch(
                prompts=prompts,
                suite_name=suite_name,
                tags=tags,
                parallel=parallel,
                save_results=True,
                progress_callback=progress_callback,
            )

        # Display summary
        _display_batch_summary(results)

        return results

    finally:
        runner.close()


def _load_prompts(file_path: str) -> list[str]:
    """Load prompts from a file.

    Args:
        file_path: Path to prompts file

    Returns:
        List of prompts
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    # Try JSON first
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # List of strings
            if all(isinstance(p, str) for p in data):
                return data
            # List of objects with 'prompt' field
            return [item.get("prompt", "") for item in data if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    # Fall back to line-by-line
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _run_from_suite(
    suite_file: str,
    model: Optional[DiogenesModel] = None,
    model_name: str = "Qwen/Qwen3-0.6B",
    model_path: Optional[str] = None,
    use_4bit: bool = False,
    temperature: float = 0.7,
    max_length: int = 512,
    top_p: float = 0.9,
    storage_path: Optional[str] = None,
    storage_backend: str = "jsonl",
    parallel: bool = True,
    max_workers: int = 4,
) -> list[TestResult]:
    """Run tests from a test suite file.

    Args:
        suite_file: Path to test suite JSON file
        model: Pre-loaded model
        model_name: Model name for loading
        model_path: Path to local model
        use_4bit: Use 4-bit quantization
        temperature: Sampling temperature
        max_length: Maximum generation length
        top_p: Nucleus sampling parameter
        storage_path: Path to store results
        storage_backend: Storage backend
        parallel: Use parallel execution
        max_workers: Maximum number of workers

    Returns:
        List of TestResult objects
    """
    console.print(f"[bold blue]Loading test suite: {suite_file}[/bold blue]")

    suite = TestSuite.from_json(suite_file)
    console.print(f"[dim]Suite: {suite.name} - {len(suite.test_cases)} test cases[/dim]")

    # Create runner
    config = TestConfig(
        model_name=model_name,
        model_path=model_path,
        use_4bit=use_4bit,
        temperature=temperature,
        max_length=max_length,
        top_p=top_p,
        storage_path=storage_path,
        storage_backend=storage_backend,
        max_workers=max_workers,
    )

    runner = TestRunner(model=model, config=config)

    try:
        results = []

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                f"[green]Running {suite.name}...",
                total=len(suite.test_cases),
            )

            def progress_callback(current: int, total: int) -> None:
                progress.update(task_id, completed=current)

            results = runner.run_suite(
                suite=suite,
                parallel=parallel,
                save_results=True,
                progress_callback=progress_callback,
            )

        _display_batch_summary(results, suite.name)

        return results

    finally:
        runner.close()


def _display_batch_summary(results: list[TestResult], suite_name: Optional[str] = None) -> None:
    """Display summary of batch test results.

    Args:
        results: List of TestResult objects
        suite_name: Optional suite name
    """
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    # Calculate statistics
    total = len(results)
    mode_counts: dict[str, int] = {}
    total_confidence = 0.0
    total_latency = 0.0
    correct_count = 0
    evaluated_count = 0

    for result in results:
        mode_counts[result.epistemic_mode] = mode_counts.get(result.epistemic_mode, 0) + 1
        total_confidence += result.confidence
        total_latency += result.latency_ms

        if result.is_correct is not None:
            evaluated_count += 1
            if result.is_correct:
                correct_count += 1

    # Summary panel
    summary_lines = [
        f"[bold]Total Tests:[/bold] {total}",
        f"[bold]Average Confidence:[/bold] {total_confidence / total:.4f}",
        f"[bold]Average Latency:[/bold] {total_latency / total:.2f} ms",
    ]

    if evaluated_count > 0:
        accuracy = correct_count / evaluated_count
        summary_lines.append(f"[bold]Accuracy:[/bold] {accuracy:.2%} ({correct_count}/{evaluated_count})")

    console.print(
        Panel(
            "\n".join(summary_lines),
            title=f"[bold]Batch Test Summary[/bold]" + (f" - {suite_name}" if suite_name else ""),
            border_style="green",
        )
    )

    # Mode distribution table
    table = Table(title="Epistemic Mode Distribution", show_header=True, header_style="bold magenta")
    table.add_column("Mode", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Percentage", style="yellow")

    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        percentage = count / total * 100
        table.add_row(mode, str(count), f"{percentage:.1f}%")

    console.print(table)


def load_prompts_from_library(
    library_name: str,
    category: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Load prompts from a built-in prompt library.

    Args:
        library_name: Name of the prompt library
        category: Optional category filter

    Returns:
        List of prompt dictionaries
    """
    # Built-in prompt libraries would be stored in suites/
    library_path = Path(__file__).parent.parent / "suites" / f"{library_name}.json"

    if not library_path.exists():
        raise FileNotFoundError(f"Prompt library not found: {library_name}")

    with open(library_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = data.get("test_cases", [])

    if category:
        prompts = [p for p in prompts if p.get("category") == category]

    return prompts
