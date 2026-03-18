"""Diogenes Model Test Tool - CLI Entry Point.

A comprehensive testing tool for evaluating Diogenes model performance,
epistemic mode detection, and reliability metrics.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml

from diogenes.testing.commands.batch import batch_test
from diogenes.testing.commands.compare import compare_models
from diogenes.testing.commands.quick import interactive_quick_test, quick_test
from diogenes.testing.commands.report import export_results, generate_report


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer(
    name="diogenes-test",
    help="Diogenes Model Test Tool - Comprehensive testing for epistemic LLM evaluation",
    add_completion=False,
)


# ============================================================================
# Quick Test Commands
# ============================================================================


@app.command("quick")
def quick_cmd(
    prompt: Optional[str] = typer.Argument(
        None,
        help="Input prompt to test. If not provided, enters interactive mode.",
    ),
    model_name: str = typer.Option(
        "Qwen/Qwen3-0.6B",
        "--model",
        "-m",
        help="Model name or path to load",
    ),
    model_path: Optional[str] = typer.Option(
        None,
        "--model-path",
        "-p",
        help="Path to local model directory",
    ),
    use_4bit: bool = typer.Option(
        False,
        "--4bit",
        help="Use 4-bit quantization",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Sampling temperature",
        min=0.0,
        max=2.0,
    ),
    max_length: int = typer.Option(
        512,
        "--max-length",
        "-l",
        help="Maximum generation length",
        min=1,
        max=4096,
    ),
    top_p: float = typer.Option(
        0.9,
        "--top-p",
        help="Nucleus sampling parameter",
        min=0.0,
        max=1.0,
    ),
    attn_implementation: str = typer.Option(
        "eager",
        "--attn-implementation",
        "-a",
        help="Attention implementation: eager, flash_attention_2, sdpa",
    ),
    storage: Optional[str] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Path to store results (JSONL or SQLite)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Run in interactive mode",
    ),
) -> None:
    """Run a quick single-prompt test.

    Examples:
        diogenes-test quick "What is the capital of France?"
        diogenes-test quick -m ./models/my-model -t 0.5
        diogenes-test quick --interactive
        diogenes-test quick --attn-implementation eager
    """
    if interactive or prompt is None:
        # Interactive mode
        interactive_quick_test(
            model_name=model_name,
            model_path=model_path,
            use_4bit=use_4bit,
            temperature=temperature,
            max_length=max_length,
            top_p=top_p,
            attn_implementation=attn_implementation,
            storage_path=storage,
            verbose=verbose,
        )
    else:
        # Single prompt mode
        quick_test(
            prompt=prompt,
            model_name=model_name,
            model_path=model_path,
            use_4bit=use_4bit,
            temperature=temperature,
            max_length=max_length,
            top_p=top_p,
            attn_implementation=attn_implementation,
            storage_path=storage,
            verbose=verbose,
        )


# ============================================================================
# Batch Test Commands
# ============================================================================


@app.command("batch")
def batch_cmd(
    prompts_file: Optional[str] = typer.Option(
        None,
        "--prompts",
        "-f",
        help="Path to file containing prompts (one per line or JSON)",
    ),
    suite: Optional[str] = typer.Option(
        None,
        "--suite",
        help="Path to test suite JSON file",
    ),
    model_name: str = typer.Option(
        "Qwen/Qwen3-0.6B",
        "--model",
        "-m",
        help="Model name or path to load",
    ),
    model_path: Optional[str] = typer.Option(
        None,
        "--model-path",
        "-p",
        help="Path to local model directory",
    ),
    use_4bit: bool = typer.Option(
        False,
        "--4bit",
        help="Use 4-bit quantization",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Sampling temperature",
    ),
    max_length: int = typer.Option(
        512,
        "--max-length",
        "-l",
        help="Maximum generation length",
    ),
    storage: str = typer.Option(
        "./test_results",
        "--storage",
        "-s",
        help="Path to store results",
    ),
    storage_backend: str = typer.Option(
        "jsonl",
        "--backend",
        help="Storage backend (jsonl or sqlite)",
    ),
    parallel: bool = typer.Option(
        True,
        "--parallel/--sequential",
        help="Use parallel execution",
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        "-w",
        help="Number of parallel workers",
        min=1,
        max=16,
    ),
    suite_name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Name for this test suite",
    ),
) -> None:
    """Run batch tests on multiple prompts.

    Examples:
        diogenes-test batch -f prompts.txt
        diogenes-test batch --suite suites/epistemic_modes.json
        diogenes-test batch -f prompts.json --parallel --workers 8
    """
    batch_test(
        prompts_file=prompts_file,
        suite_file=suite,
        model_name=model_name,
        model_path=model_path,
        use_4bit=use_4bit,
        temperature=temperature,
        max_length=max_length,
        storage_path=storage,
        storage_backend=storage_backend,
        parallel=parallel,
        max_workers=workers,
        suite_name=suite_name,
    )


# ============================================================================
# Model Comparison Commands
# ============================================================================


@app.command("compare")
def compare_cmd(
    prompts_file: str = typer.Option(
        ...,
        "--prompts",
        "-f",
        help="Path to file containing prompts for comparison",
    ),
    model_a: str = typer.Option(
        ...,
        "--model-a",
        "-a",
        help="Model A name or path",
    ),
    model_b: str = typer.Option(
        ...,
        "--model-b",
        "-b",
        help="Model B name or path",
    ),
    model_a_name: str = typer.Option(
        "Model A",
        "--name-a",
        help="Display name for Model A",
    ),
    model_b_name: str = typer.Option(
        "Model B",
        "--name-b",
        help="Display name for Model B",
    ),
    use_4bit: bool = typer.Option(
        False,
        "--4bit",
        help="Use 4-bit quantization",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Sampling temperature",
    ),
    max_length: int = typer.Option(
        512,
        "--max-length",
        "-l",
        help="Maximum generation length",
    ),
    attn_implementation: str = typer.Option(
        "eager",
        "--attn-implementation",
        help="Attention implementation: eager, flash_attention_2, sdpa",
    ),
    storage: Optional[str] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Path to store comparison results",
    ),
    auto_judge: bool = typer.Option(
        False,
        "--auto-judge",
        help="Automatically judge winner based on confidence",
    ),
) -> None:
    """Compare two models (A/B testing).

    Examples:
        diogenes-test compare -f prompts.txt -a model_v1 -b model_v2
        diogenes-test compare -f ab_tests.json --model-a ./models/baseline --model-b ./models/finetuned
    """
    # Load prompts
    from diogenes.testing.commands.batch import _load_prompts

    prompts = _load_prompts(prompts_file)

    compare_models(
        prompts=prompts,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        model_a_path=model_a,
        model_b_path=model_b,
        use_4bit=use_4bit,
        temperature=temperature,
        max_length=max_length,
        attn_implementation=attn_implementation,
        storage_path=storage,
        auto_judge=auto_judge,
    )


# ============================================================================
# Report Commands
# ============================================================================


@app.command("report")
def report_cmd(
    storage: str = typer.Option(
        "./test_results",
        "--storage",
        "-s",
        help="Path to test results storage",
    ),
    output: str = typer.Option(
        "./reports",
        "--output",
        "-o",
        help="Output path for report",
    ),
    format: str = typer.Option(
        "html",
        "--format",
        "-f",
        help="Output format (markdown, html, json)",
    ),
    suite_name: Optional[str] = typer.Option(
        None,
        "--suite",
        "-n",
        help="Filter by suite name",
    ),
    start_date: Optional[str] = typer.Option(
        None,
        "--start",
        help="Filter by start date (ISO format)",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end",
        help="Filter by end date (ISO format)",
    ),
    no_plots: bool = typer.Option(
        False,
        "--no-plots",
        help="Exclude plots from report",
    ),
) -> None:
    """Generate report from test results.

    Examples:
        diogenes-test report -s ./results -o ./report.html
        diogenes-test report --suite epistemic_modes --format markdown
    """
    generate_report(
        storage_path=storage,
        output_path=output,
        format=format,
        suite_name=suite_name,
        start_date=start_date,
        end_date=end_date,
        include_plots=not no_plots,
    )


@app.command("export")
def export_cmd(
    storage: str = typer.Option(
        "./test_results",
        "--storage",
        "-s",
        help="Path to test results storage",
    ),
    output: str = typer.Option(
        "./export.csv",
        "--output",
        "-o",
        help="Output path for exported file",
    ),
    format: str = typer.Option(
        "csv",
        "--format",
        "-f",
        help="Export format (csv, json, parquet)",
    ),
    suite_name: Optional[str] = typer.Option(
        None,
        "--suite",
        "-n",
        help="Filter by suite name",
    ),
) -> None:
    """Export test results to a file.

    Examples:
        diogenes-test export -s ./results -o results.csv
        diogenes-test export --format json -o results.json
    """
    export_results(
        storage_path=storage,
        output_path=output,
        format=format,
        suite_name=suite_name,
    )


# ============================================================================
# Utility Commands
# ============================================================================


@app.command("stats")
def stats_cmd(
    storage: str = typer.Option(
        "./test_results",
        "--storage",
        "-s",
        help="Path to test results storage",
    ),
) -> None:
    """Show storage statistics."""
    from diogenes.testing.core.storage import TestStorage

    storage_obj = TestStorage(storage)
    stats = storage_obj.get_statistics()
    storage_obj.close()

    print("\n=== Storage Statistics ===")
    print(f"Storage Path: {stats['storage_path']}")
    print(f"Backend: {stats['backend']}")
    print(f"Total Results: {stats['total_results']}")
    print(f"Average Confidence: {stats['avg_confidence']:.4f}")
    print(f"Average Latency: {stats['avg_latency_ms']:.2f} ms")
    print("\nMode Distribution:")
    for mode, count in sorted(stats["mode_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {mode}: {count}")
    print()


@app.command("init")
def init_cmd(
    config_path: str = typer.Option(
        "./configs/testing.yaml",
        "--config",
        "-c",
        help="Path to create configuration file",
    ),
) -> None:
    """Initialize a testing configuration file."""
    config = {
        "testing": {
            "storage_path": "./test_results",
            "storage_backend": "jsonl",
            "max_workers": 4,
        },
        "model": {
            "name": "Qwen/Qwen3-0.6B",
            "cache_dir": "./models",
            "use_4bit": False,
        },
        "inference": {
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        },
    }

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration file created: {config_path}")


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
