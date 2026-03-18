"""Quick test command for single-prompt testing."""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from diogenes.model import DiogenesModel, EpistemicMode
from diogenes.inference import DiogenesInference

from diogenes.testing.core.storage import TestResult, TestStorage


logger = logging.getLogger(__name__)
console = Console()


def quick_test(
    prompt: str,
    model: Optional[DiogenesModel] = None,
    model_name: str = "Qwen/Qwen3-0.6B",
    model_path: Optional[str] = None,
    use_4bit: bool = False,
    temperature: float = 0.7,
    max_length: int = 512,
    top_p: float = 0.9,
    return_logprobs: bool = True,
    storage_path: Optional[str] = None,
    verbose: bool = False,
    attn_implementation: str = "eager",
) -> TestResult:
    """Run a quick single-prompt test.

    Args:
        prompt: Input prompt to test
        model: Pre-loaded model (optional)
        model_name: Model name for loading
        model_path: Path to local model
        use_4bit: Use 4-bit quantization
        temperature: Sampling temperature
        max_length: Maximum generation length
        top_p: Nucleus sampling parameter
        return_logprobs: Return log probabilities
        storage_path: Path to store results
        verbose: Show detailed output
        attn_implementation: Attention implementation to use.
                           Options: "auto", "eager", "flash_attention_2", "sdpa"

    Returns:
        TestResult object
    """
    import uuid
    import time

    # Load model if not provided
    if model is None:
        console.print("[bold blue]Loading model...[/bold blue]")
        model = DiogenesModel.from_pretrained(
            model_name_or_path=model_name,
            use_4bit=use_4bit,
            cache_dir=model_path,
            attn_implementation=attn_implementation,
        )

    # Create inference engine
    inference = DiogenesInference(
        model=model,
        default_max_length=max_length,
        default_temperature=temperature,
    )

    # Run inference
    console.print("[bold green]Running inference...[/bold green]")
    start_time = time.time()

    result = inference.generate(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        return_logprobs=return_logprobs,
    )

    latency_ms = (time.time() - start_time) * 1000

    # Create test result
    test_result = TestResult(
        test_id=str(uuid.uuid4()),
        prompt=prompt,
        response=result.text,
        epistemic_mode=result.epistemic_mode.value,
        confidence=result.confidence,
        tokens=result.tokens,
        logprobs=result.logprobs,
        model_name=model_name,
        model_path=model_path,
        temperature=temperature,
        max_length=max_length,
        latency_ms=latency_ms,
        token_count=len(result.tokens),
    )

    # Save to storage if path provided
    if storage_path:
        storage = TestStorage(storage_path)
        storage.save(test_result)
        storage.close()
        console.print(f"[dim]Result saved to: {storage_path}[/dim]")

    # Display results
    _display_result(test_result, verbose)

    return test_result


def _display_result(result: TestResult, verbose: bool = False) -> None:
    """Display test result in a formatted way.

    Args:
        result: TestResult to display
        verbose: Show detailed information
    """
    # Mode color mapping
    mode_colors = {
        EpistemicMode.DIRECT_ANSWER.value: "green",
        EpistemicMode.CAUTIOUS_LIMIT.value: "yellow",
        EpistemicMode.ABSTAIN.value: "red",
        EpistemicMode.CLARIFY.value: "cyan",
        EpistemicMode.REJECT_PREMISE.value: "magenta",
        EpistemicMode.REQUEST_TOOL.value: "blue",
        EpistemicMode.PROBABILISTIC.value: "white",
    }

    mode_color = mode_colors.get(result.epistemic_mode, "white")

    # Response panel
    response_text = Text(result.response)
    console.print(
        Panel(
            response_text,
            title="[bold]Response[/bold]",
            border_style="blue",
        )
    )

    # Metrics table
    table = Table(title="Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Epistemic Mode", Text(result.epistemic_mode, style=mode_color))
    table.add_row("Confidence", f"{result.confidence:.4f}")
    table.add_row("Latency", f"{result.latency_ms:.2f} ms")
    table.add_row("Token Count", str(result.token_count))

    console.print(table)

    # Verbose output
    if verbose:
        console.print("\n[bold]Detailed Information:[/bold]")
        console.print(f"  Model: {result.model_name}")
        console.print(f"  Temperature: {result.temperature}")
        console.print(f"  Max Length: {result.max_length}")
        console.print(f"  Test ID: {result.test_id}")
        console.print(f"  Timestamp: {result.timestamp}")

        if result.logprobs:
            avg_logprob = sum(result.logprobs) / len(result.logprobs)
            console.print(f"  Avg Log Probability: {avg_logprob:.4f}")


def interactive_quick_test(
    model: Optional[DiogenesModel] = None,
    model_name: str = "Qwen/Qwen3-0.6B",
    attn_implementation: str = "eager",
    **kwargs,
) -> None:
    """Run interactive quick test mode.

    Args:
        model: Pre-loaded model
        model_name: Model name for loading
        attn_implementation: Attention implementation to use
        **kwargs: Additional arguments passed to quick_test
    """
    console.print("[bold cyan]Diogenes Quick Test - Interactive Mode[/bold cyan]")
    console.print("[dim]Type 'quit' or 'exit' to stop[/dim]\n")

    while True:
        try:
            prompt = console.input("[bold green]Prompt:[/bold green] ").strip()

            if prompt.lower() in ("quit", "exit", "q"):
                console.print("[yellow]Exiting interactive mode[/yellow]")
                break

            if not prompt:
                continue

            quick_test(
                prompt=prompt,
                model=model,
                model_name=model_name,
                attn_implementation=attn_implementation,
                **kwargs,
            )
            console.print()  # Empty line

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
