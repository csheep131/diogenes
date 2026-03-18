"""Model comparison command for A/B testing."""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from diogenes.model import DiogenesModel
from diogenes.inference import DiogenesInference, InferenceResult

from diogenes.testing.core.storage import TestResult


logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ComparisonResult:
    """Result of comparing two models on a single prompt."""

    prompt: str
    model_a_response: str
    model_b_response: str
    model_a_mode: str
    model_b_mode: str
    model_a_confidence: float
    model_b_confidence: float
    model_a_latency_ms: float
    model_b_latency_ms: float
    winner: str = ""  # 'A', 'B', or 'tie'
    delta_confidence: float = 0.0
    delta_latency_ms: float = 0.0


@dataclass
class ComparisonSummary:
    """Summary of model comparison across multiple prompts."""

    model_a_name: str
    model_b_name: str
    total_prompts: int
    model_a_wins: int = 0
    model_b_wins: int = 0
    ties: int = 0
    model_a_avg_confidence: float = 0.0
    model_b_avg_confidence: float = 0.0
    model_a_avg_latency_ms: float = 0.0
    model_b_avg_latency_ms: float = 0.0
    mode_distribution_a: dict[str, int] = field(default_factory=dict)
    mode_distribution_b: dict[str, int] = field(default_factory=dict)
    detailed_results: list[ComparisonResult] = field(default_factory=list)

    @property
    def model_a_win_rate(self) -> float:
        """Calculate model A win rate."""
        if self.total_prompts == 0:
            return 0.0
        return self.model_a_wins / self.total_prompts

    @property
    def model_b_win_rate(self) -> float:
        """Calculate model B win rate."""
        if self.total_prompts == 0:
            return 0.0
        return self.model_b_wins / self.total_prompts

    @property
    def tie_rate(self) -> float:
        """Calculate tie rate."""
        if self.total_prompts == 0:
            return 0.0
        return self.ties / self.total_prompts


def compare_models(
    prompts: list[str],
    model_a: Optional[DiogenesModel] = None,
    model_b: Optional[DiogenesModel] = None,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    model_a_path: Optional[str] = None,
    model_b_path: Optional[str] = None,
    use_4bit: bool = False,
    temperature: float = 0.7,
    max_length: int = 512,
    top_p: float = 0.9,
    storage_path: Optional[str] = None,
    auto_judge: bool = False,
) -> ComparisonSummary:
    """Compare two models on the same set of prompts.

    Args:
        prompts: List of prompts to test
        model_a: Pre-loaded model A
        model_b: Pre-loaded model B
        model_a_name: Display name for model A
        model_b_name: Display name for model B
        model_a_path: Path to model A
        model_b_path: Path to model B
        use_4bit: Use 4-bit quantization
        temperature: Sampling temperature
        max_length: Maximum generation length
        top_p: Nucleus sampling parameter
        storage_path: Path to store results
        auto_judge: Automatically judge winner based on confidence

    Returns:
        ComparisonSummary object
    """
    import time
    import uuid

    console.print("[bold cyan]Starting Model Comparison[/bold cyan]")
    console.print(f"[dim]Model A: {model_a_name}[/dim]")
    console.print(f"[dim]Model B: {model_b_name}[/dim]")
    console.print(f"[dim]Prompts: {len(prompts)}[/dim]\n")

    # Load models if not provided
    if model_a is None:
        console.print(f"[bold blue]Loading Model A: {model_a_path or model_a_name}[/bold blue]")
        model_a = DiogenesModel.from_pretrained(
            model_name_or_path=model_a_path or model_a_name,
            use_4bit=use_4bit,
        )

    if model_b is None:
        console.print(f"[bold blue]Loading Model B: {model_b_path or model_b_name}[/bold blue]")
        model_b = DiogenesModel.from_pretrained(
            model_name_or_path=model_b_path or model_b_name,
            use_4bit=use_4bit,
        )

    # Create inference engines
    inference_a = DiogenesInference(model=model_a, default_max_length=max_length, default_temperature=temperature)
    inference_b = DiogenesInference(model=model_b, default_max_length=max_length, default_temperature=temperature)

    # Run comparison
    detailed_results: list[ComparisonResult] = []
    mode_dist_a: dict[str, int] = {}
    mode_dist_b: dict[str, int] = {}
    total_conf_a = 0.0
    total_conf_b = 0.0
    total_latency_a = 0.0
    total_latency_b = 0.0
    wins_a = 0
    wins_b = 0
    ties = 0

    for i, prompt in enumerate(prompts):
        console.print(f"[dim]Prompt {i + 1}/{len(prompts)}[/dim]")

        # Run both models
        start_a = time.time()
        result_a: InferenceResult = inference_a.generate(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            return_logprobs=True,
        )
        latency_a = (time.time() - start_a) * 1000

        start_b = time.time()
        result_b: InferenceResult = inference_b.generate(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            return_logprobs=True,
        )
        latency_b = (time.time() - start_b) * 1000

        # Determine winner
        winner = _determine_winner(
            result_a=result_a,
            result_b=result_b,
            auto_judge=auto_judge,
        )

        if winner == "A":
            wins_a += 1
        elif winner == "B":
            wins_b += 1
        else:
            ties += 1

        # Update statistics
        mode_dist_a[result_a.epistemic_mode.value] = mode_dist_a.get(result_a.epistemic_mode.value, 0) + 1
        mode_dist_b[result_b.epistemic_mode.value] = mode_dist_b.get(result_b.epistemic_mode.value, 0) + 1
        total_conf_a += result_a.confidence
        total_conf_b += result_b.confidence
        total_latency_a += latency_a
        total_latency_b += latency_b

        # Create comparison result
        comp_result = ComparisonResult(
            prompt=prompt,
            model_a_response=result_a.text,
            model_b_response=result_b.text,
            model_a_mode=result_a.epistemic_mode.value,
            model_b_mode=result_b.epistemic_mode.value,
            model_a_confidence=result_a.confidence,
            model_b_confidence=result_b.confidence,
            model_a_latency_ms=latency_a,
            model_b_latency_ms=latency_b,
            winner=winner,
            delta_confidence=result_a.confidence - result_b.confidence,
            delta_latency_ms=latency_a - latency_b,
        )
        detailed_results.append(comp_result)

        # Save to storage if path provided
        if storage_path:
            _save_comparison_results(
                storage_path=storage_path,
                result=comp_result,
                model_a_name=model_a_name,
                model_b_name=model_b_name,
            )

    # Create summary
    n = len(prompts)
    summary = ComparisonSummary(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        total_prompts=n,
        model_a_wins=wins_a,
        model_b_wins=wins_b,
        ties=ties,
        model_a_avg_confidence=total_conf_a / n if n > 0 else 0.0,
        model_b_avg_confidence=total_conf_b / n if n > 0 else 0.0,
        model_a_avg_latency_ms=total_latency_a / n if n > 0 else 0.0,
        model_b_avg_latency_ms=total_latency_b / n if n > 0 else 0.0,
        mode_distribution_a=mode_dist_a,
        mode_distribution_b=mode_dist_b,
        detailed_results=detailed_results,
    )

    # Display summary
    _display_comparison_summary(summary)

    return summary


def _determine_winner(
    result_a: InferenceResult,
    result_b: InferenceResult,
    auto_judge: bool = False,
) -> str:
    """Determine the winner between two model responses.

    Args:
        result_a: Result from model A
        result_b: Result from model B
        auto_judge: Use automatic judging based on confidence

    Returns:
        'A', 'B', or 'tie'
    """
    if auto_judge:
        # Simple confidence-based judging
        if result_a.confidence > result_b.confidence + 0.05:
            return "A"
        elif result_b.confidence > result_a.confidence + 0.05:
            return "B"
        else:
            return "tie"

    # For manual judging, we default to tie
    # In a full implementation, this would prompt the user
    return "tie"


def _save_comparison_results(
    storage_path: str,
    result: ComparisonResult,
    model_a_name: str,
    model_b_name: str,
) -> None:
    """Save comparison results to storage.

    Args:
        storage_path: Path to storage
        result: ComparisonResult to save
        model_a_name: Name of model A
        model_b_name: Name of model B
    """
    import uuid

    storage = TestStorage(storage_path)

    # Save model A result
    storage.save(
        TestResult(
            test_id=str(uuid.uuid4()),
            prompt=result.prompt,
            response=result.model_a_response,
            epistemic_mode=result.model_a_mode,
            confidence=result.model_a_confidence,
            model_name=model_a_name,
            latency_ms=result.model_a_latency_ms,
            tags=["comparison", "model_a"],
        )
    )

    # Save model B result
    storage.save(
        TestResult(
            test_id=str(uuid.uuid4()),
            prompt=result.prompt,
            response=result.model_b_response,
            epistemic_mode=result.model_b_mode,
            confidence=result.model_b_confidence,
            model_name=model_b_name,
            latency_ms=result.model_b_latency_ms,
            tags=["comparison", "model_b"],
        )
    )

    storage.close()


def _display_comparison_summary(summary: ComparisonSummary) -> None:
    """Display comparison summary.

    Args:
        summary: ComparisonSummary to display
    """
    # Overall results panel
    results_text = Text()
    results_text.append(f"Total Prompts: {summary.total_prompts}\n", style="bold")
    results_text.append(f"{summary.model_a_name} Wins: ", style="bold")
    results_text.append(f"{summary.model_a_wins} ({summary.model_a_win_rate:.1%})\n", style="green")
    results_text.append(f"{summary.model_b_name} Wins: ", style="bold")
    results_text.append(f"{summary.model_b_wins} ({summary.model_b_win_rate:.1%})\n", style="blue")
    results_text.append(f"Ties: ", style="bold")
    results_text.append(f"{summary.ties} ({summary.tie_rate:.1%})\n", style="yellow")

    console.print(
        Panel(
            results_text,
            title="[bold]Comparison Results[/bold]",
            border_style="cyan",
        )
    )

    # Metrics comparison table
    table = Table(title="Performance Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column(summary.model_a_name, style="green")
    table.add_column(summary.model_b_name, style="blue")
    table.add_column("Delta", style="yellow")

    table.add_row(
        "Avg Confidence",
        f"{summary.model_a_avg_confidence:.4f}",
        f"{summary.model_b_avg_confidence:.4f}",
        f"{summary.model_a_avg_confidence - summary.model_b_avg_confidence:+.4f}",
    )
    table.add_row(
        "Avg Latency (ms)",
        f"{summary.model_a_avg_latency_ms:.2f}",
        f"{summary.model_b_avg_latency_ms:.2f}",
        f"{summary.model_a_avg_latency_ms - summary.model_b_avg_latency_ms:+.2f}",
    )

    console.print(table)

    # Mode distribution comparison
    all_modes = set(summary.mode_distribution_a.keys()) | set(summary.mode_distribution_b.keys())

    mode_table = Table(title="Epistemic Mode Distribution", show_header=True, header_style="bold magenta")
    mode_table.add_column("Mode", style="cyan")
    mode_table.add_column(summary.model_a_name, style="green")
    mode_table.add_column(summary.model_b_name, style="blue")

    for mode in sorted(all_modes):
        count_a = summary.mode_distribution_a.get(mode, 0)
        count_b = summary.mode_distribution_b.get(mode, 0)
        mode_table.add_row(mode, str(count_a), str(count_b))

    console.print(mode_table)


def side_by_side_display(result: ComparisonResult) -> None:
    """Display side-by-side comparison of a single result.

    Args:
        result: ComparisonResult to display
    """
    from rich.columns import Columns
    from rich.panel import Panel

    panel_a = Panel(
        result.model_a_response,
        title=f"Model A ({result.model_a_mode})",
        border_style="green",
    )
    panel_b = Panel(
        result.model_b_response,
        title=f"Model B ({result.model_b_mode})",
        border_style="blue",
    )

    console.print(Columns([panel_a, panel_b]))

    # Winner announcement
    if result.winner:
        winner_style = "green" if result.winner == "A" else "blue" if result.winner == "B" else "yellow"
        winner_name = "Model A" if result.winner == "A" else "Model B" if result.winner == "B" else "Tie"
        console.print(f"[{winner_style}]Winner: {winner_name}[/{winner_style}]")
        console.print()
