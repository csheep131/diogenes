"""Report generation command for test results."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from diogenes.testing.core.storage import TestResult, TestStorage


logger = logging.getLogger(__name__)
console = Console()


def generate_report(
    storage_path: str,
    output_path: str,
    format: str = "markdown",
    suite_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_plots: bool = True,
) -> str:
    """Generate a report from test results.

    Args:
        storage_path: Path to test results storage
        output_path: Path for output report
        format: Output format ('markdown', 'html', 'json')
        suite_name: Filter by suite name
        start_date: Filter by start date (ISO format)
        end_date: Filter by end date (ISO format)
        include_plots: Include plotly visualizations

    Returns:
        Path to generated report
    """
    # Load results
    storage = TestStorage(storage_path)
    results = storage.load(
        suite_name=suite_name,
        start_date=start_date,
        end_date=end_date,
    )
    storage.close()

    if not results:
        console.print("[yellow]No results found for the specified filters[/yellow]")
        return ""

    console.print(f"[bold green]Generating report from {len(results)} results...[/bold green]")

    # Generate report based on format
    if format == "markdown":
        return _generate_markdown_report(results, output_path, include_plots)
    elif format == "html":
        return _generate_html_report(results, output_path, include_plots)
    elif format == "json":
        return _generate_json_report(results, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")


def _generate_markdown_report(
    results: list[TestResult],
    output_path: str,
    include_plots: bool = True,
) -> str:
    """Generate a Markdown report.

    Args:
        results: List of TestResult objects
        output_path: Path for output file
        include_plots: Include plot references

    Returns:
        Path to generated report
    """
    # Calculate statistics
    stats = _calculate_statistics(results)

    # Build report
    lines = []
    lines.append("# Diogenes Model Test Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.utcnow().isoformat()}")
    lines.append(f"**Results Analyzed:** {len(results)}")
    if stats.get("suite_name"):
        lines.append(f"**Suite:** {stats['suite_name']}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Average Confidence:** {stats['avg_confidence']:.4f}")
    lines.append(f"- **Average Latency:** {stats['avg_latency_ms']:.2f} ms")
    lines.append(f"- **Total Tokens:** {stats['total_tokens']}")
    if stats.get("accuracy") is not None:
        lines.append(f"- **Accuracy:** {stats['accuracy']:.2%}")
    lines.append("")

    # Epistemic Mode Distribution
    lines.append("## Epistemic Mode Distribution")
    lines.append("")
    lines.append("| Mode | Count | Percentage |")
    lines.append("|------|-------|------------|")
    for mode, count in sorted(stats["mode_distribution"].items(), key=lambda x: -x[1]):
        percentage = count / len(results) * 100
        lines.append(f"| {mode} | {count} | {percentage:.1f}% |")
    lines.append("")

    # Confidence Distribution
    lines.append("## Confidence Distribution")
    lines.append("")
    lines.append(f"- **Min:** {stats['confidence_min']:.4f}")
    lines.append(f"- **Max:** {stats['confidence_max']:.4f}")
    lines.append(f"- **Mean:** {stats['avg_confidence']:.4f}")
    lines.append(f"- **Std Dev:** {stats['confidence_std']:.4f}")
    lines.append("")

    # Latency Distribution
    lines.append("## Latency Distribution")
    lines.append("")
    lines.append(f"- **Min:** {stats['latency_min']:.2f} ms")
    lines.append(f"- **Max:** {stats['latency_max']:.2f} ms")
    lines.append(f"- **Mean:** {stats['avg_latency_ms']:.2f} ms")
    lines.append(f"- **P95:** {stats['latency_p95']:.2f} ms")
    lines.append("")

    # Model Information
    if results[0].model_name:
        lines.append("## Model Information")
        lines.append("")
        lines.append(f"- **Model Name:** {results[0].model_name}")
        lines.append(f"- **Temperature:** {results[0].temperature}")
        lines.append(f"- **Max Length:** {results[0].max_length}")
        lines.append("")

    # Detailed Results (sample)
    lines.append("## Sample Results")
    lines.append("")
    lines.append("### First 10 Results")
    lines.append("")

    for i, result in enumerate(results[:10], 1):
        lines.append(f"#### Result {i}")
        lines.append("")
        lines.append(f"**Prompt:** {result.prompt[:200]}{'...' if len(result.prompt) > 200 else ''}")
        lines.append("")
        lines.append(f"**Response:** {result.response[:300]}{'...' if len(result.response) > 300 else ''}")
        lines.append("")
        lines.append(f"- Mode: `{result.epistemic_mode}`")
        lines.append(f"- Confidence: {result.confidence:.4f}")
        lines.append(f"- Latency: {result.latency_ms:.2f} ms")
        lines.append("")

    # Plots section
    if include_plots:
        lines.append("## Visualizations")
        lines.append("")
        lines.append("See accompanying HTML report for interactive visualizations.")
        lines.append("")

    # Write report
    output_file = Path(output_path)
    if not output_file.suffix:
        output_file = output_file / "report.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"[bold green]Report saved to: {output_file}[/bold green]")
    return str(output_file)


def _generate_html_report(
    results: list[TestResult],
    output_path: str,
    include_plots: bool = True,
) -> str:
    """Generate an HTML report with Plotly visualizations.

    Args:
        results: List of TestResult objects
        output_path: Path for output file
        include_plots: Include Plotly plots

    Returns:
        Path to generated report
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    # Calculate statistics
    stats = _calculate_statistics(results)

    # Create plots
    plots_html = []

    if include_plots:
        # Mode distribution pie chart
        fig_mode = go.Figure(
            data=[
                go.Pie(
                    labels=list(stats["mode_distribution"].keys()),
                    values=list(stats["mode_distribution"].values()),
                    hole=0.3,
                )
            ]
        )
        fig_mode.update_layout(title="Epistemic Mode Distribution")
        plots_html.append(fig_mode.to_html(full_html=False, include_plotlyjs="cdn"))

        # Confidence histogram
        confidences = [r.confidence for r in results]
        fig_conf = go.Figure(
            data=[go.Histogram(x=confidences, nbinsx=20, marker_color="steelblue")]
        )
        fig_conf.update_layout(title="Confidence Distribution", xaxis_title="Confidence", yaxis_title="Count")
        plots_html.append(fig_conf.to_html(full_html=False, include_plotlyjs="cdn"))

        # Latency box plot
        latencies = [r.latency_ms for r in results]
        fig_lat = go.Figure(data=[go.Box(y=latencies, name="Latency (ms)")])
        fig_lat.update_layout(title="Latency Distribution")
        plots_html.append(fig_lat.to_html(full_html=False, include_plotlyjs="cdn"))

        # Confidence over time (if timestamps vary)
        if len(results) > 1:
            timestamps = [r.timestamp for r in results]
            fig_time = go.Figure(
                data=[go.Scatter(x=timestamps, y=confidences, mode="markers+lines")]
            )
            fig_time.update_layout(title="Confidence Over Time", xaxis_title="Time", yaxis_title="Confidence")
            plots_html.append(fig_time.to_html(full_html=False, include_plotlyjs="cdn"))

    # Build HTML
    html_parts = []
    html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Diogenes Model Test Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .stat-label { color: #666; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: 600; }
        .plot-container { margin: 30px 0; }
    </style>
</head>
<body>
    <div class="container">
    """)

    # Header
    html_parts.append(f"<h1>Diogenes Model Test Report</h1>")
    html_parts.append(f"<p><strong>Generated:</strong> {datetime.utcnow().isoformat()}</p>")
    html_parts.append(f"<p><strong>Results Analyzed:</strong> {len(results)}</p>")

    # Stats grid
    html_parts.append("<h2>Executive Summary</h2>")
    html_parts.append('<div class="stats-grid">')
    html_parts.append(f'<div class="stat-card"><div class="stat-value">{stats["avg_confidence"]:.4f}</div><div class="stat-label">Avg Confidence</div></div>')
    html_parts.append(f'<div class="stat-card"><div class="stat-value">{stats["avg_latency_ms"]:.1f} ms</div><div class="stat-label">Avg Latency</div></div>')
    html_parts.append(f'<div class="stat-card"><div class="stat-value">{len(results)}</div><div class="stat-label">Total Tests</div></div>')
    if stats.get("accuracy") is not None:
        html_parts.append(f'<div class="stat-card"><div class="stat-value">{stats["accuracy"]:.1%}</div><div class="stat-label">Accuracy</div></div>')
    html_parts.append("</div>")

    # Mode distribution table
    html_parts.append("<h2>Epistemic Mode Distribution</h2>")
    html_parts.append("<table><thead><tr><th>Mode</th><th>Count</th><th>Percentage</th></tr></thead><tbody>")
    for mode, count in sorted(stats["mode_distribution"].items(), key=lambda x: -x[1]):
        percentage = count / len(results) * 100
        html_parts.append(f"<tr><td>{mode}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>")
    html_parts.append("</tbody></table>")

    # Plots
    if plots_html:
        html_parts.append("<h2>Visualizations</h2>")
        for plot in plots_html:
            html_parts.append(f'<div class="plot-container">{plot}</div>')

    # Sample results
    html_parts.append("<h2>Sample Results</h2>")
    html_parts.append("<details><summary>Show first 10 results</summary>")
    for i, result in enumerate(results[:10], 1):
        html_parts.append(f"<h3>Result {i}</h3>")
        html_parts.append(f"<p><strong>Prompt:</strong> {result.prompt[:300]}{'...' if len(result.prompt) > 300 else ''}</p>")
        html_parts.append(f"<p><strong>Response:</strong> {result.response[:500]}{'...' if len(result.response) > 500 else ''}</p>")
        html_parts.append(f"<p><em>Mode:</em> {result.epistemic_mode} | <em>Confidence:</em> {result.confidence:.4f} | <em>Latency:</em> {result.latency_ms:.2f} ms</p>")
    html_parts.append("</details>")

    # Footer
    html_parts.append("""
    </div>
</body>
</html>
    """)

    # Write report
    output_file = Path(output_path)
    if not output_file.suffix:
        output_file = output_file / "report.html"
    elif output_file.suffix != ".html":
        output_file = output_file.with_suffix(".html")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    console.print(f"[bold green]HTML report saved to: {output_file}[/bold green]")
    return str(output_file)


def _generate_json_report(results: list[TestResult], output_path: str) -> str:
    """Generate a JSON report.

    Args:
        results: List of TestResult objects
        output_path: Path for output file

    Returns:
        Path to generated report
    """
    # Calculate statistics
    stats = _calculate_statistics(results)

    # Build report
    report = {
        "metadata": {
            "generated": datetime.utcnow().isoformat(),
            "total_results": len(results),
            "suite_name": stats.get("suite_name"),
        },
        "statistics": stats,
        "results": [r.to_dict() for r in results],
    }

    # Write report
    output_file = Path(output_path)
    if not output_file.suffix:
        output_file = output_file / "report.json"
    elif output_file.suffix != ".json":
        output_file = output_file.with_suffix(".json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    console.print(f"[bold green]JSON report saved to: {output_file}[/bold green]")
    return str(output_file)


def _calculate_statistics(results: list[TestResult]) -> dict[str, Any]:
    """Calculate statistics from test results.

    Args:
        results: List of TestResult objects

    Returns:
        Dictionary with calculated statistics
    """
    import numpy as np

    confidences = [r.confidence for r in results]
    latencies = [r.latency_ms for r in results]
    mode_counts: dict[str, int] = {}
    total_tokens = 0
    correct_count = 0
    evaluated_count = 0

    for result in results:
        mode_counts[result.epistemic_mode] = mode_counts.get(result.epistemic_mode, 0) + 1
        total_tokens += result.token_count

        if result.is_correct is not None:
            evaluated_count += 1
            if result.is_correct:
                correct_count += 1

    stats = {
        "avg_confidence": np.mean(confidences) if confidences else 0.0,
        "confidence_min": min(confidences) if confidences else 0.0,
        "confidence_max": max(confidences) if confidences else 0.0,
        "confidence_std": np.std(confidences) if confidences else 0.0,
        "avg_latency_ms": np.mean(latencies) if latencies else 0.0,
        "latency_min": min(latencies) if latencies else 0.0,
        "latency_max": max(latencies) if latencies else 0.0,
        "latency_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "mode_distribution": mode_counts,
        "total_tokens": total_tokens,
        "suite_name": results[0].suite_name if results else None,
    }

    if evaluated_count > 0:
        stats["accuracy"] = correct_count / evaluated_count

    return stats


def export_results(
    storage_path: str,
    output_path: str,
    format: str = "csv",
    **filters,
) -> str:
    """Export test results to a file.

    Args:
        storage_path: Path to test results storage
        output_path: Path for output file
        format: Export format ('csv', 'json', 'parquet')
        **filters: Filter arguments for loading results

    Returns:
        Path to exported file
    """
    import pandas as pd

    # Load results
    storage = TestStorage(storage_path)
    results = storage.load(**filters)
    storage.close()

    if not results:
        console.print("[yellow]No results found for export[/yellow]")
        return ""

    # Convert to DataFrame
    data = [r.to_dict() for r in results]
    df = pd.DataFrame(data)

    # Export based on format
    output_file = Path(output_path)

    if format == "csv":
        if not output_file.suffix:
            output_file = output_file.with_suffix(".csv")
        df.to_csv(output_file, index=False)
    elif format == "json":
        if not output_file.suffix:
            output_file = output_file.with_suffix(".json")
        df.to_json(output_file, orient="records", indent=2)
    elif format == "parquet":
        if not output_file.suffix:
            output_file = output_file.with_suffix(".parquet")
        df.to_parquet(output_file, index=False)
    else:
        raise ValueError(f"Unknown export format: {format}")

    console.print(f"[bold green]Results exported to: {output_file}[/bold green]")
    return str(output_file)
