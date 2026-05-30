"""Display utilities for meta-rubric evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from autorubric.types import CriterionVerdict, EnsembleEvaluationReport

if TYPE_CHECKING:
    from autorubric.rubric import Rubric

    from ._improve import IterationResult

DisplayMode = Literal["stdout", "html"]


def _load_section_mapping(meta_rubric_path: Path) -> dict[str, str]:
    """Load criterion name to section name mapping from meta rubric JSON."""
    with open(meta_rubric_path, encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for section in data.get("rubric", {}).get("sections", []):
        section_name = section.get("name", "Unknown")
        for criterion in section.get("criteria", []):
            crit_name = criterion.get("name")
            if crit_name:
                mapping[crit_name] = section_name
    return mapping


def _error_category(error: str | None) -> str:
    """Extract the leading category from an error string (e.g. "infrastructure")."""
    if not error:
        return "unknown"
    return error.split(":", 1)[0].strip() or "unknown"


def _get_verdict_display(
    verdict: CriterionVerdict | None, weight: float, error: str | None = None
) -> tuple[str, str]:
    """Get verdict text and color based on verdict and weight.

    For positive weights: MET is good (green), UNMET is bad (red)
    For negative weights (anti-patterns): MET is bad (red), UNMET is good (green)

    When ``error`` is set, the verdict was synthesized from a failed judge call;
    surface a distinct ERROR badge (bright_magenta) so it is visually separate from
    a genuine N/A.
    """
    if error is not None:
        return f"ERROR ({_error_category(error)})", "bright_magenta"

    if verdict is None or verdict == CriterionVerdict.CANNOT_ASSESS:
        return "N/A", "yellow"

    is_met = verdict == CriterionVerdict.MET
    is_anti_pattern = weight < 0

    if is_anti_pattern:
        if is_met:
            return "DETECTED", "red"
        else:
            return "CLEAR", "green"
    else:
        if is_met:
            return "MET", "green"
        else:
            return "UNMET", "red"


def _group_criteria_by_section(
    result: EnsembleEvaluationReport,
    section_mapping: dict[str, str],
) -> tuple[dict[str, list], list[tuple[str, str, str]]]:
    """Group criteria by section and collect issues."""
    sections: dict[str, list] = {}
    issues_found: list[tuple[str, str, str]] = []

    for cr in result.report or []:
        crit_name = cr.criterion.name or "unnamed"
        section_name = section_mapping.get(crit_name, "Other")
        if section_name not in sections:
            sections[section_name] = []
        sections[section_name].append(cr)

        verdict = cr.final_verdict
        weight = cr.criterion.weight
        is_issue = (weight > 0 and verdict == CriterionVerdict.UNMET) or (
            weight < 0 and verdict == CriterionVerdict.MET
        )
        if is_issue and cr.final_reason:
            issues_found.append((crit_name, section_name, cr.final_reason))

    return sections, issues_found


def display_to_stdout(
    result: EnsembleEvaluationReport,
    meta_rubric_path: Path,
    title: str,
    rubric_summary: str | None = None,
) -> None:
    """Display meta-rubric evaluation results to stdout using rich formatting."""
    console = Console()

    header_text = Text()
    header_text.append(f"{title}\n", style="bold white")
    if rubric_summary:
        header_text.append(f"\n{rubric_summary}", style="dim")

    console.print(Panel(header_text, border_style="blue"))
    console.print()

    score_table = Table(show_header=False, box=None, padding=(0, 2))
    score_table.add_column("Label", style="bold")
    score_table.add_column("Value")

    # A grade-FAILURE has no score (score=None): render "n/a" with a neutral/red color.
    if result.score is None:
        score_color = "red"
        score_text = "n/a"
    else:
        score_color = "green" if result.score >= 0.7 else "yellow" if result.score >= 0.5 else "red"
        score_text = f"{result.score:.2f}"
    score_table.add_row("Score", Text(score_text, style=score_color))
    raw_score_text = "n/a" if result.raw_score is None else f"{result.raw_score:.2f}"
    score_table.add_row("Raw Score", raw_score_text)
    if result.completion_cost:
        score_table.add_row("Cost", f"${result.completion_cost:.6f}")

    console.print(Panel(score_table, title="Results", border_style="cyan"))
    console.print()

    section_mapping = _load_section_mapping(meta_rubric_path)
    sections, issues_found = _group_criteria_by_section(result, section_mapping)

    for section_name, criteria in sections.items():
        is_anti_pattern_section = "Anti-Pattern" in section_name

        table = Table(
            title=section_name,
            title_style="bold magenta" if is_anti_pattern_section else "bold blue",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Criterion", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Weight", justify="right")

        for cr in criteria:
            verdict = cr.final_verdict
            weight = cr.criterion.weight
            name = cr.criterion.name or "unnamed"

            error = getattr(cr, "error", None)
            status_text, status_color = _get_verdict_display(verdict, weight, error)
            weight_str = f"{'+' if weight > 0 else ''}{weight}"

            table.add_row(
                name,
                Text(status_text, style=status_color),
                weight_str,
            )

        console.print(table)
        console.print()

    if issues_found:
        console.print(
            Panel(
                Text("Issues Found", style="bold"),
                border_style="red",
                subtitle="Feedback for rubric improvement",
            )
        )
        console.print()

        for name, section, reason in issues_found:
            issue_panel = Panel(
                Text(reason, style="white"),
                title=f"[bold cyan]{name}[/bold cyan] [dim]({section})[/dim]",
                border_style="red" if "Anti-Pattern" in section else "yellow",
            )
            console.print(issue_panel)
            console.print()
    else:
        console.print(
            Panel(
                Text("No issues found!", style="bold green"),
                border_style="green",
            )
        )


def _color_to_css(color: str) -> str:
    """Convert rich color names to CSS colors."""
    color_map = {
        "green": "#22c55e",
        "red": "#ef4444",
        "yellow": "#eab308",
        "cyan": "#06b6d4",
        "magenta": "#d946ef",
        "blue": "#3b82f6",
        "white": "#f8fafc",
        "dim": "#94a3b8",
    }
    return color_map.get(color, color)


def render_to_html(
    result: EnsembleEvaluationReport,
    meta_rubric_path: Path,
    title: str,
    rubric_summary: str | None = None,
) -> str:
    """Render meta-rubric evaluation results as styled HTML."""
    section_mapping = _load_section_mapping(meta_rubric_path)
    sections, issues_found = _group_criteria_by_section(result, section_mapping)

    # A grade-FAILURE has no score (score=None): render "n/a" with a neutral/red color.
    if result.score is None:
        score_color = _color_to_css("red")
        score_text = "n/a"
    else:
        score_color = (
            _color_to_css("green")
            if result.score >= 0.7
            else _color_to_css("yellow")
            if result.score >= 0.5
            else _color_to_css("red")
        )
        score_text = f"{result.score:.2f}"
    raw_score_text = "n/a" if result.raw_score is None else f"{result.raw_score:.2f}"

    html_parts = [
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meta-Rubric Evaluation Report</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #475569;
            --green: #22c55e;
            --red: #ef4444;
            --yellow: #eab308;
            --cyan: #06b6d4;
            --magenta: #d946ef;
            --blue: #3b82f6;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
            max-width: 900px;
            margin: 0 auto;
        }
        .panel {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 1.5rem;
            overflow: hidden;
        }
        .panel-header {
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-secondary);
        }
        .panel-title {
            font-size: 1.25rem;
            font-weight: 600;
        }
        .panel-subtitle {
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }
        .panel-body { padding: 1.5rem; }
        .results-grid {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 0.5rem 1.5rem;
        }
        .results-label { font-weight: 600; }
        .section-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1.5rem;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            padding: 0.75rem 1rem;
            background: var(--bg-secondary);
            border-radius: 6px 6px 0 0;
        }
        .section-title.anti-pattern { color: var(--magenta); }
        .section-title.normal { color: var(--blue); }
        .section-table th, .section-table td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        .section-table th {
            background: var(--bg-tertiary);
            font-weight: 600;
        }
        .section-table td:first-child { color: var(--cyan); }
        .section-table td:nth-child(2) { text-align: center; }
        .section-table td:nth-child(3) { text-align: right; }
        .status { font-weight: 600; padding: 0.25rem 0.75rem; border-radius: 4px; }
        .status-met { color: var(--green); background: rgba(34, 197, 94, 0.1); }
        .status-unmet { color: var(--red); background: rgba(239, 68, 68, 0.1); }
        .status-detected { color: var(--red); background: rgba(239, 68, 68, 0.1); }
        .status-clear { color: var(--green); background: rgba(34, 197, 94, 0.1); }
        .status-na { color: var(--yellow); background: rgba(234, 179, 8, 0.1); }
        .error-badge {
            font-weight: 600;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            color: var(--magenta);
            background: rgba(217, 70, 239, 0.12);
            border: 1px solid var(--magenta);
        }
        .issues-header {
            background: var(--bg-secondary);
            border-left: 4px solid var(--red);
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            border-radius: 0 6px 6px 0;
        }
        .issues-header h3 { color: var(--red); }
        .issues-header p { color: var(--text-secondary); font-size: 0.875rem; }
        .issue-card {
            border: 1px solid var(--border-color);
            border-radius: 6px;
            margin-bottom: 1rem;
            overflow: hidden;
        }
        .issue-card.anti-pattern { border-left: 4px solid var(--red); }
        .issue-card.normal { border-left: 4px solid var(--yellow); }
        .issue-card-header {
            padding: 0.75rem 1rem;
            background: var(--bg-secondary);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .issue-name { color: var(--cyan); font-weight: 600; }
        .issue-section { color: var(--text-secondary); font-size: 0.875rem; }
        .issue-reason { padding: 1rem; white-space: pre-wrap; }
        .success-panel {
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid var(--green);
            border-radius: 6px;
            padding: 1.5rem;
            text-align: center;
            color: var(--green);
            font-weight: 600;
        }
    </style>
</head>
<body>
""",
    ]

    subtitle_html = ""
    if rubric_summary:
        subtitle_html = f"<div class='panel-subtitle'>{_escape_html(rubric_summary)}</div>"

    cost_html = ""
    if result.completion_cost:
        cost_html = (
            f"<span class='results-label'>Cost</span><span>${result.completion_cost:.6f}</span>"
        )

    html_parts.append(f"""
    <div class="panel" style="border-color: var(--blue);">
        <div class="panel-body">
            <div class="panel-title">{_escape_html(title)}</div>
            {subtitle_html}
        </div>
    </div>

    <div class="panel" style="border-color: var(--cyan);">
        <div class="panel-header">
            <div class="panel-title">Results</div>
        </div>
        <div class="panel-body">
            <div class="results-grid">
                <span class="results-label">Score</span>
                <span style="color: {score_color}; font-weight: 600;">{score_text}</span>
                <span class="results-label">Raw Score</span>
                <span>{raw_score_text}</span>
                {cost_html}
            </div>
        </div>
    </div>
""")

    for section_name, criteria in sections.items():
        is_anti_pattern = "Anti-Pattern" in section_name
        section_class = "anti-pattern" if is_anti_pattern else "normal"

        html_parts.append(f"""
    <div class="section-title {section_class}">{_escape_html(section_name)}</div>
    <table class="section-table">
        <thead>
            <tr><th>Criterion</th><th>Status</th><th>Weight</th></tr>
        </thead>
        <tbody>
""")

        for cr in criteria:
            verdict = cr.final_verdict
            weight = cr.criterion.weight
            name = cr.criterion.name or "unnamed"
            error = getattr(cr, "error", None)
            status_text, status_color = _get_verdict_display(verdict, weight, error)
            if error is not None:
                status_class = "status-error"
            else:
                status_class = f"status-{status_text.lower()}"
            weight_str = f"{'+' if weight > 0 else ''}{weight}"

            if error is not None:
                badge_html = f'<span class="error-badge">{_escape_html(status_text)}</span>'
            else:
                badge_html = f'<span class="status {status_class}">{status_text}</span>'

            html_parts.append(f"""
            <tr>
                <td>{_escape_html(name)}</td>
                <td>{badge_html}</td>
                <td>{weight_str}</td>
            </tr>
""")

        html_parts.append("        </tbody>\n    </table>\n")

    if issues_found:
        html_parts.append("""
    <div class="issues-header">
        <h3>Issues Found</h3>
        <p>Feedback for rubric improvement</p>
    </div>
""")
        for name, section, reason in issues_found:
            card_class = "anti-pattern" if "Anti-Pattern" in section else "normal"
            html_parts.append(f"""
    <div class="issue-card {card_class}">
        <div class="issue-card-header">
            <span class="issue-name">{_escape_html(name)}</span>
            <span class="issue-section">{_escape_html(section)}</span>
        </div>
        <div class="issue-reason">{_escape_html(reason)}</div>
    </div>
""")
    else:
        html_parts.append("""
    <div class="success-panel">No issues found!</div>
""")

    html_parts.append("""
</body>
</html>
""")

    return "".join(html_parts)


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _fmt_pct(value: float | None) -> str:
    """Format a fraction as a percent, rendering an undefined (None) value as n/a."""
    return "n/a" if value is None else f"{value:.0%}"


def _fmt_num(value: float | None) -> str:
    """Format a scalar to 2 decimals, rendering an undefined (None) value as n/a."""
    return "n/a" if value is None else f"{value:.2f}"


def _fmt_int(value: int | None) -> str:
    """Format an integer count, rendering a missing (None) value as n/a."""
    return "n/a" if value is None else str(value)


def display_meta_rubric_result(
    result: EnsembleEvaluationReport,
    meta_rubric_path: Path,
    title: str,
    rubric_summary: str | None = None,
    *,
    mode: DisplayMode = "stdout",
    output_html_path: Path | str | None = None,
) -> None:
    """Display meta-rubric evaluation results.

    Args:
        result: The evaluation result to display.
        meta_rubric_path: Path to the meta-rubric JSON file (for section mapping).
        title: Title for the report.
        rubric_summary: Optional summary text.
        mode: Display mode - "stdout" for terminal, "html" for HTML file.
        output_html_path: Path for HTML output (required when mode="html").
    """
    if mode == "stdout":
        display_to_stdout(result, meta_rubric_path, title, rubric_summary)
    elif mode == "html":
        if output_html_path is None:
            raise ValueError("output_html_path is required when mode='html'")
        html_content = render_to_html(result, meta_rubric_path, title, rubric_summary)
        output_path = Path(output_html_path)
        output_path.write_text(html_content, encoding="utf-8")


def render_improvement_report_html(
    iterations: list[IterationResult],
    convergence_reason: str,
    total_cost: float,
    original_rubric: Rubric,
    final_rubric: Rubric,
) -> str:
    """Render a consolidated HTML improvement report with Bootstrap accordion.

    Args:
        iterations: All iteration results from the improvement loop.
        convergence_reason: Why the loop stopped.
        total_cost: Total cost across all iterations.
        original_rubric: The rubric before improvements.
        final_rubric: The best/final rubric after improvements.

    Returns:
        Complete HTML string for the report.
    """
    parts: list[str] = [
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rubric Improvement Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
          rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #475569;
            --green: #22c55e;
            --red: #ef4444;
            --yellow: #eab308;
            --cyan: #06b6d4;
            --blue: #3b82f6;
        }
        * { box-sizing: border-box; }
        body {
            font-family: system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
            max-width: 960px;
            margin: 0 auto;
        }
        h1 { color: var(--text-primary); margin-bottom: 0.5rem; }
        .stats { color: var(--text-secondary); margin-bottom: 2rem; }
        .stats span { margin-right: 1.5rem; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 1rem; }
        th, td {
            padding: 0.5rem 0.75rem;
            border-bottom: 1px solid var(--border-color);
            text-align: left;
        }
        th { background: var(--bg-tertiary); font-weight: 600; }
        td { background: var(--bg-secondary); }
        .badge-accepted { color: var(--green); }
        .badge-rejected { color: var(--red); }
        .accordion-button {
            background: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }
        .accordion-button:not(.collapsed) {
            background: var(--bg-tertiary) !important;
        }
        .accordion-button::after {
            filter: invert(1);
        }
        .accordion-body {
            background: var(--bg-primary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-top: none;
        }
        .metrics-bar {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 1rem;
            color: var(--text-secondary);
        }
        .metrics-bar strong { color: var(--text-primary); }
        .issue-anti { color: var(--red); }
        .issue-gap { color: var(--yellow); }
        .weight-col { color: #d946ef; }
        .crit-name { color: var(--cyan); }
        .final-panel {
            border: 2px solid var(--blue);
            border-radius: 8px;
            padding: 1.5rem;
            margin-top: 2rem;
        }
        .final-panel h2 { color: var(--blue); }
        a { color: var(--cyan); }
    </style>
</head>
<body>
"""
    ]

    # Header
    parts.append("<h1>Rubric Improvement Report</h1>\n")
    parts.append("<div class='stats'>")
    parts.append(f"<span><strong>Iterations:</strong> {len(iterations)}</span>")
    parts.append(f"<span><strong>Convergence:</strong> {_escape_html(convergence_reason)}</span>")
    if total_cost > 0:
        parts.append(f"<span><strong>Total cost:</strong> ${total_cost:.4f}</span>")
    parts.append("</div>\n")

    # Summary table
    parts.append("<h2>Summary</h2>\n<table>\n<thead><tr>")
    parts.append(
        "<th>Iter</th><th>Quality</th><th>Agreement</th>"
        "<th>Issues</th><th>Fixed</th><th>New</th><th>Status</th>"
    )
    parts.append("</tr></thead>\n<tbody>\n")
    for it in iterations:
        agr = f"{it.agreement:.0%}" if it.agreement is not None else "N/A"
        fixed = str(len(it.issues_fixed)) if it.issues_fixed else "-"
        intro = str(len(it.issues_introduced)) if it.issues_introduced else "-"
        if it.accepted:
            status = "<span class='badge-accepted'>Accepted</span>"
        else:
            status = (
                f"<span class='badge-rejected'>Rejected: "
                f"{_escape_html(it.rejection_reason or '')}</span>"
            )
        parts.append(
            f"<tr><td>{it.iteration}</td><td>{it.quality_score:.1%}</td>"
            f"<td>{agr}</td><td>{len(it.issues)}</td>"
            f"<td>{fixed}</td><td>{intro}</td><td>{status}</td></tr>\n"
        )
    parts.append("</tbody>\n</table>\n")

    # Per-iteration accordion
    parts.append("<h2>Iteration Details</h2>\n")
    parts.append('<div class="accordion" id="iterAccordion">\n')
    for it in iterations:
        iter_id = f"iter{it.iteration}"
        agr_str = f"{it.agreement:.0%}" if it.agreement is not None else "N/A"
        # Adapt header text for held-out mode (no issues/agreement)
        if it.held_out_diagnostics is not None:
            header_label = f"Iteration {it.iteration} — Accuracy: {it.quality_score:.1%}"
        else:
            header_label = (
                f"Iteration {it.iteration} — Quality: {it.quality_score:.1%}, "
                f"Agreement: {agr_str}, Issues: {len(it.issues)}"
            )
        parts.append('<div class="accordion-item">\n')
        parts.append(
            f'<h2 class="accordion-header">'
            f'<button class="accordion-button collapsed" type="button" '
            f'data-bs-toggle="collapse" data-bs-target="#{iter_id}">'
            f"{header_label}"
            f"</button></h2>\n"
        )
        parts.append(
            f'<div id="{iter_id}" class="accordion-collapse collapse" '
            f'data-bs-parent="#iterAccordion">\n'
            f'<div class="accordion-body">\n'
        )

        # Metrics bar
        cost_str = f"${it.completion_cost:.4f}" if it.completion_cost else "N/A"
        if it.held_out_diagnostics is not None:
            ho = it.held_out_diagnostics
            # Neutral handling-mode label only — no statistical conflation note here
            # (those live solely in MetricsResult.summary()).
            parts.append(
                f'<div class="metrics-bar">'
                f"<span><strong>Accuracy:</strong> {it.quality_score:.1%}</span>"
                f"<span><strong>Coverage:</strong> {_fmt_pct(ho.mean_coverage)}</span>"
                f"<span><strong>CA-rate:</strong> {_fmt_pct(ho.mean_ca_rate)}</span>"
                f"<span><strong>Handling:</strong> CANNOT_ASSESS={ho.cannot_assess}</span>"
                f"<span><strong>Cost:</strong> {cost_str}</span>"
                f"</div>\n"
            )
        else:
            parts.append(
                f'<div class="metrics-bar">'
                f"<span><strong>Quality:</strong> {it.quality_score:.1%}</span>"
                f"<span><strong>Agreement:</strong> {agr_str}</span>"
                f"<span><strong>Cost:</strong> {cost_str}</span>"
                f"</div>\n"
            )

        # Held-out per-criterion table. "Raw % Agreement" is the judge-vs-ground-
        # truth accuracy; alongside it we surface kappa, coverage, CA-rate, precision,
        # the FP/FN rates, and the per-criterion 2x2 (TP/FP/TN/FN). No conflation note
        # here — single-source discipline keeps it in MetricsResult.summary().
        if it.held_out_diagnostics is not None:
            parts.append("<h4>Per-Criterion Diagnostics</h4>\n<table>\n<thead><tr>")
            parts.append(
                "<th>Criterion</th><th>Raw % Agreement</th><th>Kappa</th>"
                "<th>Coverage</th><th>CA-rate</th><th>Precision</th>"
                "<th>FP Rate</th><th>FN Rate</th>"
                "<th>TP</th><th>FP</th><th>TN</th><th>FN</th>"
            )
            parts.append("</tr></thead>\n<tbody>\n")
            for cr in sorted(
                it.held_out_diagnostics.per_criterion,
                key=lambda c: c.accuracy,
            ):
                acc_color = (
                    "var(--green)"
                    if cr.accuracy >= 0.9
                    else ("var(--yellow)" if cr.accuracy >= 0.7 else "var(--red)")
                )
                cm = cr.confusion_matrix
                precision = cm.precision if cm is not None else None
                tp = cm.tp if cm is not None else None
                fp = cm.fp if cm is not None else None
                tn = cm.tn if cm is not None else None
                fn = cm.fn if cm is not None else None
                parts.append(
                    f"<tr><td class='crit-name'>"
                    f"{_escape_html(cr.criterion_name)}</td>"
                    f"<td style='color: {acc_color};'>{cr.accuracy:.0%}</td>"
                    f"<td>{_fmt_num(cr.kappa)}</td>"
                    f"<td>{_fmt_pct(cr.coverage)}</td>"
                    f"<td>{_fmt_pct(cr.ca_rate)}</td>"
                    f"<td>{_fmt_pct(precision)}</td>"
                    f"<td>{cr.false_positive_rate:.0%}</td>"
                    f"<td>{cr.false_negative_rate:.0%}</td>"
                    f"<td>{_fmt_int(tp)}</td>"
                    f"<td>{_fmt_int(fp)}</td>"
                    f"<td>{_fmt_int(tn)}</td>"
                    f"<td>{_fmt_int(fn)}</td></tr>\n"
                )
            parts.append("</tbody>\n</table>\n")

        # Issues table (meta-rubric mode only)
        elif it.issues:
            from ._improve import _match_issue_to_criteria

            parts.append("<h4>Issues</h4>\n<table>\n<thead><tr>")
            parts.append("<th>Criterion</th><th>Type</th><th>Rubric #</th><th>Feedback</th>")
            parts.append("</tr></thead>\n<tbody>\n")
            for issue in it.issues:
                if issue.is_antipattern:
                    tag = "<span class='issue-anti'>ANTI-PATTERN</span>"
                else:
                    tag = "<span class='issue-gap'>QUALITY GAP</span>"
                refs = _match_issue_to_criteria(issue, it.rubric)
                ref_str = ", ".join(f"#{r}" for r in refs) if refs else "--"
                parts.append(
                    f"<tr><td class='crit-name'>"
                    f"{_escape_html(issue.criterion_name)}</td>"
                    f"<td>{tag}</td>"
                    f"<td>{_escape_html(ref_str)}</td>"
                    f"<td>{_escape_html(issue.feedback)}</td></tr>\n"
                )
            parts.append("</tbody>\n</table>\n")
        else:
            parts.append("<p style='color: var(--green);'>No issues found.</p>\n")

        # Rubric table
        parts.append("<h4>Rubric</h4>\n<table>\n<thead><tr>")
        parts.append("<th>#</th><th>Weight</th><th>Requirement</th>")
        parts.append("</tr></thead>\n<tbody>\n")
        for i, c in enumerate(it.rubric.rubric, 1):
            sign = "+" if c.weight > 0 else ""
            parts.append(
                f"<tr><td>{i}</td>"
                f"<td class='weight-col'>{sign}{c.weight}</td>"
                f"<td>{_escape_html(c.requirement)}</td></tr>\n"
            )
        parts.append("</tbody>\n</table>\n")

        # Link to per-iteration eval HTML (meta-rubric mode only)
        if it.quality_report is not None:
            eval_file = f"eval-iter-{it.iteration:02d}.html"
            parts.append(f"<p><a href='{eval_file}'>View detailed evaluation report</a></p>\n")

        parts.append("</div>\n</div>\n</div>\n")
    parts.append("</div>\n")

    # Final rubric panel
    parts.append('<div class="final-panel">\n')
    parts.append("<h2>Final Rubric</h2>\n<table>\n<thead><tr>")
    parts.append("<th>#</th><th>Weight</th><th>Requirement</th>")
    parts.append("</tr></thead>\n<tbody>\n")
    for i, c in enumerate(final_rubric.rubric, 1):
        sign = "+" if c.weight > 0 else ""
        parts.append(
            f"<tr><td>{i}</td>"
            f"<td class='weight-col'>{sign}{c.weight}</td>"
            f"<td>{_escape_html(c.requirement)}</td></tr>\n"
        )
    parts.append("</tbody>\n</table>\n</div>\n")

    parts.append("""
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js">
</script>
</body>
</html>
""")

    return "".join(parts)
