from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import AgentStep, StepTrace, ToolResult


def compact_json(value: Any, limit: int = 700) -> str:
    text = json.dumps(value, ensure_ascii=True, default=str, sort_keys=True)
    return text if len(text) <= limit else f"{text[:limit]}... [truncated]"


def summarize_tool_result(result: ToolResult, limit: int = 700) -> str:
    if not result.success:
        return f"FAILED: {result.error}"
    return compact_json(result.data, limit=limit)


def render_trace(trace: StepTrace) -> str:
    lines = [
        f"# Step Trace - {trace.patient_id}",
        "",
        "> This is an auditable action trace with brief decision rationales, not hidden chain-of-thought.",
        "",
    ]
    for step in trace.steps:
        lines.extend(
            [
                f"## Step {step.number}",
                f"- Reasoning: {step.reasoning}",
                f"- Tool/action: `{step.tool}`",
                f"- Inputs: `{compact_json(step.inputs, limit=400)}`",
                f"- Result: {step.result}",
                f"- Next decision: {step.next_decision}",
                "",
            ]
        )
    lines.extend(["## Stop Reason", trace.stop_reason or "Not recorded", ""])
    return "\n".join(lines)


def add_trace_step(
    trace: StepTrace,
    *,
    number: int,
    reasoning: str,
    tool: str,
    inputs: dict[str, Any],
    result: ToolResult,
    next_decision: str,
) -> None:
    trace.steps.append(
        AgentStep(
            number=number,
            reasoning=reasoning,
            tool=tool,
            inputs=inputs,
            result=summarize_tool_result(result),
            next_decision=next_decision,
        )
    )


def write_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")

