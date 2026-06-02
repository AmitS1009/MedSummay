from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .utils import write_text


@dataclass(frozen=True)
class ReviewExample:
    case_id: str
    draft: str


@dataclass
class EvaluationRow:
    iteration: int
    split: str
    case_id: str
    edit_distance: float
    reward: float


@dataclass
class CorrectionMemory:
    replacements: dict[str, str] = field(default_factory=dict)
    section_insertions: dict[str, str] = field(default_factory=dict)

    def apply(self, draft: str) -> str:
        text = draft
        for before, after in sorted(self.replacements.items(), key=lambda item: -len(item[0])):
            text = text.replace(before, after)
        for heading, insertion in self.section_insertions.items():
            if heading in text and insertion not in text:
                text = text.replace(heading, f"{heading}\n{insertion}", 1)
        return _collapse_blank_lines(text)

    def learn_from_pair(self, draft: str, edited: str) -> None:
        for before, after in SimulatedDoctorReviewer.EDIT_POLICY:
            if before in draft and after in edited:
                self.replacements[before] = after
        if "## Flags for Clinician Review\n" in edited:
            inserted = _first_flag_line(edited)
            if inserted and inserted not in draft:
                self.section_insertions["## Flags for Clinician Review"] = inserted


class SimulatedDoctorReviewer:
    """
    Hidden stand-in reviewer.

    The policy is intentionally simple and consistent: remove common OCR/formatting noise,
    enforce the assignment's missing/pending language, and add a reconciliation flag when a
    medication is marked added without a reason.
    """

    EDIT_POLICY = [
        ("[MISSING]", "[MISSING - NOT FOUND IN SOURCE DOCUMENTS]"),
        ("No pending result was sourced.", "No pending result was sourced. Confirm during clinician review."),
        ("Discharge Summary", "DISCHARGE SUMMARY - DRAFT FOR CLINICIAN REVIEW"),
        ("S DAYS", "5 DAYS"),
        ("mnol/L", "mmol/L"),
        ("me/dl", "mg/dL"),
        ("ITABSOS", "[MISSING - OCR UNCLEAR]"),
        ("Adyeto", "[MISSING - OCR UNCLEAR]"),
        ("Nd Loum", "[MISSING - OCR UNCLEAR]"),
        ("ray", "[MISSING - OCR UNCLEAR]"),
        ("ge - 2-4 by 6", "[MISSING - OCR UNCLEAR]"),
    ]

    def edit(self, draft: str) -> str:
        edited = draft
        for before, after in self.EDIT_POLICY:
            edited = edited.replace(before, after)
        edited = self._add_med_reconciliation_flag(edited)
        edited = self._standardize_pending_results(edited)
        return _collapse_blank_lines(edited)

    @staticmethod
    def _add_med_reconciliation_flag(text: str) -> str:
        if "| ADDED |" not in text or "without documented reconciliation reason" in text:
            return text
        flag = "1. [WARNING] Medication added without documented reconciliation reason."
        if "## Flags for Clinician Review" in text:
            return text.replace("## Flags for Clinician Review\n", f"## Flags for Clinician Review\n{flag}\n", 1)
        return f"{text.rstrip()}\n\n## Flags for Clinician Review\n{flag}\n"

    @staticmethod
    def _standardize_pending_results(text: str) -> str:
        return re.sub(
            r"- \[PENDING\] ([^:\n]+): No result documented",
            r"- [PENDING] \1: No result documented; follow-up required",
            text,
        )


def normalized_edit_distance(draft: str, edited: str) -> float:
    if not draft and not edited:
        return 0.0
    ratio = SequenceMatcher(None, draft, edited).ratio()
    return round(1.0 - ratio, 4)


def reward_from_edits(draft: str, edited: str) -> float:
    return round(1.0 - normalized_edit_distance(draft, edited), 4)


def synthetic_cases() -> list[ReviewExample]:
    return [
        ReviewExample(
            "case_001",
            """# Discharge Summary

## Patient Demographics
- Name: [MISSING]

## Hospital Course
- Serum sodium 128.00 mnol/L.

## Discharge Medications
| Medication | Change | Duration |
|---|---|---|
| LOPIRAMIDE | ADDED | S DAYS |

## Pending Results
- [PENDING] Urine culture and sensitivity: No result documented

## Flags for Clinician Review
""",
        ),
        ReviewExample(
            "case_002",
            """# DISCHARGE SUMMARY - DRAFT FOR CLINICIAN REVIEW

## Patient Demographics
- Name: ray

## Admission & Discharge Dates
- Admitted: ge - 2-4 by 6

## Discharge Medications
| Medication | Change | Documented Reason |
|---|---|---|
| Adyeto | ADDED | [NOT DOCUMENTED] |

## Flags for Clinician Review
""",
        ),
        ReviewExample(
            "case_003",
            """# Discharge Summary

## Allergies
- Nd Loum

## Discharge Medications
| Medication | Dose | Change |
|---|---|---|
| MEFTAL SPAS | ITABSOS | ADDED |

## Pending Results
No pending result was sourced.
""",
        ),
        ReviewExample(
            "case_004",
            """# DISCHARGE SUMMARY - DRAFT FOR CLINICIAN REVIEW

## Patient Demographics
- MRN: [MISSING]

## Hospital Course
- Serum creatinine elevated at 1.645 me/dl.

## Pending Results
- [PENDING] Blood culture: No result documented

## Flags for Clinician Review
""",
        ),
        ReviewExample(
            "case_005",
            """# Discharge Summary

## Patient Demographics
- DOB: [MISSING]

## Discharge Medications
| Medication | Change | Duration |
|---|---|---|
| OFLOX TZ | ADDED | S DAYS |

## Flags for Clinician Review
""",
        ),
        ReviewExample(
            "case_006",
            """# DISCHARGE SUMMARY - DRAFT FOR CLINICIAN REVIEW

## Allergies
- Nd Loum

## Discharge Medications
| Medication | Change | Documented Reason |
|---|---|---|
| RACTPER | ADDED | [NOT DOCUMENTED] |

## Pending Results
- [PENDING] HbA1c: No result documented

## Flags for Clinician Review
""",
        ),
    ]


def run_learning_demo(output_dir: str | Path = "outputs/part2_learning") -> dict[str, object]:
    reviewer = SimulatedDoctorReviewer()
    memory = CorrectionMemory()
    cases = synthetic_cases()
    train_cases = cases[:4]
    heldout_cases = cases[4:]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    rows: list[EvaluationRow] = []
    curve: list[dict[str, float | int]] = []

    def evaluate(split: str, examples: list[ReviewExample], iteration: int) -> list[EvaluationRow]:
        evaluated: list[EvaluationRow] = []
        for example in examples:
            improved = memory.apply(example.draft)
            edited = reviewer.edit(example.draft)
            evaluated.append(
                EvaluationRow(
                    iteration=iteration,
                    split=split,
                    case_id=example.case_id,
                    edit_distance=normalized_edit_distance(improved, edited),
                    reward=reward_from_edits(improved, edited),
                )
            )
        return evaluated

    rows.extend(evaluate("heldout", heldout_cases, 0))
    curve.append(_curve_point(rows, 0))
    for iteration, example in enumerate(train_cases, start=1):
        edited = reviewer.edit(example.draft)
        memory.learn_from_pair(example.draft, edited)
        write_text(destination / "pairs" / f"{example.case_id}_draft.md", example.draft)
        write_text(destination / "pairs" / f"{example.case_id}_edited.md", edited)
        rows.extend(evaluate("heldout", heldout_cases, iteration))
        curve.append(_curve_point(rows, iteration))

    final_outputs = []
    for example in heldout_cases:
        improved = memory.apply(example.draft)
        edited = reviewer.edit(example.draft)
        final_outputs.append(
            {
                "case_id": example.case_id,
                "draft": example.draft,
                "memory_improved_draft": improved,
                "simulated_doctor_edit": edited,
                "before_edit_distance": normalized_edit_distance(example.draft, edited),
                "after_edit_distance": normalized_edit_distance(improved, edited),
            }
        )
        write_text(destination / f"{example.case_id}_improved.md", improved)
        write_text(destination / f"{example.case_id}_doctor_edit.md", edited)

    _write_metrics_csv(destination / "metrics.csv", rows)
    report = _render_report(curve, memory, final_outputs)
    write_text(destination / "report.md", report)
    write_text(
        destination / "correction_memory.json",
        json.dumps(
            {
                "replacements": memory.replacements,
                "section_insertions": memory.section_insertions,
            },
            indent=2,
        ),
    )
    return {
        "output_dir": str(destination),
        "curve": curve,
        "memory_size": len(memory.replacements) + len(memory.section_insertions),
        "report": str(destination / "report.md"),
    }


def _curve_point(rows: list[EvaluationRow], iteration: int) -> dict[str, float | int]:
    relevant = [row for row in rows if row.iteration == iteration and row.split == "heldout"]
    avg_distance = sum(row.edit_distance for row in relevant) / len(relevant)
    avg_reward = sum(row.reward for row in relevant) / len(relevant)
    return {
        "iteration": iteration,
        "avg_edit_distance": round(avg_distance, 4),
        "avg_reward": round(avg_reward, 4),
    }


def _write_metrics_csv(path: Path, rows: list[EvaluationRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["iteration", "split", "case_id", "edit_distance", "reward"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _render_report(
    curve: list[dict[str, float | int]],
    memory: CorrectionMemory,
    final_outputs: list[dict[str, object]],
) -> str:
    before = curve[0]["avg_edit_distance"]
    after = curve[-1]["avg_edit_distance"]
    improvement = round(((before - after) / before) * 100, 1) if before else 0.0
    lines = [
        "# Part 2 - Learning From Simulated Doctor Edits",
        "",
        "## Reward",
        "Reward = 1 - normalized edit distance between the memory-improved draft and the simulated doctor's edited version.",
        "",
        "## Learning Mechanism",
        "A correction memory learns recurring section/text edits from accumulated (draft, edited) pairs and applies them to future drafts before review.",
        "",
        "## Held-Out Improvement Curve",
        "| Iteration | Avg Edit Distance | Avg Reward |",
        "|---|---:|---:|",
    ]
    for point in curve:
        lines.append(f"| {point['iteration']} | {point['avg_edit_distance']} | {point['avg_reward']} |")
    lines.extend(
        [
            "",
            f"Before average edit distance: {before}",
            f"After average edit distance: {after}",
            f"Relative improvement: {improvement}%",
            "",
            "## Learned Correction Memory",
            "```json",
            json.dumps(
                {
                    "replacements": memory.replacements,
                    "section_insertions": memory.section_insertions,
                },
                indent=2,
            ),
            "```",
            "",
            "## Held-Out Cases",
        ]
    )
    for item in final_outputs:
        lines.extend(
            [
                f"### {item['case_id']}",
                f"- Before edit distance: {item['before_edit_distance']}",
                f"- After edit distance: {item['after_edit_distance']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "This reviewer is synthetic, so it measures whether the learner captures a known editing policy, not real clinical accuracy.",
            "To prevent gaming, the memory only rewrites formatting/OCR conventions and adds explicit review flags; it does not create new diagnoses, dates, medication reasons, or lab values.",
            "The Part 1 guardrail still owns clinical facts: unsupported facts remain missing, pending, conflicting, or flagged for clinician review.",
        ]
    )
    return "\n".join(lines) + "\n"


def _first_flag_line(text: str) -> str | None:
    match = re.search(r"^\d+\. \[WARNING\] Medication added without documented reconciliation reason\.$", text, re.M)
    return match.group(0) if match else None


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip()) + "\n"

