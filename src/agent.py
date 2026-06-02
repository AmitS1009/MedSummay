from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .llm import GroqLLM, LLMError
from .models import (
    AgentAction,
    ClinicalFact,
    DataStatus,
    Diagnosis,
    DischargeSummary,
    LabResult,
    Medication,
    MedicationChange,
    NoteRecord,
    Procedure,
    ReviewFlag,
    SourceRef,
    StepTrace,
    ToolResult,
)
from .prompts import PLANNER_SYSTEM_PROMPT
from .tools import ToolRegistry
from .utils import add_trace_step, compact_json, render_trace, summarize_tool_result, write_text


class DischargeSummaryAgent:
    def __init__(
        self,
        notes: list[NoteRecord],
        llm: GroqLLM,
        *,
        max_steps: int = 25,
        on_step: Callable[[int, AgentAction, ToolResult], None] | None = None,
    ) -> None:
        if not notes:
            raise ValueError("At least one note is required")
        self.notes = notes
        self.llm = llm
        self.max_steps = max_steps
        self.on_step = on_step
        patient_id = Path(notes[0].path).parent.name
        self.summary = DischargeSummary(patient_id=patient_id)
        self.trace = StepTrace(patient_id=patient_id)
        self.tools = ToolRegistry(notes, llm)
        self._action_counts: Counter[str] = Counter()
        self._observations: list[str] = []
        self._fallback_queue = self._build_fallback_queue()

    def run(self) -> tuple[DischargeSummary, StepTrace]:
        stop_reason = ""
        for step_number in range(1, self.max_steps + 1):
            action = self._choose_action()
            action.inputs = self._prepare_inputs(action.tool, action.inputs)
            action_key = f"{action.tool}:{json.dumps(action.inputs, sort_keys=True, default=str)}"
            self._action_counts[action_key] += 1
            if self._action_counts[action_key] >= 3 and action.tool != "flag_for_clinician":
                action = AgentAction(
                    reasoning="The same action was selected repeatedly, so the loop guard is escalating it.",
                    tool="flag_for_clinician",
                    inputs={"issue": f"Repeated agent action blocked: {action_key}", "category": "loop-control"},
                    next_decision="Continue with a different source or finalize with visible missing-data flags.",
                )
            result = self._call_with_retry(action.tool, action.inputs)
            self._apply_result(action.tool, action.inputs, result)
            self._observations.append(f"{action.tool}: {summarize_tool_result(result, limit=500)}")
            add_trace_step(
                self.trace,
                number=step_number,
                reasoning=action.reasoning,
                tool=action.tool,
                inputs=action.inputs,
                result=result,
                next_decision=action.next_decision,
            )
            if self.on_step:
                self.on_step(step_number, action, result)
            if action.tool == "finalize" and result.success:
                stop_reason = "Planner selected finalize."
                break
        else:
            self.summary.step_cap_reached = True
            self._add_flag("loop-control", f"Agent reached the hard cap of {self.max_steps} iterations.")
            stop_reason = f"Hard step cap reached ({self.max_steps})."
        self._run_final_safety_checks()
        self.trace.stop_reason = stop_reason
        return self.summary, self.trace

    def _prepare_inputs(self, tool: str, inputs: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(inputs)
        if tool == "reconcile_medications":
            prepared["admission_medications"] = [
                med.model_dump(mode="json") for med in self.summary.admission_medications
            ]
            prepared["discharge_medications"] = [
                med.model_dump(mode="json") for med in self.summary.discharge_medications
            ]
        elif tool == "check_drug_interactions":
            prepared["medications"] = [
                med.model_dump(mode="json") for med in self.summary.discharge_medications
            ]
        return prepared

    def _choose_action(self) -> AgentAction:
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"AVAILABLE TOOLS:\n{self.tools.available_tools}\n\n"
                    f"NOTE CATALOG:\n{compact_json(self._note_catalog(), limit=4500)}\n\n"
                    f"CURRENT DRAFT STATE:\n{compact_json(self._state_snapshot(), limit=5000)}\n\n"
                    f"RECENT OBSERVATIONS:\n{compact_json(self._observations[-8:], limit=5000)}"
                ),
            },
        ]
        try:
            raw = self.llm.call_json(messages, max_tokens=650)
            action = AgentAction.model_validate(raw)
            if action.tool not in self.tools.available_tools:
                raise ValueError(f"Planner selected unknown tool: {action.tool}")
            return action
        except (LLMError, ValueError) as exc:
            self._add_flag("tool-failure", f"Planner call failed; using deterministic fallback: {exc}")
            if self._fallback_queue:
                return self._fallback_queue.pop(0)
            return AgentAction(
                reasoning="Planner was unavailable and the fallback actions are exhausted.",
                tool="finalize",
                inputs={},
                next_decision="Finalize with visible missing-data and failure flags.",
            )

    def _call_with_retry(self, tool: str, inputs: dict[str, Any]) -> ToolResult:
        result = self.tools.call(tool, inputs)
        if result.success or not result.retryable:
            return result
        retry = self.tools.call(tool, inputs)
        if not retry.success:
            self._add_flag("tool-failure", f"`{tool}` failed after one agent-level retry: {retry.error}")
        return retry

    def _apply_result(self, tool: str, inputs: dict[str, Any], result: ToolResult) -> None:
        if not result.success:
            self._add_flag("tool-failure", f"`{tool}` failed: {result.error}")
            return
        if tool.startswith("extract_"):
            self._apply_extraction(result)
        elif tool == "reconcile_medications":
            self.summary.medication_changes = [
                MedicationChange.model_validate(change) for change in result.data.get("changes", [])
            ]
            for change in self.summary.medication_changes:
                if change.needs_review:
                    self._add_flag(
                        "medication-reconciliation",
                        f"{change.medication}: {change.change_type} without a documented reason.",
                    )
        elif tool == "check_drug_interactions":
            for interaction in result.data.get("interactions", []):
                self._add_flag(
                    "drug-interaction",
                    f"{interaction['medications']}: {interaction['message']}",
                    severity=interaction["severity"],
                )
        elif tool == "flag_for_clinician":
            flagged = result.data["flagged"]
            self._add_flag(flagged["category"], flagged["message"], severity=flagged["severity"])
        if result.review_required:
            self._add_flag("evidence-review", f"`{tool}` returned content requiring clinician review.")

    def _apply_extraction(self, result: ToolResult) -> None:
        source = SourceRef(
            note_id=result.data["note_id"],
            page_number=result.data["page"],
            evidence=None,
        )
        accepted = result.data.get("accepted", {})
        for row in accepted.get("fields", []):
            self._apply_field(row, source)
        for row in accepted.get("diagnoses", []):
            self._apply_diagnosis(row, source)
        for row in accepted.get("medications", []):
            self._apply_medication(row, source)
        for row in accepted.get("labs", []):
            self._apply_lab(row, source)
        for row in accepted.get("hospital_course", []):
            self._append_fact(self.summary.hospital_course, self._fact_from_row(row, source))
        for row in accepted.get("procedures", []):
            item_source = self._source_with_evidence(source, row)
            procedure = Procedure(
                name=row["name"],
                date=row.get("date"),
                status=self._status(row.get("status")),
                sources=[item_source],
            )
            if not any(existing.name.lower() == procedure.name.lower() for existing in self.summary.procedures):
                self.summary.procedures.append(procedure)
        for row in accepted.get("follow_up", []):
            self._append_fact(self.summary.follow_up_instructions, self._fact_from_row(row, source))
        for row in accepted.get("pending_results", []):
            row["status"] = "PENDING"
            self._append_pending_lab(row, source)
        rejected = result.data.get("rejected_unsupported", [])
        if rejected:
            self._add_flag(
                "unsupported-extraction",
                f"Rejected {len(rejected)} unsupported extracted item(s) from {source.note_id}.",
            )

    def _apply_field(self, row: dict[str, Any], source: SourceRef) -> None:
        fact = self._fact_from_row(row, source)
        field = row.get("field")
        demographic_fields = {"name", "mrn", "dob", "age", "sex"}
        if field in demographic_fields:
            current = getattr(self.summary.demographics, field)
            setattr(self.summary.demographics, field, self._merge_fact(current, fact, f"demographics.{field}"))
        elif field in {"admission_date", "discharge_date", "allergies", "discharge_condition"}:
            current = getattr(self.summary, field)
            setattr(self.summary, field, self._merge_fact(current, fact, field))

    def _apply_diagnosis(self, row: dict[str, Any], source: SourceRef) -> None:
        diagnosis = Diagnosis(
            name=row["name"],
            category=row.get("category", "unclear"),
            status=self._status(row.get("status")),
            sources=[self._source_with_evidence(source, row)],
        )
        target = (
            self.summary.principal_diagnoses
            if diagnosis.category == "principal"
            else self.summary.secondary_diagnoses
        )
        existing = next((item for item in target if item.name.lower() == diagnosis.name.lower()), None)
        if existing:
            existing.sources.extend(diagnosis.sources)
        else:
            target.append(diagnosis)
        if diagnosis.category == "principal":
            distinct = {item.name.lower() for item in self.summary.principal_diagnoses}
            if len(distinct) > 1:
                self._add_flag(
                    "diagnosis-conflict",
                    f"Multiple principal diagnoses were extracted: {', '.join(sorted(distinct))}.",
                )

    def _apply_medication(self, row: dict[str, Any], source: SourceRef) -> None:
        medication = Medication(
            name=row["name"],
            dose=row.get("dose"),
            frequency=row.get("frequency"),
            route=row.get("route"),
            duration=row.get("duration"),
            list_type=row.get("list_type", "unclear"),
            documented_reason=row.get("documented_reason"),
            status=self._status(row.get("status")),
            sources=[self._source_with_evidence(source, row)],
        )
        if medication.list_type == "admission":
            self._upsert_medication(self.summary.admission_medications, medication)
        elif medication.list_type == "discharge":
            self._upsert_medication(self.summary.discharge_medications, medication)
        else:
            self._add_flag(
                "medication-reconciliation",
                f"{medication.name} was extracted from an {medication.list_type} list and was not silently assigned to admission or discharge.",
            )

    def _apply_lab(self, row: dict[str, Any], source: SourceRef) -> None:
        if self._status(row.get("status")) == DataStatus.PENDING:
            self._append_pending_lab(row, source)

    def _append_pending_lab(self, row: dict[str, Any], source: SourceRef) -> None:
        lab = LabResult(
            name=row["name"],
            value=row.get("value"),
            unit=row.get("unit"),
            reference_range=row.get("reference_range"),
            status=DataStatus.PENDING,
            sources=[self._source_with_evidence(source, row)],
        )
        if not any(existing.name.lower() == lab.name.lower() for existing in self.summary.pending_results):
            self.summary.pending_results.append(lab)

    def _merge_fact(self, current: ClinicalFact, new: ClinicalFact, label: str) -> ClinicalFact:
        if current.status in {DataStatus.MISSING, DataStatus.UNAVAILABLE}:
            return new
        if new.status in {DataStatus.MISSING, DataStatus.UNAVAILABLE}:
            return current
        if (current.value or "").strip().lower() == (new.value or "").strip().lower():
            current.sources.extend(new.sources)
            return current
        values = [current.value or "", new.value or ""]
        current.status = DataStatus.CONFLICTING
        current.review_note = f"{label} differs across notes: {' vs '.join(values)}"
        current.sources.extend(new.sources)
        self._add_flag("conflict", current.review_note, sources=current.sources)
        return current

    @staticmethod
    def _append_fact(target: list[ClinicalFact], fact: ClinicalFact) -> None:
        if not any((item.value or "").lower() == (fact.value or "").lower() for item in target):
            target.append(fact)

    @staticmethod
    def _upsert_medication(target: list[Medication], medication: Medication) -> None:
        existing = next((item for item in target if item.name.lower() == medication.name.lower()), None)
        if existing is None:
            target.append(medication)
        elif existing.signature() == medication.signature():
            existing.sources.extend(medication.sources)
        else:
            target.append(medication)

    @staticmethod
    def _source_with_evidence(source: SourceRef, row: dict[str, Any]) -> SourceRef:
        return source.model_copy(update={"evidence": row.get("evidence")})

    def _fact_from_row(self, row: dict[str, Any], source: SourceRef) -> ClinicalFact:
        return ClinicalFact(
            value=row.get("value"),
            status=self._status(row.get("status")),
            sources=[self._source_with_evidence(source, row)],
            confidence=1.0,
        )

    @staticmethod
    def _status(value: str | None) -> DataStatus:
        try:
            return DataStatus(value or "CONFIRMED")
        except ValueError:
            return DataStatus.CONFIRMED

    def _run_final_safety_checks(self) -> None:
        reconciliation = self.tools.reconcile_medications(
            [med.model_dump(mode="json") for med in self.summary.admission_medications],
            [med.model_dump(mode="json") for med in self.summary.discharge_medications],
        )
        self._apply_result("reconcile_medications", {}, reconciliation)
        interactions = self.tools.check_drug_interactions(
            [med.model_dump(mode="json") for med in self.summary.discharge_medications]
        )
        self._apply_result("check_drug_interactions", {}, interactions)
        required_fact_fields = {
            "admission date": self.summary.admission_date,
            "discharge date": self.summary.discharge_date,
            "allergies": self.summary.allergies,
            "discharge condition": self.summary.discharge_condition,
        }
        for label, fact in required_fact_fields.items():
            if fact.status in {DataStatus.MISSING, DataStatus.UNAVAILABLE}:
                self._add_flag("missing-data", f"Required field not sourced: {label}.")
        if not self.summary.principal_diagnoses:
            self._add_flag("missing-data", "Principal diagnosis not sourced.")
        if not self.summary.hospital_course:
            self._add_flag("missing-data", "Hospital course not sourced.")
        if not self.summary.procedures:
            self._add_flag("missing-data", "Procedures not sourced; confirm whether none occurred.")
        if not self.summary.discharge_medications:
            self._add_flag("missing-data", "Discharge medications not sourced.")
        if not self.summary.follow_up_instructions:
            self._add_flag("missing-data", "Follow-up instructions not sourced.")
        for pending in self.summary.pending_results:
            self._add_flag("pending-result", f"{pending.name} remains pending and requires follow-up.")

    def _add_flag(
        self,
        category: str,
        message: str,
        *,
        severity: str = "warning",
        sources: list[SourceRef] | None = None,
    ) -> None:
        flag = ReviewFlag(category=category, message=message, severity=severity, sources=sources or [])
        if not any(existing.category == flag.category and existing.message == flag.message for existing in self.summary.review_flags):
            self.summary.review_flags.append(flag)

    def _note_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "note_id": note.note_id,
                "page": note.page_number,
                "type": note.note_type.value,
                "characters": len(note.text),
                "readable": bool(note.text),
            }
            for note in self.notes
        ]

    def _state_snapshot(self) -> dict[str, Any]:
        return {
            "patient_id": self.summary.patient_id,
            "demographics": self.summary.demographics.model_dump(mode="json"),
            "admission_date": self.summary.admission_date.model_dump(mode="json"),
            "discharge_date": self.summary.discharge_date.model_dump(mode="json"),
            "principal_diagnoses": [item.model_dump(mode="json") for item in self.summary.principal_diagnoses],
            "secondary_diagnoses": [item.model_dump(mode="json") for item in self.summary.secondary_diagnoses],
            "hospital_course_count": len(self.summary.hospital_course),
            "procedures_count": len(self.summary.procedures),
            "admission_medications": [item.model_dump(mode="json") for item in self.summary.admission_medications],
            "discharge_medications": [item.model_dump(mode="json") for item in self.summary.discharge_medications],
            "pending_results": [item.model_dump(mode="json") for item in self.summary.pending_results],
            "review_flags": [item.model_dump(mode="json") for item in self.summary.review_flags],
        }

    def _build_fallback_queue(self) -> list[AgentAction]:
        actions = [
            AgentAction(
                reasoning="Inspect the available note catalog before selecting source documents.",
                tool="list_notes",
                inputs={},
                next_decision="Read the highest-yield readable source notes.",
            )
        ]
        priority = {"discharge": 0, "admission": 1, "lab": 2, "medication": 3, "progress": 4}
        selected = sorted(
            (note for note in self.notes if note.text and note.note_type.value in priority),
            key=lambda note: (priority[note.note_type.value], note.page_number),
        )[:4]
        for note in selected:
            actions.append(
                AgentAction(
                    reasoning=f"Read a high-yield {note.note_type.value} source note.",
                    tool="read_note",
                    inputs={"note_id": note.note_id},
                    next_decision="Attempt source-grounded extraction from this note.",
                )
            )
            for extraction_tool in self._extractors_for_note(note):
                actions.append(
                    AgentAction(
                        reasoning=f"Extract source-grounded fields from {note.note_id}.",
                        tool=extraction_tool,
                        inputs={"note_id": note.note_id},
                        next_decision="Use accepted evidence and visibly flag unsupported output.",
                    )
                )
        return actions

    @staticmethod
    def _extractors_for_note(note: NoteRecord) -> list[str]:
        if note.note_type.value == "discharge":
            return ["extract_diagnoses", "extract_medications", "extract_clinical_details"]
        if note.note_type.value == "admission":
            return ["extract_demographics", "extract_diagnoses", "extract_medications", "extract_clinical_details"]
        if note.note_type.value == "lab":
            return ["extract_labs"]
        if note.note_type.value == "medication":
            return ["extract_medications"]
        return ["extract_clinical_details", "extract_diagnoses"]


def _display_diagnoses(diagnoses: list[Diagnosis]) -> list[str]:
    return [diagnosis.name for diagnosis in diagnoses] or ["[MISSING - NOT FOUND IN SOURCE DOCUMENTS]"]


def _display_fact_list(facts: list[ClinicalFact]) -> list[str]:
    return [fact.display() for fact in facts] or ["[MISSING - NOT FOUND IN SOURCE DOCUMENTS]"]


def render_summary(summary: DischargeSummary) -> str:
    demographics = summary.demographics
    lines = [
        "# DISCHARGE SUMMARY - DRAFT FOR CLINICIAN REVIEW",
        "",
        "> This draft was generated from source documents only. It must be reviewed and finalized by a clinician.",
        "",
        "## Patient Demographics",
        f"- Name: {demographics.name.display()}",
        f"- MRN: {demographics.mrn.display()}",
        f"- DOB: {demographics.dob.display()}",
        f"- Age: {demographics.age.display()}",
        f"- Sex: {demographics.sex.display()}",
        "",
        "## Admission & Discharge Dates",
        f"- Admitted: {summary.admission_date.display()}",
        f"- Discharged: {summary.discharge_date.display()}",
        "",
        "## Principal Diagnosis",
    ]
    lines.extend(f"- {value}" for value in _display_diagnoses(summary.principal_diagnoses))
    lines.extend(["", "## Secondary Diagnoses"])
    lines.extend(f"- {value}" for value in _display_diagnoses(summary.secondary_diagnoses))
    lines.extend(["", "## Hospital Course"])
    lines.extend(f"- {value}" for value in _display_fact_list(summary.hospital_course))
    lines.extend(["", "## Procedures"])
    if summary.procedures:
        lines.extend(f"- {item.name}{f' ({item.date})' if item.date else ''}" for item in summary.procedures)
    else:
        lines.append("- [MISSING - NOT FOUND IN SOURCE DOCUMENTS] Confirm whether no procedures occurred.")
    lines.extend(
        [
            "",
            "## Discharge Medications",
            "| Medication | Dose | Frequency | Route | Duration |",
            "|---|---|---|---|---|",
        ]
    )
    if summary.discharge_medications:
        for med in summary.discharge_medications:
            lines.append(
                f"| {med.name} | {med.dose or '[MISSING]'} | {med.frequency or '[MISSING]'} | "
                f"{med.route or '[MISSING]'} | {med.duration or '[MISSING]'} |"
            )
    else:
        lines.append("| [MISSING - NOT FOUND IN SOURCE DOCUMENTS] |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Medication Changes From Admission",
            "| Medication | Change | Admission | Discharge | Documented Reason | Review |",
            "|---|---|---|---|---|---|",
        ]
    )
    if summary.medication_changes:
        for change in summary.medication_changes:
            review = "FLAGGED" if change.needs_review else ""
            lines.append(
                f"| {change.medication} | {change.change_type} | {change.admission_details or '-'} | "
                f"{change.discharge_details or '-'} | {change.documented_reason or '[NOT DOCUMENTED]'} | {review} |"
            )
    else:
        lines.append("| [MISSING - RECONCILIATION COULD NOT IDENTIFY MEDICATIONS] |  |  |  |  | FLAGGED |")
    lines.extend(["", "## Allergies", f"- {summary.allergies.display()}", "", "## Follow-Up Instructions"])
    lines.extend(f"- {value}" for value in _display_fact_list(summary.follow_up_instructions))
    lines.extend(["", "## Pending Results"])
    if summary.pending_results:
        lines.extend(f"- [PENDING] {lab.name}: {lab.value or 'No result documented'}" for lab in summary.pending_results)
    else:
        lines.append("- No pending result was sourced. Confirm during clinician review.")
    lines.extend(["", "## Discharge Condition", f"- {summary.discharge_condition.display()}", "", "## Flags for Clinician Review"])
    if summary.review_flags:
        lines.extend(
            f"{index}. [{flag.severity.upper()}] {flag.message}"
            for index, flag in enumerate(summary.review_flags, start=1)
        )
    else:
        lines.append("1. No automated flag was raised. Clinician review is still required.")
    lines.append("")
    return "\n".join(lines)


def save_agent_outputs(
    summary: DischargeSummary,
    trace: StepTrace,
    output_root: str | Path = "outputs",
) -> tuple[Path, Path]:
    destination = Path(output_root) / summary.patient_id
    summary_path = destination / "discharge_summary.md"
    trace_path = destination / "step_trace.md"
    write_text(summary_path, render_summary(summary))
    write_text(trace_path, render_trace(trace))
    write_text(destination / "discharge_summary.json", summary.model_dump_json(indent=2))
    return summary_path, trace_path
