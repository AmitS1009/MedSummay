from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable

from .llm import GroqLLM, LLMError
from .models import Medication, MedicationChange, NoteRecord, SourceRef, ToolResult
from .prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_TASKS


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def evidence_is_supported(evidence: str | None, source_text: str) -> bool:
    if not evidence or not evidence.strip():
        return False
    return _normalize_text(evidence) in _normalize_text(source_text)


def medication_details(medication: Medication | None) -> str | None:
    if medication is None:
        return None
    details = [medication.dose, medication.frequency, medication.route]
    return " | ".join(value for value in details if value) or medication.name


class ToolRegistry:
    def __init__(self, notes: list[NoteRecord], llm: GroqLLM) -> None:
        self.notes = {note.note_id: note for note in notes}
        self.llm = llm
        self.flags: list[dict[str, Any]] = []
        self._handlers: dict[str, Callable[..., ToolResult]] = {
            "list_notes": self.list_notes,
            "read_note": self.read_note,
            "search_notes": self.search_notes,
            "extract_demographics": self.extract_demographics,
            "extract_diagnoses": self.extract_diagnoses,
            "extract_medications": self.extract_medications,
            "extract_labs": self.extract_labs,
            "extract_clinical_details": self.extract_clinical_details,
            "reconcile_medications": self.reconcile_medications,
            "check_drug_interactions": self.check_drug_interactions,
            "flag_for_clinician": self.flag_for_clinician,
            "finalize": self.finalize,
        }

    @property
    def available_tools(self) -> list[str]:
        return list(self._handlers)

    def call(self, name: str, inputs: dict[str, Any] | None = None) -> ToolResult:
        if name not in self._handlers:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        try:
            return self._handlers[name](**(inputs or {}))
        except TypeError as exc:
            return ToolResult(success=False, error=f"Invalid inputs for {name}: {exc}")
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"{name} failed: {type(exc).__name__}: {exc}",
                retryable=True,
            )

    def list_notes(self) -> ToolResult:
        records = [
            {
                "note_id": note.note_id,
                "page": note.page_number,
                "type": note.note_type.value,
                "characters": len(note.text),
                "readable": bool(note.text),
                "extraction_method": note.extraction_method,
                "extraction_error": note.extraction_error,
            }
            for note in self.notes.values()
        ]
        return ToolResult(success=True, data={"notes": records})

    def read_note(self, note_id: str) -> ToolResult:
        note = self.notes.get(note_id)
        if note is None:
            return ToolResult(success=False, error=f"Unknown note_id: {note_id}")
        if not note.text.strip():
            return ToolResult(
                success=False,
                error=f"{note_id} has no readable OCR text",
                review_required=True,
            )
        return ToolResult(
            success=True,
            data={
                "note_id": note.note_id,
                "page": note.page_number,
                "type": note.note_type.value,
                "text": note.text,
                "extraction_method": note.extraction_method,
            },
        )

    def search_notes(self, query: str) -> ToolResult:
        if not query.strip():
            return ToolResult(success=False, error="query must not be empty")
        normalized_query = query.lower()
        matches: list[dict[str, Any]] = []
        for note in self.notes.values():
            lowered = note.text.lower()
            start = lowered.find(normalized_query)
            if start < 0:
                continue
            context_start = max(0, start - 100)
            context_end = min(len(note.text), start + len(query) + 180)
            matches.append(
                {
                    "note_id": note.note_id,
                    "page": note.page_number,
                    "type": note.note_type.value,
                    "snippet": note.text[context_start:context_end].replace("\n", " "),
                }
            )
        return ToolResult(success=True, data={"query": query, "matches": matches[:20]})

    def _extract(self, note_id: str, task: str) -> ToolResult:
        note = self.notes.get(note_id)
        if note is None:
            return ToolResult(success=False, error=f"Unknown note_id: {note_id}")
        if not note.text.strip():
            return ToolResult(
                success=False,
                error=f"{note_id} has no readable OCR text",
                review_required=True,
            )
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{EXTRACTION_TASKS[task]}\n\nSOURCE NOTE {note_id}:\n{note.text}",
            },
        ]
        try:
            raw = self.llm.call_json(messages, max_tokens=1800)
        except LLMError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                retryable=exc.retryable,
                review_required=True,
            )
        accepted, rejected = self._validate_evidence(raw, note.text)
        return ToolResult(
            success=True,
            data={
                "note_id": note_id,
                "page": note.page_number,
                "accepted": accepted,
                "rejected_unsupported": rejected,
            },
            review_required=bool(rejected),
        )

    @staticmethod
    def _validate_evidence(raw: dict[str, Any], source_text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        accepted: dict[str, Any] = {}
        rejected: list[dict[str, Any]] = []
        for key, value in raw.items():
            if not isinstance(value, list):
                rejected.append({"section": key, "reason": "Expected a list", "value": value})
                continue
            accepted_rows: list[dict[str, Any]] = []
            for row in value:
                if not isinstance(row, dict):
                    rejected.append({"section": key, "reason": "Expected an object", "value": row})
                    continue
                if evidence_is_supported(row.get("evidence"), source_text):
                    accepted_rows.append(row)
                else:
                    rejected.append(
                        {
                            "section": key,
                            "reason": "Evidence excerpt was absent from the source note",
                            "value": row,
                        }
                    )
            accepted[key] = accepted_rows
        return accepted, rejected

    def extract_demographics(self, note_id: str) -> ToolResult:
        return self._extract(note_id, "demographics")

    def extract_diagnoses(self, note_id: str) -> ToolResult:
        return self._extract(note_id, "diagnoses")

    def extract_medications(self, note_id: str) -> ToolResult:
        return self._extract(note_id, "medications")

    def extract_labs(self, note_id: str) -> ToolResult:
        return self._extract(note_id, "labs")

    def extract_clinical_details(self, note_id: str) -> ToolResult:
        return self._extract(note_id, "clinical_details")

    def reconcile_medications(
        self,
        admission_medications: list[dict[str, Any]],
        discharge_medications: list[dict[str, Any]],
    ) -> ToolResult:
        admission = [Medication.model_validate(item) for item in admission_medications]
        discharge = [Medication.model_validate(item) for item in discharge_medications]
        admission_by_name = {med.name.strip().lower(): med for med in admission}
        discharge_by_name = {med.name.strip().lower(): med for med in discharge}
        changes: list[MedicationChange] = []
        for name in sorted(set(admission_by_name) | set(discharge_by_name)):
            old = admission_by_name.get(name)
            new = discharge_by_name.get(name)
            if old is None and new is not None:
                change_type = "ADDED"
            elif old is not None and new is None:
                change_type = "STOPPED"
            elif old and new and old.signature() != new.signature():
                change_type = "CHANGED"
            else:
                change_type = "CONTINUED"
            documented_reason = (new and new.documented_reason) or (old and old.documented_reason)
            changes.append(
                MedicationChange(
                    medication=(new or old).name,  # type: ignore[union-attr]
                    change_type=change_type,
                    admission_details=medication_details(old),
                    discharge_details=medication_details(new),
                    documented_reason=documented_reason,
                    needs_review=change_type in {"ADDED", "STOPPED", "CHANGED"} and not documented_reason,
                )
            )
        return ToolResult(success=True, data={"changes": [change.model_dump(mode="json") for change in changes]})

    def check_drug_interactions(self, medications: list[dict[str, Any]]) -> ToolResult:
        meds = [Medication.model_validate(item) for item in medications]
        names = {_normalize_drug_name(medication.name) for medication in meds}
        interactions: list[dict[str, str]] = []
        known_interactions = [
            ({"warfarin", "aspirin"}, "major", "Increased bleeding risk"),
            ({"warfarin", "ibuprofen"}, "major", "Increased bleeding risk"),
            ({"lisinopril", "potassium chloride"}, "major", "Hyperkalemia risk"),
            ({"sildenafil", "nitroglycerin"}, "major", "Potentially severe hypotension"),
            ({"metformin", "contrast media"}, "warning", "Review renal function and contrast exposure"),
        ]
        for pair, severity, message in known_interactions:
            if pair <= names:
                interactions.append(
                    {
                        "medications": " + ".join(sorted(pair)),
                        "severity": severity,
                        "message": message,
                    }
                )
        return ToolResult(success=True, data={"interactions": interactions})

    def flag_for_clinician(
        self,
        issue: str,
        category: str = "clinical-review",
        severity: str = "warning",
    ) -> ToolResult:
        flag = {"category": category, "message": issue, "severity": severity}
        if flag not in self.flags:
            self.flags.append(flag)
        return ToolResult(success=True, data={"flagged": flag})

    @staticmethod
    def finalize() -> ToolResult:
        return ToolResult(success=True, data={"finalize": True})


def _normalize_drug_name(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", value.lower()).strip()
