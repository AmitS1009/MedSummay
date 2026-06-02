from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


MISSING_TEXT = "[MISSING - NOT FOUND IN SOURCE DOCUMENTS]"


class DataStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    MISSING = "MISSING"
    PENDING = "PENDING"
    CONFLICTING = "CONFLICTING"
    UNAVAILABLE = "UNAVAILABLE"


class NoteType(str, Enum):
    ADMISSION = "admission"
    PROGRESS = "progress"
    LAB = "lab"
    MEDICATION = "medication"
    DISCHARGE = "discharge"
    NURSING = "nursing"
    PROCEDURE = "procedure"
    UNKNOWN = "unknown"


class SourceRef(BaseModel):
    note_id: str
    page_number: int | None = None
    path: str | None = None
    evidence: str | None = None

    def label(self) -> str:
        page = f", page {self.page_number}" if self.page_number else ""
        return f"{self.note_id}{page}"


class ClinicalFact(BaseModel):
    value: str | None = None
    status: DataStatus = DataStatus.MISSING
    sources: list[SourceRef] = Field(default_factory=list)
    confidence: float | None = None
    review_note: str | None = None

    @classmethod
    def missing(cls, note: str | None = None) -> "ClinicalFact":
        return cls(value=MISSING_TEXT, status=DataStatus.MISSING, review_note=note)

    def display(self) -> str:
        if self.status == DataStatus.MISSING:
            return MISSING_TEXT
        if self.status == DataStatus.UNAVAILABLE:
            return "[UNAVAILABLE - SOURCE OR TOOL READ FAILED]"
        if self.status == DataStatus.PENDING:
            return f"[PENDING] {self.value or 'Result pending'}"
        if self.status == DataStatus.CONFLICTING:
            return f"[CONFLICT] {self.review_note or self.value or 'Conflicting source values'}"
        return self.value or MISSING_TEXT


class PatientDemographics(BaseModel):
    name: ClinicalFact = Field(default_factory=ClinicalFact.missing)
    mrn: ClinicalFact = Field(default_factory=ClinicalFact.missing)
    dob: ClinicalFact = Field(default_factory=ClinicalFact.missing)
    age: ClinicalFact = Field(default_factory=ClinicalFact.missing)
    sex: ClinicalFact = Field(default_factory=ClinicalFact.missing)


class Diagnosis(BaseModel):
    name: str
    category: str = "secondary"
    status: DataStatus = DataStatus.CONFIRMED
    sources: list[SourceRef] = Field(default_factory=list)


class Medication(BaseModel):
    name: str
    dose: str | None = None
    frequency: str | None = None
    route: str | None = None
    duration: str | None = None
    list_type: str = "unclear"
    documented_reason: str | None = None
    status: DataStatus = DataStatus.CONFIRMED
    sources: list[SourceRef] = Field(default_factory=list)

    def signature(self) -> str:
        return " | ".join(
            value.strip().lower()
            for value in [self.dose, self.frequency, self.route]
            if value and value.strip()
        )


class MedicationChange(BaseModel):
    medication: str
    change_type: str
    admission_details: str | None = None
    discharge_details: str | None = None
    documented_reason: str | None = None
    needs_review: bool = False


class LabResult(BaseModel):
    name: str
    value: str | None = None
    unit: str | None = None
    reference_range: str | None = None
    status: DataStatus = DataStatus.CONFIRMED
    sources: list[SourceRef] = Field(default_factory=list)


class Procedure(BaseModel):
    name: str
    date: str | None = None
    status: DataStatus = DataStatus.CONFIRMED
    sources: list[SourceRef] = Field(default_factory=list)


class ReviewFlag(BaseModel):
    category: str
    message: str
    severity: str = "warning"
    sources: list[SourceRef] = Field(default_factory=list)


class DischargeSummary(BaseModel):
    patient_id: str
    demographics: PatientDemographics = Field(default_factory=PatientDemographics)
    admission_date: ClinicalFact = Field(default_factory=ClinicalFact.missing)
    discharge_date: ClinicalFact = Field(default_factory=ClinicalFact.missing)
    principal_diagnoses: list[Diagnosis] = Field(default_factory=list)
    secondary_diagnoses: list[Diagnosis] = Field(default_factory=list)
    hospital_course: list[ClinicalFact] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    admission_medications: list[Medication] = Field(default_factory=list)
    discharge_medications: list[Medication] = Field(default_factory=list)
    medication_changes: list[MedicationChange] = Field(default_factory=list)
    allergies: ClinicalFact = Field(default_factory=ClinicalFact.missing)
    follow_up_instructions: list[ClinicalFact] = Field(default_factory=list)
    pending_results: list[LabResult] = Field(default_factory=list)
    discharge_condition: ClinicalFact = Field(default_factory=ClinicalFact.missing)
    review_flags: list[ReviewFlag] = Field(default_factory=list)
    step_cap_reached: bool = False


class PageText(BaseModel):
    page_number: int
    text: str = ""
    extraction_method: str = "none"
    success: bool = True
    error: str | None = None


class NoteRecord(BaseModel):
    note_id: str
    path: str
    page_number: int
    note_type: NoteType
    text: str = ""
    extraction_method: str = "none"
    extraction_error: str | None = None

    @classmethod
    def from_json(cls, value: dict[str, Any], base_dir: Path) -> "NoteRecord":
        value = dict(value)
        value["path"] = str(base_dir / value["path"])
        return cls.model_validate(value)


class ToolResult(BaseModel):
    success: bool
    data: Any = None
    error: str | None = None
    retryable: bool = False
    review_required: bool = False


class AgentAction(BaseModel):
    reasoning: str
    tool: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    next_decision: str = ""


class AgentStep(BaseModel):
    number: int
    reasoning: str
    tool: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    result: str
    next_decision: str


class StepTrace(BaseModel):
    patient_id: str
    steps: list[AgentStep] = Field(default_factory=list)
    stop_reason: str = ""

