from __future__ import annotations

import json
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import pdfplumber
from pypdf import PdfReader, PdfWriter

from .models import NoteRecord, NoteType, PageText

try:
    import fitz  # type: ignore
except ImportError:  # Optional fast path. pdfplumber remains available.
    fitz = None


def _direct_text(path: Path) -> list[str]:
    if fitz is not None:
        with fitz.open(path) as document:
            return [page.get_text("text") or "" for page in document]
    with pdfplumber.open(path) as document:
        return [page.extract_text() or "" for page in document.pages]


def _ocr_page(path: Path, page_index: int, scale: float) -> str:
    try:
        import pypdfium2 as pdfium
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("OCR dependencies are unavailable") from exc
    document = pdfium.PdfDocument(str(path))
    try:
        image = document[page_index].render(scale=scale).to_pil()
        return pytesseract.image_to_string(image, config="--psm 6")
    finally:
        document.close()


def extract_text_from_pdf(
    path: str | Path,
    *,
    ocr_fallback: bool = True,
    ocr_scale: float = 1.6,
    minimum_direct_chars: int = 40,
) -> list[PageText]:
    source = Path(path)
    direct_pages = _direct_text(source)
    pages: list[PageText] = []
    for index, direct in enumerate(direct_pages):
        text = direct.strip()
        if len(text) >= minimum_direct_chars:
            pages.append(PageText(page_number=index + 1, text=text, extraction_method="direct"))
            continue
        if not ocr_fallback:
            pages.append(
                PageText(
                    page_number=index + 1,
                    text=text,
                    extraction_method="direct-empty",
                    success=False,
                    error="Direct extraction returned insufficient text; OCR fallback disabled",
                )
            )
            continue
        try:
            ocr_text = _ocr_page(source, index, ocr_scale).strip()
            pages.append(
                PageText(
                    page_number=index + 1,
                    text=ocr_text,
                    extraction_method="ocr",
                    success=bool(ocr_text),
                    error=None if ocr_text else "OCR returned empty text",
                )
            )
        except Exception as exc:
            pages.append(
                PageText(
                    page_number=index + 1,
                    text="",
                    extraction_method="ocr-failed",
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return pages


def classify_note_type(text: str) -> NoteType:
    normalized = " ".join(text.upper().split())
    discharge_markers = ["DISCHARGE SUMMARY", "ADVICE ON DISCHARGE", "CONDITION AT DISCHARGE"]
    if any(value in normalized for value in discharge_markers) or (
        "DIAGNOSIS:" in normalized and "COURSE IN THE HOSPITAL" in normalized
    ):
        return NoteType.DISCHARGE
    if any(value in normalized for value in ["ADMISSION RECORD", "NURSING ASSESSMENT ON ADMISSION"]):
        return NoteType.ADMISSION
    if any(value in normalized for value in ["BIOCHEMISTRY REPORT", "CLINICAL PATHOLOGY REPORT", "INVESTIGATION RESULT"]):
        return NoteType.LAB
    if any(value in normalized for value in ["DRUG CHART", "DRUGS IN ONLY CAPITAL", "MEDICATION"]):
        return NoteType.MEDICATION
    if any(value in normalized for value in ["CONSULTATION SHEET", "PROGRESS NOTE"]):
        return NoteType.PROGRESS
    if any(value in normalized for value in ["NURSING DOCUMENTATION", "NURSES NOTES", "BED SORES"]):
        return NoteType.NURSING
    if any(value in normalized for value in ["PROCEDURE", "OPERATIVE NOTE"]):
        return NoteType.PROCEDURE
    return NoteType.UNKNOWN


def parse_boundary_ranges(value: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for token in value.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", token)
        if not match:
            raise ValueError(f"Invalid boundary range: {token!r}. Expected format: 1-30,31-71")
        start, end = int(match.group(1)), int(match.group(2))
        if start < 1 or end < start:
            raise ValueError(f"Invalid boundary range: {token!r}")
        ranges.append((start, end))
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] != previous[1] + 1:
            raise ValueError("Boundary ranges must be contiguous and non-overlapping")
    return ranges


def _auto_boundary_ranges(pages: list[PageText]) -> tuple[list[tuple[int, int]], list[str]]:
    warnings = [
        "Automatic patient splitting is heuristic. Confirm boundaries before clinical review."
    ]
    starts = [1]
    for page in pages:
        normalized = " ".join(page.text.upper().split())
        strong_admission_reset = "ADMISSION RECORD (1)" in normalized
        strong_discharge_reset = "DISCHARGE SUMMARY" in normalized and page.page_number > 1
        if page.page_number > starts[-1] + 5 and (strong_admission_reset or strong_discharge_reset):
            starts.append(page.page_number)
    ranges: list[tuple[int, int]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] - 1 if index + 1 < len(starts) else len(pages)
        ranges.append((start, end))
    if len(ranges) == 1:
        warnings.append("No confident patient boundary was detected; generated a single reviewable patient folder.")
    else:
        warnings.append(f"Detected {len(ranges)} possible patient groups: {ranges}")
    return ranges, warnings


def split_into_patients(
    pages: list[PageText],
    boundary_ranges: list[tuple[int, int]] | None = None,
) -> tuple[OrderedDict[str, list[PageText]], list[str]]:
    if boundary_ranges is None:
        boundary_ranges, warnings = _auto_boundary_ranges(pages)
    else:
        warnings = ["Using explicit operator-supplied patient page boundaries."]
    if boundary_ranges and boundary_ranges[-1][1] > len(pages):
        raise ValueError(f"Boundary range exceeds PDF length ({len(pages)} pages)")
    grouped: OrderedDict[str, list[PageText]] = OrderedDict()
    for index, (start, end) in enumerate(boundary_ranges, start=1):
        grouped[f"patient_{index:03d}"] = pages[start - 1 : end]
    return grouped, warnings


def save_patient_notes(
    source_pdf: str | Path,
    patient_id: str,
    pages: Iterable[PageText],
    output_root: str | Path = "data",
) -> list[NoteRecord]:
    source = Path(source_pdf)
    destination = Path(output_root) / patient_id
    destination.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(source))
    records: list[NoteRecord] = []
    for page in pages:
        note_type = classify_note_type(page.text)
        filename = f"page_{page.page_number:03d}_{note_type.value}.pdf"
        writer = PdfWriter()
        writer.add_page(reader.pages[page.page_number - 1])
        with (destination / filename).open("wb") as handle:
            writer.write(handle)
        records.append(
            NoteRecord(
                note_id=filename.removesuffix(".pdf"),
                path=filename,
                page_number=page.page_number,
                note_type=note_type,
                text=page.text,
                extraction_method=page.extraction_method,
                extraction_error=page.error,
            )
        )
    payload = {
        "patient_id": patient_id,
        "source_pdf": str(source),
        "notes": [record.model_dump(mode="json") for record in records],
    }
    (destination / "index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return records


def preprocess_pdf(
    source_pdf: str | Path,
    *,
    output_root: str | Path = "data",
    boundaries: str | None = None,
    ocr_scale: float = 1.6,
) -> tuple[OrderedDict[str, list[NoteRecord]], list[str]]:
    pages = extract_text_from_pdf(source_pdf, ocr_scale=ocr_scale)
    ranges = parse_boundary_ranges(boundaries) if boundaries else None
    grouped, warnings = split_into_patients(pages, ranges)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    for stale_patient_dir in destination.glob("patient_*"):
        if stale_patient_dir.is_dir():
            shutil.rmtree(stale_patient_dir)
    saved: OrderedDict[str, list[NoteRecord]] = OrderedDict()
    for patient_id, patient_pages in grouped.items():
        saved[patient_id] = save_patient_notes(source_pdf, patient_id, patient_pages, output_root)
    empty_pages = [str(page.page_number) for page in pages if not page.success]
    if empty_pages:
        warnings.append(f"OCR returned empty or failed text for pages: {', '.join(empty_pages)}")
    return saved, warnings


def load_patient_notes(patient_dir: str | Path) -> list[NoteRecord]:
    directory = Path(patient_dir)
    payload = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    return [NoteRecord.from_json(value, directory) for value in payload["notes"]]
