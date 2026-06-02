from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agent import DischargeSummaryAgent
from src.llm import GroqLLM, LLMError
from src.learning import (
    SimulatedDoctorReviewer,
    normalized_edit_distance,
    run_learning_demo,
    synthetic_cases,
)
from src.models import Medication, NoteRecord, NoteType
from src.pdf_processor import classify_note_type, parse_boundary_ranges
from src.tools import ToolRegistry, evidence_is_supported


class DummyLLM:
    def call_json(self, messages, max_tokens=1200):
        raise RuntimeError("not used")


class RepeatingPlannerLLM:
    def call_json(self, messages, max_tokens=1200):
        return {
            "reasoning": "Keep listing notes to exercise the cap.",
            "tool": "list_notes",
            "inputs": {},
            "next_decision": "List again.",
        }


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry([], DummyLLM())

    def test_source_evidence_must_exist(self):
        source = "Condition at discharge: Hemodynamically stable"
        self.assertTrue(evidence_is_supported("Hemodynamically stable", source))
        self.assertFalse(evidence_is_supported("Patient improved completely", source))

    def test_reconciliation_surfaces_unexplained_changes(self):
        admission = [Medication(name="Lisinopril", dose="10 mg", frequency="daily")]
        discharge = [
            Medication(name="Lisinopril", dose="20 mg", frequency="daily"),
            Medication(name="Aspirin", dose="75 mg", frequency="daily"),
        ]
        result = self.registry.reconcile_medications(
            [item.model_dump(mode="json") for item in admission],
            [item.model_dump(mode="json") for item in discharge],
        )
        changes = {item["medication"]: item for item in result.data["changes"]}
        self.assertEqual(changes["Lisinopril"]["change_type"], "CHANGED")
        self.assertTrue(changes["Lisinopril"]["needs_review"])
        self.assertEqual(changes["Aspirin"]["change_type"], "ADDED")

    def test_mock_interaction_checker_escalates_known_pair(self):
        medications = [Medication(name="Warfarin"), Medication(name="Aspirin")]
        result = self.registry.check_drug_interactions(
            [item.model_dump(mode="json") for item in medications]
        )
        self.assertEqual(len(result.data["interactions"]), 1)
        self.assertEqual(result.data["interactions"][0]["severity"], "major")


class ProcessorTests(unittest.TestCase):
    def test_note_classifier(self):
        self.assertEqual(classify_note_type("CASE RECORD ADMISSION RECORD (1)"), NoteType.ADMISSION)
        self.assertEqual(classify_note_type("CONDITION AT DISCHARGE stable"), NoteType.DISCHARGE)
        self.assertEqual(classify_note_type("BIOCHEMISTRY REPORT"), NoteType.LAB)

    def test_boundary_ranges(self):
        self.assertEqual(parse_boundary_ranges("1-30,31-71"), [(1, 30), (31, 71)])
        with self.assertRaises(ValueError):
            parse_boundary_ranges("1-30,32-71")


class AgentControlTests(unittest.TestCase):
    def test_agent_enforces_hard_iteration_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            note = NoteRecord(
                note_id="note_001",
                path=str(Path(directory) / "note_001.pdf"),
                page_number=1,
                note_type=NoteType.UNKNOWN,
                text="Readable source note",
            )
            agent = DischargeSummaryAgent([note], RepeatingPlannerLLM(), max_steps=3)
            summary, trace = agent.run()
        self.assertTrue(summary.step_cap_reached)
        self.assertEqual(len(trace.steps), 3)
        self.assertIn("Hard step cap reached", trace.stop_reason)


class AuthenticationFailure(Exception):
    status_code = 401


class RejectingCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise AuthenticationFailure("invalid key")


class RejectingClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": RejectingCompletions()})()


class RejectingGroqLLM(GroqLLM):
    def __init__(self):
        super().__init__(max_retries=3)
        self.rejecting_client = RejectingClient()

    @property
    def client(self):
        return self.rejecting_client


class LLMFailureTests(unittest.TestCase):
    def test_authentication_failure_does_not_retry(self):
        llm = RejectingGroqLLM()
        with self.assertRaises(LLMError) as raised:
            llm.call([{"role": "user", "content": "hello"}])
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(llm.rejecting_client.chat.completions.calls, 1)


class LearningTests(unittest.TestCase):
    def test_reward_distance_detects_edits(self):
        self.assertEqual(normalized_edit_distance("same", "same"), 0.0)
        self.assertGreater(normalized_edit_distance("draft", "edited"), 0.0)

    def test_simulated_reviewer_has_hidden_consistent_policy(self):
        reviewer = SimulatedDoctorReviewer()
        edited = reviewer.edit("Name: ray\nMedication | ADDED |\n## Flags for Clinician Review\n")
        self.assertIn("[MISSING - OCR UNCLEAR]", edited)
        self.assertIn("Medication added without documented reconciliation reason", edited)

    def test_learning_demo_improves_heldout_edit_burden(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_learning_demo(Path(directory))
        curve = result["curve"]
        self.assertGreater(curve[0]["avg_edit_distance"], curve[-1]["avg_edit_distance"])
        self.assertEqual(len(synthetic_cases()), 6)


if __name__ == "__main__":
    unittest.main()
