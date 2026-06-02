# MedDraft - Clinician-Review Discharge Summary Agent

MedDraft is an agentic AI system that reads messy scanned patient notes and produces a structured discharge-summary draft for clinician review. It is built for clinical safety first: when a fact cannot be sourced, the system marks it missing, pending, conflicting, or escalated instead of guessing.

The repository includes both required Part 1 outputs and the Part 2 learning-from-edits extension.

## What Is Included

```text
patient.pdf                       scanned source PDF
src/                              source code
data/patient_*/                   OCR-processed per-patient note folders
outputs/patient_*/                generated summaries and step traces
outputs/part2_learning/           Part 2 simulated edit-learning results
tests/                            safety and learning tests
README.md                         this document
```

Generated outputs are already present:

- `outputs/patient_001/discharge_summary.md`
- `outputs/patient_001/step_trace.md`
- `outputs/patient_002/discharge_summary.md`
- `outputs/patient_002/step_trace.md`
- `outputs/part2_learning/report.md`
- `outputs/part2_learning/metrics.csv`
- `outputs/part2_learning/correction_memory.json`

## How To Run

Create `.env` from `.env.example` and add your Groq key:

```bash
GROQ_API_KEY=your_key_here
GROQ_MODEL=qwen/qwen3-32b
```

Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
venv/bin/python -m unittest discover -s tests -v
```

Preprocess the scanned PDF:

```bash
venv/bin/python -m src.main preprocess --pdf patient.pdf --boundaries 1-23,24-71
```

Run the Part 1 discharge-summary agent:

```bash
venv/bin/python -m src.main run --patient all --max-steps 25
```

Run the Part 2 learning demo:

```bash
venv/bin/python -m src.main learn
```

## Part 1: Agent Loop Design

The main agent loop is implemented from scratch in `src/agent.py`. It follows a bounded ReAct-style cycle:

```text
plan -> choose tool -> execute -> observe -> update draft -> re-plan
```

The agent starts with a note catalog, then decides which tool to call next. It can read individual notes, search notes, extract structured facts, reconcile medications, check mock drug interactions, flag clinician-review issues, and finalize. Each patient run has a hard `max_steps=25` cap, repeated-action protection, and a deterministic fallback path if the planner or a tool fails.

Each step is written to `step_trace.md` with:

- brief reasoning
- selected tool/action
- tool inputs
- observed result
- next decision

This gives an auditable trace of how the draft was produced.

## PDF Ingestion

The source PDF is scanned, so direct PDF text extraction returns little or no text. The pipeline therefore uses OCR fallback and saves each page as a per-patient note with provenance:

- original page number
- note type classification
- OCR text
- extraction method
- extraction warnings

The supplied PDF has out-of-order scanned forms, so the run command uses confirmed page boundaries:

```text
patient_001: pages 1-23
patient_002: pages 24-71
```

Unreadable OCR pages are reported instead of hidden.

## No-Fabrication Guardrail

The core safety rule is: **no clinical fact is accepted unless it is supported by source text.**

Every LLM extraction prompt requires a short evidence excerpt. The tool layer then checks that the excerpt appears in the OCR text. If evidence is missing or unsupported, that extraction is rejected and flagged for clinician review.

Required fields that cannot be sourced are rendered as:

```text
[MISSING - NOT FOUND IN SOURCE DOCUMENTS]
```

Pending items remain `[PENDING]`. Conflicting facts are marked `[CONFLICT]`. Medication changes without a documented reason are not silently resolved; they are flagged for reconciliation.

## Failure, Conflict, and Safety Handling

The system is designed to fail visibly:

- Empty OCR pages are reported during preprocessing.
- Groq/API failures are retried, then surfaced in the trace and final flags.
- Repeated actions are blocked by a loop guard.
- If planning fails, deterministic fallback reads high-yield notes and still finalizes safely.
- Missing required sections are explicitly flagged.
- Medication reconciliation compares admission vs discharge medications.
- Added, stopped, or changed medications without documented reasons are flagged.
- A mock drug-interaction checker escalates known unsafe combinations.

The draft is never auto-finalized as a clinical document. It is always labeled as a draft for clinician review.

## Part 2: Learning From Doctor Edits

Part 2 is implemented in `src/learning.py` as a synthetic feedback loop that does not modify the Part 1 safety agent.

### Reward Signal

The reward is based on edit burden:

```text
reward = 1 - normalized_edit_distance(memory_improved_draft, simulated_doctor_edit)
```

Lower edit distance means the simulated clinician had to do less editing, so reward is higher.

### Simulated Reviewer

Because no real doctor-edited data is provided, the project includes a hidden simulated reviewer. It applies a consistent editing policy to drafts:

- expands unsafe shorthand like `[MISSING]`
- standardizes pending-result follow-up wording
- cleans common OCR artifacts
- adds a review flag when a medication is added without a documented reconciliation reason

This produces `(draft, edited)` pairs for training and evaluation.

### Learning Mechanism

The learner is a structured correction memory. It learns recurring safe replacements and section insertions from accumulated edit pairs, then applies those corrections to future drafts before simulated review.

It is deliberately constrained: it can improve style, OCR cleanup, and explicit review flags, but it does not create new diagnoses, medication reasons, dates, lab values, or other clinical facts.

### Measured Results

The Part 2 demo evaluates on held-out synthetic cases and writes results to `outputs/part2_learning/report.md`.

Current result:

```text
Before average edit distance: 0.5653
After average edit distance: 0.013
Relative improvement: 97.7%
```

Improvement curve:

```text
iteration 0: avg_edit_distance=0.5653 avg_reward=0.4347
iteration 1: avg_edit_distance=0.0319 avg_reward=0.9682
iteration 2: avg_edit_distance=0.0319 avg_reward=0.9682
iteration 3: avg_edit_distance=0.0130 avg_reward=0.9870
iteration 4: avg_edit_distance=0.0130 avg_reward=0.9870
```
