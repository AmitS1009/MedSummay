from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import DischargeSummaryAgent, save_agent_outputs
from .llm import GroqLLM
from .learning import run_learning_demo
from .pdf_processor import load_patient_notes, preprocess_pdf
from .utils import summarize_tool_result


def _preprocess(args: argparse.Namespace) -> int:
    saved, warnings = preprocess_pdf(
        args.pdf,
        output_root=args.data_dir,
        boundaries=args.boundaries,
        ocr_scale=args.ocr_scale,
    )
    print(f"Preprocessed {args.pdf} into {len(saved)} patient folder(s).")
    for patient_id, notes in saved.items():
        readable = sum(bool(note.text) for note in notes)
        print(f"- {patient_id}: {len(notes)} page-note(s), {readable} with OCR text")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


def _run_patient(patient_dir: Path, args: argparse.Namespace) -> bool:
    try:
        notes = load_patient_notes(patient_dir)
        llm = GroqLLM(model=args.model, timeout_seconds=args.timeout, max_retries=args.llm_retries)
        def print_step(step_number, action, result):
            print(f"  STEP {step_number:02d} reasoning: {action.reasoning}", flush=True)
            print(f"           action: {action.tool} {action.inputs}", flush=True)
            print(f"           result: {summarize_tool_result(result, limit=280)}", flush=True)
            print(f"           next:   {action.next_decision}", flush=True)

        agent = DischargeSummaryAgent(notes, llm, max_steps=args.max_steps, on_step=print_step)
        summary, trace = agent.run()
        summary_path, trace_path = save_agent_outputs(summary, trace, output_root=args.output_dir)
        print(
            f"- {summary.patient_id}: {len(trace.steps)} agent step(s), "
            f"{len(summary.review_flags)} review flag(s)"
        )
        print(f"  summary: {summary_path}")
        print(f"  trace:   {trace_path}")
        return True
    except Exception as exc:
        print(f"- {patient_dir.name}: FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def _run(args: argparse.Namespace) -> int:
    patient_dirs: list[Path]
    if args.patient == "all":
        patient_dirs = sorted(Path(args.data_dir).glob("patient_*"))
    else:
        supplied = Path(args.patient)
        patient_dirs = [supplied if supplied.exists() else Path(args.data_dir) / args.patient]
    if not patient_dirs:
        print("No preprocessed patient folders found. Run the preprocess command first.", file=sys.stderr)
        return 1
    print(f"Running discharge-summary agent for {len(patient_dirs)} patient folder(s).")
    outcomes = [_run_patient(patient_dir, args) for patient_dir in patient_dirs]
    return 0 if all(outcomes) else 1


def _learn(args: argparse.Namespace) -> int:
    result = run_learning_demo(args.output_dir)
    print("Part 2 learning demo complete.")
    print(f"- output_dir: {result['output_dir']}")
    print(f"- report: {result['report']}")
    print(f"- memory_size: {result['memory_size']}")
    print("- improvement curve:")
    for point in result["curve"]:
        print(
            f"  iteration {point['iteration']}: "
            f"avg_edit_distance={point['avg_edit_distance']} "
            f"avg_reward={point['avg_reward']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clinical-safe discharge summary drafting agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="OCR and split a source PDF")
    preprocess.add_argument("--pdf", default="patient.pdf", help="Source PDF path")
    preprocess.add_argument("--data-dir", default="data", help="Preprocessed patient folder root")
    preprocess.add_argument(
        "--boundaries",
        help="Operator-confirmed 1-based page ranges, for example: 1-45,46-71",
    )
    preprocess.add_argument("--ocr-scale", type=float, default=1.6, help="OCR render scale")
    preprocess.set_defaults(func=_preprocess)

    run = subparsers.add_parser("run", help="Run the bounded agent on one patient or all patients")
    run.add_argument("--patient", default="all", help="Patient folder, patient ID, or all")
    run.add_argument("--data-dir", default="data", help="Preprocessed patient folder root")
    run.add_argument("--output-dir", default="outputs", help="Output folder root")
    run.add_argument("--model", default=None, help="Groq model override")
    run.add_argument("--timeout", type=float, default=30.0, help="Groq timeout in seconds")
    run.add_argument("--llm-retries", type=int, default=3, help="Groq retry attempts per call")
    run.add_argument("--max-steps", type=int, default=25, help="Hard agent iteration cap")
    run.set_defaults(func=_run)

    learn = subparsers.add_parser("learn", help="Run Part 2 simulated doctor-edit learning demo")
    learn.add_argument(
        "--output-dir",
        default="outputs/part2_learning",
        help="Part 2 learning output folder",
    )
    learn.set_defaults(func=_learn)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
