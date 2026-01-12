# main.py
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
from pipeline import run_training
from plan import Plan

try:
    from llm_agent import make_plan_llm, LLMPlannerError
except Exception:
    make_plan_llm = None
    LLMPlannerError = Exception


DEFAULT_CSV = Path("data") / "titanic.csv"


def _read_instruction(args: argparse.Namespace):
    # Read instruction from --infile or prompt the user
    if args.infile:
        p = Path(args.infile)
        if not p.exists():
            raise FileNotFoundError(f"--infile not found: {p}")
        return p.read_text(encoding="utf-8").strip()

    return input("Enter one instruction message: ").strip()


def _format_output(instruction: str, planner_used: str, plan, used_columns, cm, metrics: dict):
    # Create a format for output
    lines = []
    lines.append("Instruction")
    lines.append("-----------")
    lines.append(instruction)
    lines.append("")

    lines.append("Plan")
    lines.append("----")
    lines.append(f"Model: {plan.model}")
    if plan.keep_only_columns:
        lines.append(f"Keep only: {', '.join(plan.keep_only_columns)}")
    if plan.drop_columns:
        lines.append(f"Drop: {', '.join(plan.drop_columns)}")
    lines.append("")

    lines.append("Used Columns")
    lines.append("--------------------")
    lines.append(", ".join(used_columns) if used_columns else "(none)")
    lines.append("")

    lines.append("Confusion Matrix")
    lines.append("----------------------------------------")
    lines.append(f"{cm[0, 0]:>5} {cm[0, 1]:>5}")
    lines.append(f"{cm[1, 0]:>5} {cm[1, 1]:>5}")
    lines.append("")

    lines.append("Metrics")
    lines.append("-------")
    lines.append(f"Accuracy : {metrics['accuracy']:.4f}")
    lines.append(f"Precision: {metrics['precision']:.4f}")
    lines.append(f"Recall   : {metrics['recall']:.4f}")
    lines.append(f"F1       : {metrics['f1']:.4f}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Titanic ML Agent")
    parser.add_argument("--infile", type=str, default=None, help="Path to .txt file containing the instruction.")
    parser.add_argument("--outfile", type=str, default=None, help="Path to .txt file to save output.")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Path to Titanic CSV")
    args = parser.parse_args(argv)

    try:
        instruction = _read_instruction(args)
        if not instruction:
            print("ERROR: Instruction is empty.", file=sys.stderr)
            return 2

        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
            return 2

        df = pd.read_csv(csv_path)

        # Convert instruction to plan
        if make_plan_llm is None:
            print("ERROR: llm_agent.py is not available.", file=sys.stderr)
            return 3

        plan = make_plan_llm(instruction, available_columns=df.columns.tolist())
        planner_used = "llm"

        # Train and evaluate model
        result = run_training(df, plan)

        # Format output
        output_text = _format_output(
            instruction=instruction,
            planner_used=planner_used,
            plan=plan,
            used_columns=result.used_columns,
            cm=result.confusion,
            metrics=result.metrics,
        )

        print(output_text)

        # Save in outfile
        if args.outfile:
            out_path = Path(args.outfile)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output_text, encoding="utf-8")

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
