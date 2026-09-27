"""Self-check: verify your submitted notebook still has intact grading labels
BEFORE you submit, so you catch problems yourself instead of losing points.

Compares your notebook in `Submitted Work/<topic>/` (or, if not submitted yet,
`Personal Workspace/<topic>/`) against the untouched original in
`Assignments/<topic>/`, and reports any exercise whose grading tag looks
broken -- usually caused by accidentally deleting/duplicating a cell,
splitting/merging cells, or changing a cell's type (code <-> markdown).

This mirrors the same checks the instructor's grading pipeline runs
(ANLP-Admin/autograder/integrity_check.py) -- it's a standalone copy so this
repo has no dependency on ANLP-Admin.

Usage:
    python check_my_submission.py "01 tokenization"
    python check_my_submission.py --all
"""
import argparse
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).parent
ASSIGNMENTS_DIR = ROOT / "Assignments"
SUBMITTED_DIR = ROOT / "Submitted Work"
WORKSPACE_DIR = ROOT / "Personal Workspace"

_MARKER_TEXT_RE = re.compile(r"###\s*BEGIN SOLUTION.*?###\s*END SOLUTION", re.DOTALL)


def _cell_source(cell) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def _find_notebook(directory: Path):
    if not directory.exists():
        return None
    notebooks = [f for f in directory.glob("*.ipynb") if not f.name.startswith(".")]
    return notebooks[0] if notebooks else None


def _walk(nb):
    """Same solution<->grade pairing walk used by the instructor's checker.
    Returns (grade_id_occurrences, resolved) -- see integrity_check.py for details.
    """
    solution_generation = 0
    last_solution_index = None
    last_solution_generation = None

    grade_id_occurrences: dict[str, list[int]] = {}
    resolved: dict[str, dict] = {}

    for idx, cell in enumerate(nb.cells):
        meta = cell.get("metadata", {}).get("nbgrader", {})

        if meta.get("solution"):
            solution_generation += 1
            last_solution_index = idx
            last_solution_generation = solution_generation

        if not meta.get("grade"):
            continue

        grade_id = meta.get("grade_id")
        if not grade_id:
            continue

        grade_id_occurrences.setdefault(grade_id, []).append(idx)

        if meta.get("solution"):
            answer_index = idx
            answer_generation = solution_generation
            answer_cell_type = cell.cell_type
            source = _cell_source(cell)
        else:
            answer_index = last_solution_index
            answer_generation = last_solution_generation
            src_cell = nb.cells[last_solution_index] if last_solution_index is not None else None
            answer_cell_type = src_cell.cell_type if src_cell is not None else None
            source = _cell_source(src_cell) if src_cell is not None else ""

        resolved[grade_id] = {
            "answer_index": answer_index,
            "answer_generation": answer_generation,
            "answer_cell_type": answer_cell_type,
            "has_markers": bool(_MARKER_TEXT_RE.search(source)) if source else False,
        }

    return grade_id_occurrences, resolved


def check_topic(topic: str) -> list[tuple[str, str, str]]:
    """Returns a list of (severity, grade_id, message) for one assignment topic.
    severity is "block" or "warn". Empty list means everything looks fine.
    """
    reference_path = _find_notebook(ASSIGNMENTS_DIR / topic)
    if reference_path is None:
        return [("block", "", f"Could not find the original notebook in Assignments/{topic}/.")]

    submission_path = _find_notebook(SUBMITTED_DIR / topic) or _find_notebook(WORKSPACE_DIR / topic)
    if submission_path is None:
        return [("block", "", f"No notebook found in 'Submitted Work/{topic}/' or 'Personal Workspace/{topic}/' yet.")]

    ref_nb = nbformat.read(reference_path, as_version=4)
    sub_nb = nbformat.read(submission_path, as_version=4)

    ref_occurrences, ref_resolved = _walk(ref_nb)
    expected = {
        grade_id: ref_resolved[grade_id]["answer_cell_type"]
        for grade_id in ref_occurrences
        if len(ref_occurrences[grade_id]) == 1  # skip if the original itself looks malformed
    }

    grade_id_occurrences, resolved = _walk(sub_nb)

    findings: list[tuple[str, str, str]] = []
    seen_answer_generations: dict[int, str] = {}

    for grade_id, expected_cell_type in expected.items():
        occurrences = grade_id_occurrences.get(grade_id, [])

        if not occurrences:
            findings.append(("block", grade_id,
                "This question's grading tag is missing from your notebook. This usually happens "
                "when a graded cell got deleted and recreated, split, or merged with another cell. "
                "Copy a clean version of this cell from Assignments/ and re-enter your answer there."))
            continue

        if len(occurrences) > 1:
            findings.append(("block", grade_id,
                f"This question's grading tag appears on {len(occurrences)} different cells "
                "(likely duplicated by copy-paste). Remove the extra copies, keep only one."))
            continue

        info = resolved[grade_id]

        if info["answer_index"] is None:
            findings.append(("block", grade_id,
                "The editable answer cell for this question seems to have been deleted entirely. "
                "Restore it from a clean copy in Assignments/."))
            continue

        generation = info["answer_generation"]
        if generation in seen_answer_generations:
            earlier = seen_answer_generations[generation]
            findings.append(("block", grade_id,
                f"This question appears to be sharing its answer cell with '{earlier}' -- its own "
                "editable solution cell seems to have lost its tag or been deleted. Restore it from "
                "a clean copy in Assignments/."))
            continue
        seen_answer_generations[generation] = grade_id

        if info["answer_cell_type"] != expected_cell_type:
            findings.append(("block", grade_id,
                f"This should be a {expected_cell_type} cell, but yours is a {info['answer_cell_type']} "
                "cell. Don't convert code cells to markdown (or vice versa) -- restore the original type."))
            continue

        if not info["has_markers"]:
            findings.append(("warn", grade_id,
                "The '### BEGIN SOLUTION' / '### END SOLUTION' markers are missing from this cell. "
                "Your answer will probably still be graded, but please don't remove these markers."))

    unknown_ids = set(grade_id_occurrences) - set(expected)
    for grade_id in sorted(unknown_ids):
        findings.append(("warn", grade_id, "Unexpected grading tag found that doesn't belong here."))

    return findings


def _topics() -> list[str]:
    return sorted(d.name for d in ASSIGNMENTS_DIR.iterdir() if d.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topic", nargs="?", help='Assignment folder name, e.g. "01 tokenization"')
    parser.add_argument("--all", action="store_true", help="Check every assignment you have a notebook for")
    args = parser.parse_args()

    if not args.topic and not args.all:
        parser.error('Pass an assignment name, e.g. "01 tokenization", or --all')

    topics = _topics() if args.all else [args.topic]

    any_block = False
    for topic in topics:
        findings = check_topic(topic)
        blocks = [f for f in findings if f[0] == "block"]
        warns = [f for f in findings if f[0] == "warn"]

        if not findings:
            print(f"[OK]   {topic}: no problems found.")
            continue

        print(f"\n{topic}:")
        for severity, grade_id, message in blocks + warns:
            label = "BLOCK" if severity == "block" else "WARN "
            prefix = f"[{grade_id}] " if grade_id else ""
            print(f"  {label} {prefix}{message}")

        if blocks:
            any_block = True

    print()
    if any_block:
        print("Please fix the BLOCK item(s) above and re-check before submitting.")
        return 1

    print("No blocking issues found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
