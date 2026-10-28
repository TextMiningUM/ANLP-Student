"""
Test all release notebooks in ANLP-Student/Assignments folder.
- Makes a temporary COPY of each notebook
- Runs the copy with --allow-errors (continues past NotImplementedError from exercises)
- Original notebooks remain 100% UNTOUCHED
- Cleans up test copies after completion

Usage: python test_release_notebooks.py [notebook_number]
  e.g.: python test_release_notebooks.py 01
        python test_release_notebooks.py A1
        python test_release_notebooks.py all
"""
import subprocess
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime
import shutil

# Load .env file from Admin folder
env_file = Path(__file__).parent.parent / 'ANLP-Admin' / '.env'
if env_file.exists():
    print(f"Loading environment from: {env_file}\n")
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
else:
    print(f"⚠️  Warning: .env file not found at {env_file}")

# Notebook paths relative to ANLP-Student
NOTEBOOKS = [
    "Assignments/01 tokenization/01_ANLP_Tokenization_2026_2027.ipynb",
    "Assignments/02 document_representation/02_ANLP_Document_Representation_2026_2027.ipynb",
    "Assignments/03 measuring_quality/03_ANLP_Measuring_Quality_2026_2027.ipynb",
    "Assignments/04 syntax and semantics/04_ANLP_Grammar_Based_NLP_2026_2027.ipynb",
    "Assignments/05 statistical NLP/05_ANLP_Statistical_Language_Processing_2026_2027.ipynb",
    "Assignments/06 pytorch/06_ANLP_Introduction_to_PyTorch_2026_2027.ipynb",
    "Assignments/07 transformers/07_ANLP_Deep_Learning_Transformers_2026_2027.ipynb",
    "Assignments/08 encoder models/08_ANLP_Transformers_BERT_2026_2027.ipynb",
    "Assignments/09 decoder models/09_ANLP_Decoder_Models_GPT_2026_2027.ipynb",
    "Assignments/10 XAI in NLP/10_ANLP_XAI_for_NLP_2026_2027.ipynb",
    "Assignments/11 fine-tuning LLMs/11_ANLP_Fine_Tuning_LLMs_2026_2027.ipynb",
    "Assignments/12 multi-modal models/12_ANLP_Multimodal_Tutorial_2026_2027.ipynb",
    "Assignments/13 speech recognition/13_ANLP_Speech_Recognition_2026_2027.ipynb",
    "Assignments/14 abstracting and MT/14_ANLP_Abstracting_and_Machine_Translation_2026_2027.ipynb",
    "Assignments/15 dialog/15_ANLP_Dialogue_2026_2027.ipynb",
    "Assignments/16 agents/16_ANLP_Agents_2026_2027.ipynb",
    "Assignments/A1 NLP fundamentals/A1_ANLP_NLP_Fundamentals_2026_2027.ipynb",
    "Assignments/A2 deep learning for NLP/A2_ANLP_Deep_Learning_for_NLP_2026_2027.ipynb",
    "Assignments/A3 NLP applications/A3_ANLP_NLP_Applications_2026_2027.ipynb",
]

# Known non-critical errors per notebook (benign/expected failures unrelated to
# the self-hosted dataset/notebook changes, e.g. platform quirks or missing
# optional system binaries). Mirrors the allowlist used by
# ANLP-Admin/batch_test_all_notebooks.py so the same known issues aren't
# re-flagged as new critical failures here.
KNOWN_NON_CRITICAL = {
    '12': ['WinError 127', 'torchaudio'],
    '13': ['ValueError: Expected input batch_size', 'Whisper expects the mel input',
           'AssertionError: sample_indices must be defined', 'AssertionError: test_accuracy must be defined',
           'AssertionError: cnn_test_accuracy must be defined', 'AssertionError: Must test at least 3 SNR values',
           'RuntimeError: shape', "NameError: name 'test_loader' is not defined"],
    '14': ['ExecutableNotFound: failed to execute', 'GraphViz', 'AlignedSent', 'IndexError: list index out of range'],
}


def get_notebook_key(notebook_rel_path):
    """Extract the short identifier (e.g. '01', '08', 'A1') from a notebook path."""
    folder_name = Path(notebook_rel_path).parent.name  # e.g. "08 encoder models"
    return folder_name.split(" ")[0]


def classify_cell_errors(executed_notebook_path, nb_key):
    """
    Inspect the executed (temporary) notebook's cell outputs and classify any
    errors found:
      - Expected: a NotImplementedError raised inside an exercise cell
        (nbgrader 'solution' or 'task' cell) - this is normal for an
        unanswered release/student notebook and is not reported at all.
      - Non-critical: matches a known benign pattern in KNOWN_NON_CRITICAL
        for this notebook (platform quirks, cascading effects of earlier
        stubs, etc.) - reported as a warning.
      - Critical: anything else - a real, unexpected failure that would not
        occur in a correctly working notebook/infrastructure.

    Returns (critical_errors, non_critical_errors, expected_count) where the
    first two are lists of {'cell', 'ename', 'evalue'} dicts.
    """
    with open(executed_notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    known_patterns = KNOWN_NON_CRITICAL.get(nb_key, [])
    critical_errors = []
    non_critical_errors = []
    expected_count = 0

    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        nbgrader_meta = cell.get('metadata', {}).get('nbgrader', {})
        is_exercise = nbgrader_meta.get('solution', False) or nbgrader_meta.get('task', False)

        for output in cell.get('outputs', []):
            if output.get('output_type') != 'error':
                continue
            ename = output.get('ename', 'Unknown')
            evalue = output.get('evalue', '')
            traceback_text = ''.join(output.get('traceback', []))
            error_info = {'cell': i + 1, 'ename': ename, 'evalue': evalue}

            if is_exercise and ename == 'NotImplementedError':
                expected_count += 1
                continue

            error_text = f"{ename} {evalue} {traceback_text}"
            if any(pattern in error_text for pattern in known_patterns):
                non_critical_errors.append(error_info)
            else:
                critical_errors.append(error_info)

    return critical_errors, non_critical_errors, expected_count


def count_exercise_cells(notebook_path):
    """
    Count how many exercise cells (solution/task) are in the notebook.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    exercise_count = 0
    for cell in nb['cells']:
        metadata = cell.get('metadata', {})
        nbgrader = metadata.get('nbgrader', {})
        if nbgrader.get('solution', False) or nbgrader.get('task', False):
            exercise_count += 1
    
    return len(nb['cells']), exercise_count


def test_notebook(notebook_path, python_exec):
    """
    Test a notebook by copying it and executing with --allow-errors.
    The original notebook remains completely untouched.
    Returns (success, duration, message, cells_info)
    """
    notebook_path = Path(notebook_path)
    
    # Count cells and exercises for reporting
    total_cells, exercise_count = count_exercise_cells(notebook_path)
    
    # Create temp copy with unique name
    temp_copy = notebook_path.parent / f".test_{notebook_path.name}"
    output_file = notebook_path.parent / f".test_{notebook_path.stem}_executed.ipynb"
    
    try:
        # Copy notebook to temp location
        shutil.copy(notebook_path, temp_copy)
        
        print(f"📝 Cells: {total_cells} total, {exercise_count} exercises (will hit NotImplementedError)")
        
        # Execute with --allow-errors (continues past NotImplementedError from exercises)
        # NOTE: invoke nbconvert directly (python -m nbconvert), not via the
        # `jupyter` command dispatcher (python -m jupyter nbconvert) - the
        # dispatcher fails silently in this environment for unrelated reasons.
        start_time = time.time()
        result = subprocess.run(
            [
                python_exec, '-m', 'nbconvert',
                '--execute',
                '--allow-errors',  # Continue past NotImplementedError
                '--to', 'notebook',
                '--ExecutePreprocessor.timeout=600',
                '--ExecutePreprocessor.kernel_name=anlp313',
                '--output', str(output_file),
                str(temp_copy)
            ],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max
        )
        duration = time.time() - start_time
        
        # A file is written even if some cells errored (--allow-errors), so
        # process-level completion alone is not enough to call this a pass.
        # Inspect per-cell outputs to tell expected NotImplementedError /
        # known-benign issues apart from real, unexpected failures.
        process_completed = output_file.exists()
        nb_key = get_notebook_key(notebook_path)
        
        if process_completed:
            critical_errors, non_critical_errors, expected_count = classify_cell_errors(output_file, nb_key)
            
            if critical_errors:
                status = 'FAILED'
            elif non_critical_errors:
                status = 'PASSED_WITH_WARNINGS'
            else:
                status = 'PASSED'
            
            if status == 'PASSED':
                print(f"✅ PASSED - {duration:.1f}s (original notebook unchanged)")
                message = f"Success - {expected_count} exercises encountered NotImplementedError (expected)"
            elif status == 'PASSED_WITH_WARNINGS':
                print(f"⚠️  PASSED WITH WARNINGS - {duration:.1f}s ({len(non_critical_errors)} known non-critical error(s))")
                for err in non_critical_errors[:3]:
                    print(f"    Cell {err['cell']}: {err['ename']}: {err['evalue'][:100]}")
                message = f"Passed with {len(non_critical_errors)} known non-critical error(s)"
            else:
                print(f"❌ FAILED - {duration:.1f}s ({len(critical_errors)} unexpected critical error(s))")
                for err in critical_errors[:5]:
                    print(f"    Cell {err['cell']}: {err['ename']}: {err['evalue'][:150]}")
                message = f"Failed: {len(critical_errors)} unexpected error(s) - first: {critical_errors[0]['ename']}: {critical_errors[0]['evalue'][:150]}"
        else:
            status = 'EXECUTION_ERROR'
            critical_errors, non_critical_errors, expected_count = [], [], 0
            print(f"❌ EXECUTION ERROR - nbconvert process did not complete:")
            error_output = result.stderr if result.stderr else result.stdout
            print(error_output[-500:] if len(error_output) > 500 else error_output)
            message = f"Execution error: {error_output[-200:]}"
        
        cells_info = {
            'total': total_cells,
            'exercises': exercise_count
        }
        
        return status, duration, message, cells_info, critical_errors, non_critical_errors
        
    finally:
        # Clean up temp files
        if temp_copy.exists():
            temp_copy.unlink()
        if output_file.exists():
            output_file.unlink()


def main():
    """
    Main test runner
    """
    if len(sys.argv) < 2:
        print("Usage: python test_release_notebooks.py [notebook_number|all]")
        print("Examples:")
        print("  python test_release_notebooks.py 01")
        print("  python test_release_notebooks.py A1")
        print("  python test_release_notebooks.py all")
        sys.exit(1)
    
    target = sys.argv[1]
    
    # Get Python executable
    python_exec = sys.executable
    print(f"Using Python: {python_exec}\n")
    
    # Get base directory
    base_dir = Path(__file__).parent
    
    # Determine which notebooks to test
    if target.lower() == 'all':
        notebooks_to_test = NOTEBOOKS
    else:
        # Find notebook matching the target
        notebooks_to_test = [nb for nb in NOTEBOOKS if target.upper() in nb]
        if not notebooks_to_test:
            print(f"❌ No notebook found matching '{target}'")
            sys.exit(1)
    
    # Test each notebook
    results = []
    start_time = time.time()
    
    for notebook_rel_path in notebooks_to_test:
        notebook_path = base_dir / notebook_rel_path
        
        if not notebook_path.exists():
            print(f"⚠️  Notebook not found: {notebook_path}")
            continue
        
        print("=" * 80)
        print(f"Testing: {notebook_path}")
        print("=" * 80)
        
        try:
            status, duration, message, cells_info, critical_errors, non_critical_errors = test_notebook(notebook_path, python_exec)
            results.append({
                'notebook': str(notebook_rel_path),
                'status': status,
                'duration': duration,
                'message': message,
                'cells_info': cells_info,
                'critical_errors': critical_errors,
                'non_critical_errors': non_critical_errors
            })
        except Exception as e:
            print(f"❌ Exception during test: {e}")
            results.append({
                'notebook': str(notebook_rel_path),
                'status': 'EXECUTION_ERROR',
                'duration': 0,
                'message': f"Exception: {str(e)}",
                'cells_info': {'total': 0, 'exercises': 0},
                'critical_errors': [],
                'non_critical_errors': []
            })
        
        print()  # Blank line between tests
    
    total_time = time.time() - start_time
    
    # Generate summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results if r['status'] == 'PASSED')
    warned = sum(1 for r in results if r['status'] == 'PASSED_WITH_WARNINGS')
    failed = sum(1 for r in results if r['status'] in ('FAILED', 'EXECUTION_ERROR'))
    total_exercises = sum(r['cells_info']['exercises'] for r in results)
    
    print(f"Total notebooks tested: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"⚠️  Passed with warnings: {warned}")
    print(f"❌ Failed: {failed}")
    print(f"📚 Total exercises skipped: {total_exercises}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print()
    
    # Detailed results
    print("Detailed Results:")
    status_symbol = {'PASSED': '✅', 'PASSED_WITH_WARNINGS': '⚠️ ', 'FAILED': '❌', 'EXECUTION_ERROR': '❌'}
    for r in results:
        symbol = status_symbol.get(r['status'], '❓')
        nb_name = Path(r['notebook']).name
        print(f"  {symbol} {nb_name:50s} {r['duration']:6.1f}s  {r['cells_info']['total']:3d} cells, {r['cells_info']['exercises']:2d} exercises  [{r['status']}]")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_time': total_time,
        'notebooks_tested': len(results),
        'passed': passed,
        'passed_with_warnings': warned,
        'failed': failed,
        'total_exercises_skipped': total_exercises,
        'results': results
    }
    
    report_file = base_dir / 'notebook_test_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_file}")
    
    # Exit with error code only on real (critical) failures; warnings are OK
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
