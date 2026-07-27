# ANLP 2026-2027 — Student Repository

**Advanced Natural Language Processing** — Maastricht University  
Department of Advanced Computer Sciences, Faculty of Science and Engineering

## Structure

```
Course Materials/   ← Reference materials and lecture notes (read-only)
Assignments/        ← Graded assignments (read-only originals)
  01 tokenization/
  02 document_representation/
  03 measuring_quality/
  04 syntax and semantics/
  05 statistical NLP/
  06 pytorch/
  07 transformers/
  08 encoder models/
  09 decoder models/
  10 XAI in NLP/
  11 fine-tuning LLMs/
  12 multi-modal models/
  13 speech recognition/
  14 abstracting and MT/
  15 dialog/
  16 agents/
  A1 NLP fundamentals/
  A2 deep learning for NLP/
  A3 NLP applications/
Personal Workspace/ ← Your working directory (copy notebooks here to work on them)
  01 tokenization/
  02 document_representation/
  03 measuring_quality/
  04 syntax and semantics/
  05 statistical NLP/
  06 pytorch/
  07 transformers/
  08 encoder models/
  09 decoder models/
  10 XAI in NLP/
  11 fine-tuning LLMs/
  12 multi-modal models/
  13 speech recognition/
  14 abstracting and MT/
  15 dialog/
  16 agents/
  A1 NLP fundamentals/
  A2 deep learning for NLP/
  A3 NLP applications/
Submitted Work/     ← Copy your finished notebooks here to submit
  01 tokenization/
  02 document_representation/
  03 measuring_quality/
  04 syntax and semantics/
  05 statistical NLP/
  06 pytorch/
  07 transformers/
  08 encoder models/
  09 decoder models/
  10 XAI in NLP/
  11 fine-tuning LLMs/
  12 multi-modal models/
  13 speech recognition/
  14 abstracting and MT/
  15 dialog/
  16 agents/
  A1 NLP fundamentals/
  A2 deep learning for NLP/
  A3 NLP applications/
```

## Getting Started

### On JupyterHub (`anlp-course-um.nl`)

1. Open a terminal in JupyterLab
2. Clone this repository:
   ```bash
   git clone https://github.com/TextMiningUM/ANLP-Student.git
   cd ANLP-Student
   ```
3. Copy the assignment notebook to your `Personal Workspace/` folder (see workflow below)
4. Work on the notebook in `Personal Workspace/`
5. When finished, copy it to `Submitted Work/` to submit

### Fetching New Assignments

When new assignments are released, pull the latest changes:
```bash
cd ANLP-Student
git pull
```

## Course Materials

The `Course Materials/` folder contains lecture slides, reference notebooks, 
and supplementary materials for self-study. These are not graded but provide
valuable context for the assignments.

## Assignments

Assignment notebooks are in `Assignments/<topic>/`. Each notebook contains:
- Instructional content explaining the concepts
- **Exercise cells** where you write your code (marked with `# YOUR CODE HERE`)
- **Text answer cells** where you write explanations (marked with `YOUR ANSWER HERE`)
- **Test cells** that validate your solution (do not modify these)

> **Note:** Some assignments use a shared `dataset_loader.py` in the `Assignments/` folder.
> This file provides helper functions for loading datasets. You don't need to modify it,
> but you may import functions from it in your notebooks.

### Assignment Categories

The course includes **16 core assignments** (01-16) covering:
- **Foundations** (01-05): Tokenization, document representation, quality metrics, syntax/semantics, statistical NLP
- **Deep Learning** (06-10): PyTorch, transformers, encoder models, decoder models, explainability
- **Advanced Topics** (11-16): Fine-tuning LLMs, multi-modal models, speech recognition, MT, dialog, agents

Plus **3 major assignments** (A1-A3) that integrate multiple topics.

## Workflow

1. **Copy** the assignment notebook from `Assignments/` to the matching
   `Personal Workspace/` folder. For example:
   ```bash
   cp "Assignments/01 tokenization/01_ANLP_Tokenization_2026_2027.ipynb" \
      "Personal Workspace/01 tokenization/"
   ```
2. **Work** on the notebook inside `Personal Workspace/<topic>/`. This keeps the
   original in `Assignments/` untouched, so you can always refer back to it
   or get a fresh copy if needed.
3. **Submit** by copying your finished notebook to `Submitted Work/`:
   ```bash
   cp "Personal Workspace/01 tokenization/01_ANLP_Tokenization_2026_2027.ipynb" \
      "Submitted Work/01 tokenization/"
   ```

> **Tip:** The `Personal Workspace/` folder is yours — `git pull` will never 
> overwrite files there. This means your in-progress work is safe when you 
> fetch new assignments.

## Local Development

See [LOCAL_SETUP_STUDENT.md](LOCAL_SETUP_STUDENT.md) for detailed instructions on running the
notebooks on your own machine. The course JupyterHub is recommended, but local
development is supported for students who prefer it.

## Getting Help

- **Technical issues:** Check the [Troubleshooting](LOCAL_SETUP_STUDENT.md#12-troubleshooting) section in LOCAL_SETUP_STUDENT.md
- **Assignment questions:** Post on the course discussion forum
- **Emergency support:** Contact the course TAs via email

## Grading & Submission

### How Assignments are Graded

- Each notebook contains **test cells** that validate your solutions automatically
- Your code must pass all test cells to receive full credit
- Some assignments include **manually graded** sections (marked with `### BEGIN SOLUTION` / `### END SOLUTION`)
  - These require written explanations or analysis
  - Graders will read your answers between the solution markers
- **Points distribution:** Check the point values in each exercise cell header

### Submission Process

1. **Complete the assignment** in your `Personal Workspace/` folder
2. **Test your work:** Run all cells from top to bottom (Kernel → Restart & Run All)
   - Ensure all test cells pass
   - Check that your explanations are complete
3. **Copy to Submitted Work:**
   ```bash
   cp "Personal Workspace/<topic>/<notebook>.ipynb" "Submitted Work/<topic>/"
   ```
4. **Keep the original filename** — do not rename the notebook
5. The grading system will automatically collect notebooks from `Submitted Work/`

### Important Submission Rules

- **Deadline:** Check the course syllabus for individual assignment deadlines
- **Late submissions:** Consult the course policy in the syllabus (typically -10% per day)
- **Resubmissions:** You may resubmit before the deadline; only the latest version counts
- **Do not modify:**
  - Test cells (cells with `assert` statements)
  - Cell metadata or nbgrader tags
  - The notebook filename
- **Academic integrity:** All work must be your own
  - Collaboration policy is detailed in the course syllabus
  - Code plagiarism will be detected and penalized

### Checking Your Submission

Before the deadline, verify your submission:

```bash
# List what you've submitted
ls -la "Submitted Work/<topic>/"

# Quick test: re-run the notebook
jupyter nbconvert --to notebook --execute "Submitted Work/<topic>/<notebook>.ipynb"
```

If the notebook executes without errors, your submission is ready.

## Prerequisites

This course assumes:
- Solid Python programming skills
- Basic understanding of linear algebra and calculus
- Familiarity with machine learning concepts (classification, regression, evaluation metrics)
- Completion of introductory NLP or data science courses (recommended)

## Resources

- **Textbooks:** See course syllabus for recommended readings
- **Documentation:** All major libraries (NLTK, spaCy, Transformers, PyTorch) have excellent online docs
- **Papers:** Key research papers are linked in the assignment notebooks
- **Office hours:** Check the course schedule

---

*For local setup instructions, see [LOCAL_SETUP_STUDENT.md](LOCAL_SETUP_STUDENT.md)*

*Last updated: July 2026*
