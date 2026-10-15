# Resume Match Assistant — LLM-supported HR tool

A beginner-friendly AI/ML project that compares multiple resumes with a job description, produces an explainable advisory ranking, and exports results to CSV. The UI is built with Streamlit. Resume text is extracted locally from PDF, DOCX, and TXT files; scoring uses explicit skill evidence plus local TF-IDF cosine similarity. An optional OpenAI language-model call writes a short evidence note from the score and skill evidence only.

> **Important:** This is a learning/demo project, not a validated hiring system. The score is not a probability of job performance. Do not automatically reject or select applicants based on it. A qualified person should review the original resumes, use consistent job-related criteria, and make all employment decisions.

## Features

- Upload multiple resumes at once (`.pdf`, `.docx`, `.txt`).
- Paste a job description and define required and preferred skills.
- Auto-detect a curated list of technical skills in the job description. If the required-skills box is empty, detected skills are treated as required.
- Show overall match score, required/preferred skill coverage, TF-IDF text similarity, matched skills, and skills not found in extracted text.
- Generate optional evidence notes with the OpenAI Responses API. The LLM receives only the role title, score components, and skill lists—not resume text, filenames, or contact details.
- Download a CSV of results.
- Test the app with the fictitious sample resumes in `sample_data/`.
- Run unit tests for skill extraction and ranking.

## Project structure

```text
resume_ranking_system/
├── app.py                         # Streamlit interface and workflow
├── ranker.py                      # Skill matching, TF-IDF similarity, scoring, ranking
├── resume_parser.py               # PDF/DOCX/TXT text extraction
├── llm_explainer.py               # Optional OpenAI-generated evidence note
├── requirements.txt               # Python dependencies
├── .env.example                   # Optional OpenAI configuration template
├── sample_data/
│   ├── sample_job_description.txt
│   ├── sample_resume_01_ml_engineer.txt
│   ├── sample_resume_02_genai.txt
│   └── sample_resume_03_data_analyst.txt
└── tests/
    └── test_ranker.py
```

## How the project works

1. **Input:** The user enters a role title and job description, supplies required/preferred skills, and uploads resumes.
2. **Text extraction:** `resume_parser.py` reads text locally with `pypdf` (PDF) or `python-docx` (DOCX). TXT files are decoded directly. Scanned/image-only PDFs are not supported because OCR is not included.
3. **Skill evidence:** `ranker.py` searches for skills from an editable skill dictionary. It records exactly which listed skills were found in the extracted text and which were not found.
4. **Text similarity:** A TF-IDF vectorizer and cosine similarity compare the job description with each resume. This is a local NLP baseline, not a generative model.
5. **Advisory score:** Components are combined using fixed, visible weights. These weights are easy to change in `ranker.py` and should be evaluated on representative, appropriately governed data before any real-world use.
6. **LLM note (optional):** If enabled and configured, the OpenAI API receives only structured match evidence and writes a short explanation. The LLM does not decide the ranking and never receives the resume body in this implementation.
7. **Review/export:** The UI shows candidate-level evidence, and the user can download a CSV for further human review.

### Score formula

All components are expressed as percentages from 0 to 100.

- Required and preferred skills supplied: `65% required-skill coverage + 15% preferred-skill coverage + 20% TF-IDF similarity`.
- Required skills only: `80% required-skill coverage + 20% TF-IDF similarity`.
- Preferred skills only: `60% preferred-skill coverage + 40% TF-IDF similarity`.
- No skill list: `100% TF-IDF similarity` (the UI warns that this is a weak basis for ranking).

Skill coverage is the count of listed skills found in extracted text divided by the number of listed skills. A missing skill means **not found in the extracted text**, not that the applicant definitely lacks that ability. TF-IDF scores are relative text similarity, not a qualification probability.

## Requirements

- Python 3.10 or newer recommended.
- Internet access to install packages. Internet access is **not** required to run local ranking after dependencies are installed.
- Windows, macOS, or Linux.
- Optional: an OpenAI API key if you want LLM evidence notes. Local scoring does not require an API key.

## Run it on Windows (PowerShell)

### 1. Extract the project

Download/extract `resume_ranking_system.zip` into a folder. Open that folder in VS Code (`File → Open Folder`).

### 2. Create and activate a virtual environment

In the VS Code terminal (PowerShell), run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can instead run the commands below with the environment's Python executable without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If activation succeeds, continue with step 3.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Start the app

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open that URL in your browser. Keep the terminal open while using the app; press `Ctrl+C` in the terminal to stop it.

## Run it on macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Try the included demo data

1. Open `sample_data/sample_job_description.txt` and copy its contents into the Job description field.
2. Upload all three `.txt` files from `sample_data/` in the Resume uploader.
3. Leave Required skills blank to let the app use skills detected in the job description, or enter your own required skill list.
4. Optionally add preferred skills such as `FastAPI, Docker, LangChain`.
5. Check the permission/decision-support acknowledgement.
6. Click **Analyze resumes**.
7. Inspect the result table and candidate-by-candidate evidence, then click **Download results as CSV**.

The sample resumes are synthetic and only intended to test functionality.

## Enable optional OpenAI evidence notes

The core app works without this. To turn on the LLM notes:

1. Copy `.env.example` to a new file named `.env` in the project root.
2. Add your own API key to `.env`:

   ```text
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-5-mini
   ```

3. Restart Streamlit and select **Generate short evidence notes with OpenAI**.

Use a valid key from your OpenAI API account and check your provider's current pricing and data-use terms before use. Do not commit `.env` to GitHub. `.gitignore` excludes it. The app sends only role title, match component percentages, and skill evidence to the model for an explanatory note; resume text and file names are not sent to that LLM function. The optional API call can still incur charges.

## Run tests

From the project root with the virtual environment activated:

```powershell
python -m pytest -q
```

macOS/Linux uses the same command after activating the environment.

## Customisation ideas

- Add job-specific skills and aliases to `SKILL_ALIASES` in `ranker.py`.
- Change score weights in `rank_resumes()` and record the reason for any change.
- Add OCR only after considering its accuracy, privacy, and failure modes.
- Add evaluation against a carefully reviewed test set; measure false matches and false omissions and test for disparate impact before any employment use.
- Add authentication, retention limits, secure storage, audit logs, and access controls before considering a multi-user deployment.

## Limitations and responsible-use notes

- The tool does not validate whether a resume claim is true.
- Text extraction may miss content in columns, tables, graphics, or scanned pages.
- The skill dictionary is not exhaustive; a skill may be expressed with a synonym not included in the dictionary.
- TF-IDF is a simple lexical method and does not understand meaning as deeply as modern embedding models.
- Resume formatting and writing style can affect similarity; scores should not be treated as objective measures of candidate quality.
- Do not include age, gender, race, religion, disability, family status, photographs, or other personal/protected attributes as screening criteria. The code is designed around job-related skill evidence, but this alone does not establish legal compliance or fairness.
- Keep access limited, obtain appropriate permission, define a retention policy, and follow applicable employment and privacy laws before processing real applicant data.
