# JobCred

Project for job posting credibility analysis, verification, and RAG-assisted evaluation.

## Project Structure

```text
JobCred/
├── data/
│   ├── raw/             # Unprocessed source datasets
│   └── processed/       # Cleaned and feature-engineered datasets
├── notebooks/           # Exploratory data analysis and experimentation notebooks
├── src/
│   ├── preprocessing/   # Data cleaning and text preprocessing pipelines
│   ├── features/        # Feature extraction and engineering
│   ├── models/          # Model training, evaluation, and inference
│   ├── verification/    # Job/company verification checks and heuristics
│   └── rag/             # Retrieval-Augmented Generation components
├── app/                 # Web application / API interface
├── reports/             # Generated reports, metrics, and figures
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```
