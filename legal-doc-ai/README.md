# Legal Document AI (From Scratch)

Local full-stack legal document analysis system (React + FastAPI) for Apple Silicon.

## Core Capabilities
- Upload and preview legal PDF judgments.
- Extract entities with source spans.
- Extract important legal points (FACT/ISSUE/ARGUMENT/REASONING/DECISION).
- Generate easy-language summaries.
- Suggest next legal steps.
- Retrieval-augmented context using TF-IDF corpus index.

## Constraint Followed
This implementation is configured to train and run without market pretrained NLP checkpoints. Tokenizer and core models are trained from your dataset artifacts.

## Project Paths
- Backend: `/Users/vishalkumar/Desktop/NLP-main/legal-doc-ai/backend`
- Frontend: `/Users/vishalkumar/Desktop/NLP-main/legal-doc-ai/frontend`
- Commands: `/Users/vishalkumar/Desktop/NLP-main/legal-doc-ai/terminal.txt`
- Detailed notes: `/Users/vishalkumar/Desktop/NLP-main/legal-doc-ai/explanation.txt`

## Run Order
Use `terminal.txt` step-by-step:
1. Download dataset
2. Preprocess
3. Step1..Step10 scripts
4. Start backend + frontend

## Key Outputs
- Splits: `backend/data/processed/splits/*.jsonl`
- Checkpoints: `backend/models/checkpoints/scratch/*`
- Retrieval index: `backend/data/processed/tfidf_*`
- Visualizations: `backend/reports/*.png`
- Final report: `backend/reports/pipeline_report.md`

## API
- `GET /api/v1/health`
- `POST /api/v1/analyze/upload`
- `GET /api/v1/analyze/jobs/{job_id}`
- `POST /api/v1/analyze/stream-summary`

## Optional Groq Summary Upgrade
- Set `LEGAL_DOC_GROQ_API_KEY` in `backend/.env` to enable Groq-powered summary generation.
- The API key stays on the backend and is never exposed to the frontend.
- If the key is missing or the API call fails, the app falls back to the local summarizer.

## Disclaimer
Outputs are assistive only and are not legal advice.
