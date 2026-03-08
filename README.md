# ChronoCare AI — Medical Claims Timeline Platform

Multi-agent AI system that processes medical PDFs into audit-ready clinical timelines.

## Quick Start (Local)
```bash
pip install -r requirements.txt
set GEMINI_API_KEY=your_key_here
uvicorn server:app --reload --port 8000
```
Open: http://localhost:8000

## Deploy to Render
1. Push to GitHub
2. Create new Web Service on Render
3. Set environment variable: GEMINI_API_KEY
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`

## Endpoints
- `/` — Landing page
- `/app` — Dashboard (drop PDFs here)
- `/api/upload` — Upload PDFs (POST)
- `/api/episodes` — List episodes (GET)
- `/api/episodes/{id}` — Episode detail (GET)
- `/api/episodes/{id}/csv` — CSV export (GET)
- `/api/episodes/{id}/pdf` — PDF report (GET)

## Architecture
```
User drops PDFs → server.py
  → Agent 1: Document Classification (Gemini)
  → Agent 2: Event & Cost Extraction (Gemini)
  → Agent 3: Timeline Builder (local)
  → Agent 4: QA & Anomaly Detection (local)
  → Agent 5: Narrative Generation (Gemini)
  → Dashboard displays results
```
