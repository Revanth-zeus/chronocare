"""
ChronoCare AI - Full-Stack Backend v2
======================================
Drop PDFs -> Agents 1-5 run automatically -> Results in dashboard.

Usage:
  pip install fastapi uvicorn python-multipart PyMuPDF requests
  set GEMINI_API_KEY=your_key
  uvicorn server:app --reload --port 8000

Open: http://localhost:8000/app
"""
import os
import sys
import uuid
import json
import shutil
import time
import traceback
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

# ======================================================
# CONFIG
# ======================================================
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Add project root to path so agents/ can be imported
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ======================================================
# APP
# ======================================================
app = FastAPI(title="ChronoCare AI", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# In-memory episode store
episodes_db = {}


# ======================================================
# PIPELINE - runs in background thread
# ======================================================

def run_full_pipeline(episode_id: str, pdf_paths: list, api_key: str):
    """Run all 5 agents sequentially. Updates episodes_db in real-time."""
    ep = episodes_db[episode_id]
    ep["status"] = "processing"
    pipeline_start = time.time()

    try:
        # ------------------------------------------
        # AGENT 1: Document Ingestion & Segmentation
        # ------------------------------------------
        ep["current_step"] = "agent1"
        print(f"[{episode_id}] Agent 1: Classifying {len(pdf_paths)} PDFs...")

        from agents.agent1_ingestion import run_agent1

        agent1_results = []
        for pdf_path in pdf_paths:
            result = run_agent1(pdf_path)
            agent1_results.append(result)
            print(f"  -> {result.source_filename}: {len(result.segments)} segments")

        ep["agent1_complete"] = True
        ep["total_segments"] = sum(len(r.segments) for r in agent1_results)

        # Convert to dicts for Agent 2
        a1_dicts = [r.to_dict() for r in agent1_results]

        # Extract patient name from Agent 1 (first one found)
        patient_name_from_a1 = None
        patient_dob_from_a1 = None
        for r in agent1_results:
            if r.patient_name:
                patient_name_from_a1 = r.patient_name
            if r.patient_dob:
                patient_dob_from_a1 = r.patient_dob
            if patient_name_from_a1:
                break

        ep["patient_name_a1"] = patient_name_from_a1

        # Extract page texts for Agent 2 (it needs raw text per page)
        page_texts = {}
        all_pages = {}
        for pdf_path in pdf_paths:
            doc = fitz.open(str(pdf_path))
            fname = Path(pdf_path).name
            all_pages[fname] = {}
            for i, page in enumerate(doc):
                text = page.get_text()
                page_texts[(fname, i + 1)] = text
                all_pages[fname][i + 1] = text
            doc.close()

        # ------------------------------------------
        # AGENT 2: Event & Cost Extraction
        # ------------------------------------------
        ep["current_step"] = "agent2"
        print(f"[{episode_id}] Agent 2: Extracting events...")

        from agents.agent2_extractor import extract_segment

        all_extractions = []
        total_events = 0
        total_costs = 0
        total_meds = 0
        total_charges = 0.0

        for a1_result in a1_dicts:
            fname = a1_result.get("source_filename", "unknown")
            for seg in a1_result.get("segments", []):
                doc_type = seg.get("doc_type", "unknown")
                conf = seg.get("confidence", 0)

                # Skip low-confidence segments
                if conf < 0.15:
                    print(f"  -> [{doc_type}] conf:{conf:.2f} - SKIPPED")
                    continue

                # Get raw text for this segment's pages
                pg_start = seg.get("page_range_start", 1)
                pg_end = seg.get("page_range_end", 1)
                pages_range = range(pg_start, pg_end + 1)
                raw = "\n\n".join(page_texts.get((fname, p), "") for p in pages_range)

                # Fallback: try matching by doc_type keyword in other filenames
                if len(raw.strip()) < 20:
                    type_keywords = {
                        "ambulance_report": "ambulance",
                        "emergency_department_note": "emergency",
                        "operative_note": "operative",
                        "progress_note": "progress",
                        "lab_report": "lab",
                        "billing_statement": "billing",
                    }
                    keyword = type_keywords.get(doc_type, "")
                    for pdf_name in all_pages:
                        if keyword and keyword in pdf_name.lower():
                            raw = "\n\n".join(
                                page_texts.get((pdf_name, p), "") for p in pages_range
                            )
                            if len(raw.strip()) > 20:
                                break

                if len(raw.strip()) < 20:
                    print(f"  -> [{doc_type}] no text - SKIPPED")
                    continue

                try:
                    extraction = extract_segment(seg, raw, api_key=api_key)
                    stats = extraction.stats()
                    total_events += stats["event_count"]
                    total_costs += stats["cost_count"]
                    total_meds += stats["medication_count"]
                    total_charges += stats["total_cost"]
                    ext_dict = extraction.to_dict()
                    # Inject patient info from Agent 1 so Agent 3 can find it
                    if patient_name_from_a1 and not ext_dict.get("patient_name"):
                        ext_dict["patient_name"] = patient_name_from_a1
                    if patient_dob_from_a1 and not ext_dict.get("patient_dob"):
                        ext_dict["patient_dob"] = patient_dob_from_a1
                    all_extractions.append(ext_dict)
                    print(f"  -> [{doc_type}] {stats['event_count']} events, "
                          f"{stats['cost_count']} costs, ${stats['total_cost']:,.2f}")
                except Exception as e2:
                    print(f"  -> [{doc_type}] EXTRACTION ERROR: {e2}")
                    continue

        a2_output = {
            "agent": "agent2_extractor",
            "version": "v2",
            "summary": {
                "segments_processed": len(all_extractions),
                "total_events": total_events,
                "total_cost_items": total_costs,
                "total_medications": total_meds,
                "total_charges": total_charges,
            },
            "extractions": all_extractions,
        }

        ep["agent2_complete"] = True
        ep["total_events_raw"] = total_events
        print(f"[{episode_id}] Agent 2 done: {total_events} events, ${total_charges:,.2f}")

        # ------------------------------------------
        # AGENT 3: Timeline Builder (no API calls)
        # ------------------------------------------
        ep["current_step"] = "agent3"
        print(f"[{episode_id}] Agent 3: Building timeline...")

        from agents.agent3_timeline import build_timeline

        timeline_obj = build_timeline(a2_output)
        timeline_dict = timeline_obj.to_dict()

        ep["agent3_complete"] = True
        print(f"[{episode_id}] Agent 3 done: {timeline_dict.get('episode',{}).get('total_events',0)} events")

        # ------------------------------------------
        # AGENT 4: QA & Anomaly Detection (no API)
        # ------------------------------------------
        ep["current_step"] = "agent4"
        print(f"[{episode_id}] Agent 4: Running QA audit...")

        from agents.agent4_qa import run_audit

        qa_report = run_audit(timeline_dict)
        qa_dict = qa_report.to_dict()

        ep["agent4_complete"] = True
        print(f"[{episode_id}] Agent 4 done: {qa_dict.get('total_findings',0)} findings, "
              f"risk {qa_dict.get('risk_score',0)}/100")

        # ------------------------------------------
        # AGENT 5: Narrative Generator (API calls)
        # ------------------------------------------
        ep["current_step"] = "agent5"
        print(f"[{episode_id}] Agent 5: Generating narratives...")

        from agents.agent5_narrative import generate_full_report

        narrative = generate_full_report(timeline_dict, qa_dict, api_key=api_key)

        ep["agent5_complete"] = True
        print(f"[{episode_id}] Agent 5 done: {len(narrative.get('sections',{}))} sections")

        # ------------------------------------------
        # STORE EVERYTHING
        # ------------------------------------------
        episode_data = timeline_dict.get("episode", {})
        patient_data = timeline_dict.get("patient", {})

        ep.update({
            "status": "ready",
            "current_step": "complete",
            "patient_name": patient_data.get("name") or "Unknown Patient",
            "patient_dob": patient_data.get("dob"),
            "episode_start": episode_data.get("start_date"),
            "episode_end": episode_data.get("end_date"),
            "duration_days": episode_data.get("duration_days"),
            "total_cost": episode_data.get("total_cost", 0),
            "total_events": episode_data.get("total_events", 0),
            "total_dates": episode_data.get("total_dates", 0),
            "risk_score": qa_dict.get("risk_score", 0),
            "total_findings": qa_dict.get("total_findings", 0),
            "processing_time": round(time.time() - pipeline_start, 1),
            # Full data
            "timeline": timeline_dict,
            "qa_report": qa_dict,
            "narrative": narrative,
        })

        print(f"[{episode_id}] PIPELINE COMPLETE in {ep['processing_time']}s")
        print(f"  Patient: {ep['patient_name']}")
        print(f"  Episode: {ep['episode_start']} -> {ep['episode_end']}")
        print(f"  Cost: ${ep['total_cost']:,.2f}")
        print(f"  Risk: {ep['risk_score']}/100")

    except Exception as e:
        tb = traceback.format_exc()
        ep["status"] = "error"
        ep["error"] = str(e)
        ep["error_trace"] = tb
        ep["processing_time"] = round(time.time() - pipeline_start, 1)
        print(f"[{episode_id}] PIPELINE ERROR: {e}")
        print(tb)


# ======================================================
# API ENDPOINTS
# ======================================================

@app.get("/api")
async def api_root():
    return {
        "service": "ChronoCare AI",
        "version": "2.0.0",
        "gemini_configured": bool(GEMINI_API_KEY),
        "episodes": len(episodes_db),
    }


@app.post("/api/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...)
):
    """Upload PDF(s) -> starts full 5-agent pipeline in background."""

    # Check API key
    if not GEMINI_API_KEY:
        raise HTTPException(
            400,
            "GEMINI_API_KEY not set. Run: set GEMINI_API_KEY=your_key"
        )

    # Filter to PDFs only
    pdf_files = [f for f in files if f.filename.lower().endswith(".pdf")]

    if not pdf_files:
        raise HTTPException(
            400,
            f"No PDF files found. Only .pdf files accepted. "
            f"You uploaded: {', '.join(f.filename for f in files)}"
        )

    # Create episode directory
    episode_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(uuid.uuid4())[:6]
    episode_dir = UPLOAD_DIR / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Save PDFs to disk
    saved_paths = []
    filenames = []
    for f in pdf_files:
        # Skip combined packet if individual files exist
        if "COMPLETE" in f.filename.upper() and len(pdf_files) > 1:
            continue
        dest = episode_dir / f.filename
        content = await f.read()
        with open(dest, "wb") as out:
            out.write(content)
        saved_paths.append(str(dest))
        filenames.append(f.filename)

    if not saved_paths:
        raise HTTPException(400, "No valid PDF files to process.")

    # Create episode record
    episodes_db[episode_id] = {
        "id": episode_id,
        "status": "processing",
        "current_step": "uploading",
        "created_at": datetime.now().isoformat(),
        "pdf_count": len(saved_paths),
        "filenames": filenames,
        "patient_name": None,
        "patient_dob": None,
        "episode_start": None,
        "episode_end": None,
        "duration_days": None,
        "total_cost": 0,
        "total_events": 0,
        "total_dates": 0,
        "risk_score": 0,
        "total_findings": 0,
        "processing_time": None,
        "error": None,
        "agent1_complete": False,
        "agent2_complete": False,
        "agent3_complete": False,
        "agent4_complete": False,
        "agent5_complete": False,
    }

    # Start pipeline in background
    background_tasks.add_task(run_full_pipeline, episode_id, saved_paths, GEMINI_API_KEY)

    print(f"[{episode_id}] Upload: {len(saved_paths)} PDFs -> {filenames}")

    return {
        "episode_id": episode_id,
        "status": "processing",
        "files": len(saved_paths),
        "filenames": filenames,
    }


@app.get("/api/episodes")
async def list_episodes():
    """List all episodes, newest first (YouTube history style)."""
    result = []
    for eid, ep in sorted(
        episodes_db.items(),
        key=lambda x: x[1].get("created_at", ""),
        reverse=True
    ):
        # Summary only - exclude huge data blobs
        summary = {k: v for k, v in ep.items()
                   if k not in ("timeline", "qa_report", "narrative", "error_trace")}
        result.append(summary)
    return result


@app.get("/api/episodes/{eid}")
async def get_episode(eid: str):
    """Get full episode data including timeline, QA, narrative."""
    if eid not in episodes_db:
        raise HTTPException(404, "Episode not found")
    ep = episodes_db[eid]

    # If still processing, return status only
    if ep["status"] != "ready":
        return {
            "id": ep["id"],
            "status": ep["status"],
            "current_step": ep.get("current_step"),
            "error": ep.get("error"),
            "error_trace": ep.get("error_trace"),
        }

    return ep


@app.get("/api/episodes/{eid}/csv")
async def download_csv(eid: str):
    """Download CSV export of a completed episode."""
    if eid not in episodes_db:
        raise HTTPException(404, "Episode not found")
    ep = episodes_db[eid]
    if ep["status"] != "ready":
        raise HTTPException(400, f"Episode not ready. Status: {ep['status']}")

    tl = ep.get("timeline", {})
    qa = ep.get("qa_report", {})
    patient = tl.get("patient", {})
    episode = tl.get("episode", {})

    rows = []
    rows.append("=== CHRONOCARE AI AUDIT EXPORT ===")
    rows.append(f"Patient,{patient.get('name','N/A')},DOB,{patient.get('dob','N/A')}")
    rows.append(f"Episode,{episode.get('start_date','')} to {episode.get('end_date','')}")
    rows.append(f"Total Charges,${episode.get('total_cost',0):,.2f},Risk,{qa.get('risk_score',0)}/100")
    rows.append("")

    rows.append("=== ITEMIZED CHARGES ===")
    rows.append("Date,Category,Description,Amount,Code,Linked Event")
    for date_str, entry in sorted((tl.get("timeline") or {}).items()):
        for cost in (entry.get("cost_items") or []):
            desc = str(cost.get("description", "")).replace(",", ";")
            rows.append(
                f"{date_str},{cost.get('category','')},{desc},"
                f"{cost.get('amount',0)},{cost.get('code','')},{cost.get('linked_event_id','')}"
            )

    rows.append("")
    rows.append("=== CLINICAL EVENTS ===")
    rows.append("Date,Type,Description,ICD-10,CPT,Provider")
    for date_str, entry in sorted((tl.get("timeline") or {}).items()):
        for evt in (entry.get("events") or []):
            desc = str(evt.get("description", "")).replace(",", ";")[:80]
            icds = ";".join(evt.get("icd_codes") or [])
            cpts = ";".join(evt.get("cpt_codes") or [])
            provs = ";".join(evt.get("providers") or [])
            rows.append(f"{date_str},{evt.get('event_type','')},{desc},{icds},{cpts},{provs}")

    rows.append("")
    rows.append("=== QA FINDINGS ===")
    rows.append("Severity,Category,Title,Date,Impact,Description")
    for f in (qa.get("all_findings") or []):
        desc = str(f.get("description", "")).replace(",", ";")[:100]
        amt = f"${f['financial_impact']:,.2f}" if f.get("financial_impact") else ""
        rows.append(
            f"{f.get('severity','')},{f.get('category','')},{f.get('title','')},"
            f"{f.get('affected_date','')},{amt},{desc}"
        )

    rows.append("")
    rows.append("=== MEDICATIONS ===")
    rows.append("Name,Dose,Route,Frequency,Sources")
    for mkey, med in (tl.get("medications") or {}).items():
        rows.append(
            f"{med.get('name','')},{med.get('dose','')},{med.get('route','')},"
            f"{med.get('frequency','')},{len(med.get('source_segments',[]))}"
        )

    csv_text = "\n".join(rows)
    safe_name = str(patient.get("name", "audit")).replace(" ", "_")
    filename = f"ChronoCare_{safe_name}_{episode.get('start_date','report')}.csv"

    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/api/episodes/{eid}")
async def delete_episode(eid: str):
    """Delete an episode and its uploaded files."""
    if eid not in episodes_db:
        raise HTTPException(404, "Episode not found")
    ep_dir = UPLOAD_DIR / eid
    if ep_dir.exists():
        shutil.rmtree(ep_dir)
    del episodes_db[eid]
    return {"deleted": eid}


@app.get("/api/episodes/{eid}/pdf")
async def download_pdf_report(eid: str):
    """Generate and download PDF audit report — 1 page per day."""
    if eid not in episodes_db:
        raise HTTPException(404, "Episode not found")
    ep = episodes_db[eid]
    if ep["status"] != "ready":
        raise HTTPException(400, "Episode not ready")

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, black, grey
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
    from reportlab.lib.enums import TA_CENTER
    from io import BytesIO

    tl = ep.get("timeline", {})
    qa = ep.get("qa_report", {})
    narr = ep.get("narrative", {})
    patient = tl.get("patient", {})
    episode_info = tl.get("episode", {})
    timeline = tl.get("timeline", {})
    sections = narr.get("sections", {})

    styles = getSampleStyleSheet()
    title_s = ParagraphStyle('T', parent=styles['Heading1'], fontSize=16, fontName='Helvetica-Bold', spaceAfter=4, alignment=TA_CENTER)
    head_s = ParagraphStyle('H', parent=styles['Heading2'], fontSize=12, fontName='Helvetica-Bold', spaceAfter=4, spaceBefore=10)
    body_s = ParagraphStyle('B', parent=styles['Normal'], fontSize=9, leading=12, spaceAfter=3)
    small_s = ParagraphStyle('S', parent=styles['Normal'], fontSize=8, leading=10, textColor=grey)
    narr_s = ParagraphStyle('N', parent=styles['Normal'], fontSize=9, leading=13, spaceAfter=4, leftIndent=12, borderColor=HexColor('#7c3aed'), borderWidth=1, borderPadding=6)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.6*inch, rightMargin=0.6*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []

    # Cover page
    story.append(Spacer(1, 120))
    story.append(Paragraph("CHRONOCARE AI", ParagraphStyle('Logo', parent=title_s, fontSize=28, spaceAfter=8)))
    story.append(Paragraph("Medical Claims Audit Report", ParagraphStyle('Sub', parent=title_s, fontSize=14, textColor=grey)))
    story.append(Spacer(1, 40))
    story.append(Paragraph(f"Patient: {patient.get('name', 'N/A')}", head_s))
    story.append(Paragraph(f"DOB: {patient.get('dob', 'N/A')}", body_s))
    story.append(Paragraph(f"Episode: {episode_info.get('start_date', '')} to {episode_info.get('end_date', '')}", body_s))
    story.append(Paragraph(f"Duration: {(episode_info.get('duration_days', 0))+1} days", body_s))
    story.append(Paragraph(f"Total Charges: ${episode_info.get('total_cost', 0):,.2f}", body_s))
    story.append(Paragraph(f"Risk Score: {qa.get('risk_score', 0)}/100", body_s))
    story.append(Paragraph(f"Total Events: {episode_info.get('total_events', 0)}", body_s))
    story.append(Paragraph(f"QA Findings: {qa.get('total_findings', 0)}", body_s))
    story.append(Spacer(1, 30))

    # Executive Summary
    exec_narr = sections.get("I_executive_summary") or sections.get("executive_summary")
    if exec_narr:
        story.append(Paragraph("EXECUTIVE SUMMARY", head_s))
        story.append(Paragraph(str(exec_narr), body_s))

    # QA Assessment
    qa_narr = sections.get("II_audit_findings") or sections.get("qa_narrative")
    if qa_narr:
        story.append(Paragraph(f"AUDIT ASSESSMENT — Risk {qa.get('risk_score', 0)}/100", head_s))
        story.append(Paragraph(str(qa_narr), body_s))

    story.append(PageBreak())

    # One page per day
    date_narratives = sections.get("IV_clinical_timeline") or sections.get("date_narratives") or {}
    sorted_dates = sorted(timeline.keys())

    for idx, date_str in enumerate(sorted_dates):
        entry = timeline[date_str]
        events = entry.get("events", [])
        costs = entry.get("cost_items", [])

        story.append(Paragraph(f"DAY {idx+1} — {date_str}", title_s))
        story.append(HRFlowable(width="100%", thickness=1, color=black, spaceAfter=8))
        story.append(Paragraph(f"Events: {len(events)} | Charges: {len(costs)} | Total: ${entry.get('total_cost', 0):,.2f}", small_s))
        story.append(Spacer(1, 6))

        # Clinical narrative for this date
        dn = date_narratives.get(date_str)
        if dn:
            story.append(Paragraph("Clinical Narrative:", head_s))
            story.append(Paragraph(str(dn), body_s))
            story.append(Spacer(1, 6))

        # Events
        if events:
            story.append(Paragraph("Clinical Events:", head_s))
            for evt in events[:20]:
                desc = evt.get("description", "")[:100]
                etype = evt.get("event_type", "")
                icds = ", ".join(evt.get("icd_codes", []))
                cpts = ", ".join(evt.get("cpt_codes", []))
                provs = ", ".join(evt.get("providers", []))
                codes = f" [{icds}]" if icds else ""
                codes += f" [CPT: {cpts}]" if cpts else ""
                prov_str = f" — {provs}" if provs else ""
                story.append(Paragraph(f"• <b>{etype.upper()}</b>{codes}: {desc}{prov_str}", body_s))

        # Costs
        if costs:
            story.append(Spacer(1, 6))
            story.append(Paragraph("Charges:", head_s))
            cost_data = [["Description", "Category", "Code", "Amount"]]
            for co in costs[:15]:
                cost_data.append([
                    str(co.get("description", ""))[:40],
                    str(co.get("category", "")),
                    str(co.get("code", "")),
                    f"${co.get('amount', 0):,.2f}" if co.get("amount") else ""
                ])
            t = Table(cost_data, colWidths=[220, 80, 60, 70], hAlign='LEFT')
            t.setStyle(TableStyle([
                ('FONTSIZE', (0, 0), (-1, -1), 7.5),
                ('GRID', (0, 0), (-1, -1), 0.3, grey),
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN', (-1, 0), (-1, -1), 'RIGHT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ]))
            story.append(t)

        # QA flags for this date
        date_findings = [f for f in (qa.get("all_findings") or []) if not f.get("affected_date") or f.get("affected_date") == date_str]
        if date_findings:
            story.append(Spacer(1, 6))
            story.append(Paragraph("QA Flags:", head_s))
            for f in date_findings[:5]:
                story.append(Paragraph(f"⚑ [{f.get('severity', '').upper()}] {f.get('title', '')} — {f.get('description', '')[:80]}", body_s))

        # Page break between days
        if idx < len(sorted_dates) - 1:
            story.append(PageBreak())

    doc.build(story)
    buf.seek(0)

    safe_name = str(patient.get("name", "audit")).replace(" ", "_")
    filename = f"ChronoCare_{safe_name}_{episode_info.get('start_date', 'report')}.pdf"

    return StreamingResponse(
        buf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ======================================================
# SERVE FRONTEND
# ======================================================

@app.get("/app")
async def serve_app():
    """Serve the dashboard app."""
    frontend_path = BASE_DIR / "frontend.html"
    if frontend_path.exists():
        return HTMLResponse(frontend_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>frontend.html not found</h1>")


@app.get("/")
async def serve_landing():
    """Serve the landing page."""
    landing_path = BASE_DIR / "landing.html"
    if landing_path.exists():
        return HTMLResponse(landing_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>landing.html not found</h1>")


# Serve static files (video, images)
from fastapi.staticfiles import StaticFiles
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
