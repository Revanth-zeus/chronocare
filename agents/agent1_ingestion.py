"""
ChronoCare AI — Agent 1: Document Ingestion & Segmenter (v2)
=============================================================
Rebuilt with Gemini's own feedback:
  - Structured date context (date + type, not just strings)
  - Primary service date for timeline pinning
  - Reflection/reasoning field for transparency
  - Noise handling (blank pages, non-medical docs, OCR garbage)
  - Confidence-action logic (low confidence -> flag for review)
  - Strict JSON schema lockdown
  - Conflict resolution directives
  - Document delimiter format

Self-contained — no imports from rest of project.
"""
import json
import time
import re
import os
import hashlib
from pathlib import Path
from enum import Enum
from typing import Optional

import fitz  # PyMuPDF
import requests


# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_PAGES_PER_CHUNK = 3
LOW_CONFIDENCE_THRESHOLD = 0.70


# ═══════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════

class DocType(str, Enum):
    AMBULANCE_REPORT = "ambulance_report"
    EMERGENCY_DEPARTMENT_NOTE = "emergency_department_note"
    OPERATIVE_NOTE = "operative_note"
    PROGRESS_NOTE = "progress_note"
    LAB_REPORT = "lab_report"
    BILLING_STATEMENT = "billing_statement"
    DISCHARGE_SUMMARY = "discharge_summary"
    RADIOLOGY_REPORT = "radiology_report"
    CONSULTATION_NOTE = "consultation_note"
    PATHOLOGY_REPORT = "pathology_report"
    MULTI_TYPE_FRAGMENT = "multi_type_fragment"
    UNKNOWN = "unknown"


class DateContext:
    def __init__(self, date: str, context: str):
        self.date = date
        self.context = context
    def to_dict(self):
        return {"date": self.date, "context": self.context}


class DocumentSegment:
    def __init__(self, segment_id, source_filename, doc_type, page_range_start,
                 page_range_end, raw_text, primary_service_date=None,
                 detected_dates=None, detected_facility=None,
                 section_types=None, confidence=0.0,
                 requires_human_review=False, reflection=None,
                 patient_name=None, patient_dob=None, providers=None):
        self.segment_id = segment_id
        self.source_filename = source_filename
        self.doc_type = doc_type
        self.page_range_start = page_range_start
        self.page_range_end = page_range_end
        self.raw_text = raw_text
        self.primary_service_date = primary_service_date
        self.detected_dates = detected_dates or []
        self.detected_facility = detected_facility
        self.section_types = section_types or []
        self.confidence = confidence
        self.requires_human_review = requires_human_review
        self.reflection = reflection
        self.patient_name = patient_name
        self.patient_dob = patient_dob
        self.providers = providers or []

    def to_dict(self):
        return {
            "segment_id": self.segment_id,
            "source_filename": self.source_filename,
            "doc_type": self.doc_type,
            "page_range_start": self.page_range_start,
            "page_range_end": self.page_range_end,
            "primary_service_date": self.primary_service_date,
            "detected_dates": [d.to_dict() if isinstance(d, DateContext) else d for d in self.detected_dates],
            "detected_facility": self.detected_facility,
            "section_types": self.section_types,
            "confidence": self.confidence,
            "requires_human_review": self.requires_human_review,
            "reflection": self.reflection,
            "patient_name": self.patient_name,
            "patient_dob": self.patient_dob,
            "providers": self.providers,
            "text_preview": (self.raw_text[:300] + "...") if len(self.raw_text) > 300 else self.raw_text,
        }


class IngestionResult:
    def __init__(self, source_filename, total_pages, segments,
                 patient_name=None, patient_dob=None, facility=None,
                 processing_time_ms=0):
        self.source_filename = source_filename
        self.total_pages = total_pages
        self.segments = segments
        self.patient_name = patient_name
        self.patient_dob = patient_dob
        self.facility = facility
        self.processing_time_ms = processing_time_ms

    def to_dict(self):
        return {
            "source_filename": self.source_filename,
            "total_pages": self.total_pages,
            "segments": [s.to_dict() for s in self.segments],
            "patient_name": self.patient_name,
            "patient_dob": self.patient_dob,
            "facility": self.facility,
            "processing_time_ms": self.processing_time_ms,
        }


# ═══════════════════════════════════════════════════════
# PDF TEXT EXTRACTION
# ═══════════════════════════════════════════════════════

class PageInfo:
    def __init__(self, page_number: int, text: str):
        self.page_number = page_number
        self.text = text
        self.char_count = len(text)


def extract_pages(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        pages.append(PageInfo(page_number=i + 1, text=text))
    doc.close()
    return pages


# ═══════════════════════════════════════════════════════
# GEMINI API CALLER
# ═══════════════════════════════════════════════════════

def call_gemini(prompt: str, system_instruction: str = "") -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192},
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    # Try JSON mode first
    body["generationConfig"]["responseMimeType"] = "application/json"
    resp = requests.post(url, json=body, timeout=90)

    if resp.status_code != 200:
        print(f"    [info] JSON mode returned {resp.status_code}, retrying plain text...")
        del body["generationConfig"]["responseMimeType"]
        resp = requests.post(url, json=body, timeout=90)

    resp.raise_for_status()
    data = resp.json()

    if "candidates" not in data or not data["candidates"]:
        feedback = data.get("promptFeedback", {})
        raise RuntimeError(f"No candidates. Block: {feedback.get('blockReason','?')}. Resp: {json.dumps(data)[:500]}")

    candidate = data["candidates"][0]
    if candidate.get("finishReason") == "SAFETY":
        raise RuntimeError(f"Safety blocked: {candidate}")

    try:
        text = candidate["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Bad structure: {json.dumps(data)[:500]}") from e

    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ═══════════════════════════════════════════════════════
# CLASSIFICATION SYSTEM PROMPT (v2 — Gemini-reviewed)
# ═══════════════════════════════════════════════════════

CLASSIFICATION_SYSTEM = """You are a high-precision medical document classification engine for ChronoCare AI, an insurance claims auditing system that builds date-indexed timelines from clinical PDFs.

Your job: Given OCR-extracted text from clinical PDF pages, produce a structured JSON classification. Accuracy of document type and dates is CRITICAL — a wrong date or misclassified document breaks the entire audit timeline.

### JSON SCHEMA DEFINITION ###
Your output MUST strictly follow this schema. Do not add extra keys. Do not omit keys. Every field must match its defined type.

{
  "reflection": "string — Brief reasoning for your classification. If uncertain, explain ambiguity.",
  "doc_type": "EXACTLY one of: ambulance_report | emergency_department_note | operative_note | progress_note | lab_report | billing_statement | discharge_summary | radiology_report | consultation_note | pathology_report | multi_type_fragment | unknown",
  "confidence": 0.0 to 1.0,
  "primary_service_date": "YYYY-MM-DD or null — The date this clinical event ACTUALLY OCCURRED (not print/report/signature date)",
  "patient_info": {"name": "string or null", "dob": "YYYY-MM-DD or null"},
  "facility_name": "string or null",
  "detected_dates": [{"date": "YYYY-MM-DD", "type": "service_date|admission_date|discharge_date|surgery_date|specimen_date|report_date|signature_date|amendment_date|dob|historical_date|billing_date|dispatch_date|arrival_date|other"}],
  "section_types": ["strings from allowed list below"],
  "providers": [{"name": "string", "role": "string or null", "npi": "string or null"}],
  "requires_human_review": true/false,
  "summary": "One sentence describing clinical content"
}

### CLASSIFICATION RULES ###
- ambulance_report: EMS run sheets, pre-hospital care, PCR, transport records. Look for: unit numbers, dispatch/scene times, EMT signatures, transport billing.
- emergency_department_note: ER physician notes, ED evaluations, trauma assessments. Look for: triage level, chief complaint, ED course, disposition.
- operative_note: Surgery reports, procedure notes, anesthesia records. Look for: pre-op/post-op diagnosis, procedure performed, surgeon, EBL, findings.
- progress_note: Daily physician notes, SOAP notes, rounding notes, ICU notes. Look for: POD#, S/O/A/P format, daily orders.
- lab_report: Laboratory results, blood work panels. Look for: specimen date, reference ranges, CLIA numbers.
- billing_statement: Itemized charges, UB-04/CMS-1500. Look for: CPT/HCPCS codes, dollar amounts, billing cycles.
- discharge_summary: Final discharge notes/instructions. Look for: discharge date, condition, follow-up.
- radiology_report: CT/MRI/X-ray interpretations. Look for: modality, findings, impression.
- consultation_note: Specialist consultations. Look for: consulting service, reason, recommendations.
- pathology_report: Surgical pathology, tissue examination. Look for: specimen, gross/micro findings, path diagnosis.
- multi_type_fragment: Use when a SINGLE PAGE contains two or more distinct document types (e.g., lab results AND pathology report on the same page). Set requires_human_review = true. List ALL detected document types in the reflection field (e.g., "Page contains both lab_report and pathology_report content").
- unknown: ONLY when text is truly unclassifiable, non-medical, or too fragmentary to determine type (e.g., signature-only pages, blank pages).

### PRIMARY SERVICE DATE RULES ###
The primary_service_date is the date the clinical event ACTUALLY HAPPENED:
- Ambulance reports: dispatch/scene date
- ER notes: date of ER visit
- Operative notes: date of surgery
- Progress notes: the day being documented
- Lab reports: specimen collection date
- Billing: earliest date of service
- Discharge summaries: discharge date

### DATE EXTRACTION RULES ###
1. Extract ALL clinically relevant dates. Tag each with its context type.
2. Format: ALWAYS YYYY-MM-DD. Examples: "01/13/2025" -> "2025-01-13", "January 13, 2025" -> "2025-01-13"
3. US date format assumed (MM/DD/YYYY) since these are US medical records.
4. IGNORE: footer copyright dates, system print timestamps, watermarks.
5. Past Medical History dates: tag as "historical_date".
6. FUTURE DATE CHECK: Any date after the current audit date (provided in the prompt) is likely an OCR misread. Flag it in reflection and attempt correction (e.g., "2029" is probably "2025"). If correction is uncertain, keep the original but add requires_human_review = true.
7. AMENDMENT/LATE ENTRY DETECTION: If the text contains "Amended", "Addendum", "Late Entry", "Cosigned", or similar language indicating the note was written or modified after the service date, note this in reflection and ensure primary_service_date reflects the ORIGINAL event date, not the amendment date. Tag the amendment/signature date separately as "signature_date" or "amendment_date" in detected_dates.

### NOISE HANDLING ###
- Text < 10 meaningful words: doc_type="unknown", confidence=0.10, explain in reflection.
- Mostly OCR garbage: confidence < 0.20, note in reflection.
- Non-medical document: doc_type="unknown", explain in reflection.
- Mid-page fragment: If identifying info (patient name, facility) is missing because this appears to be a mid-page of a multi-page document, set those fields to null and note "Mid-page fragment — identifying info likely on preceding page" in reflection.

### CONFLICT RESOLUTION ###
- Conflicting demographics: use most frequently occurring values, note conflict in reflection.
- Conflicting dates: note in reflection, prioritize document body over headers/footers.

### TYPE ENFORCEMENT ###
- ALL array fields (detected_dates, section_types, providers) MUST be arrays. If empty, return []. NEVER return null for any array field.
- detected_dates: MUST be array (never null). Empty = [].
- section_types: MUST be array (never null). Empty = [].
- providers: MUST be array (never null). Empty = [].
- patient_info: MUST be object with "name" and "dob" keys. Use null for unknown.
- If confidence < 0.70, set requires_human_review = true.

### ALLOWED SECTION TYPES ###
patient_demographics, history_of_present_illness, past_medical_history, review_of_systems, physical_examination, vital_signs, diagnostics, imaging, assessment_plan, medications, procedures, operative_findings, lab_results, billing_items, signatures, scene_assessment, prehospital_interventions, transport_info, disposition, discharge_instructions, follow_up"""


# ═══════════════════════════════════════════════════════
# CLASSIFY PAGES
# ═══════════════════════════════════════════════════════

def classify_pages(pages: list, source_filename: str) -> list:
    results = []
    for chunk_start in range(0, len(pages), MAX_PAGES_PER_CHUNK):
        chunk = pages[chunk_start: chunk_start + MAX_PAGES_PER_CHUNK]

        # Use Gemini-recommended document delimiters
        page_blocks = []
        for p in chunk:
            page_blocks.append(
                f"### PAGE {p.page_number} START ###\n"
                f"{p.text[:5000]}\n"
                f"### PAGE {p.page_number} END ###"
            )

        prompt = (
            f"### AUDIT_CONTEXT ###\n"
            f'{{"current_date": "{time.strftime("%Y-%m-%d")}", "source_file": "{source_filename}"}}\n\n'
            f"### TASK ###\n"
            f"Classify these {len(chunk)} page(s). Page range: {chunk[0].page_number} to {chunk[-1].page_number}.\n"
            f"This is part of a clinical patient packet for insurance claims auditing.\n\n"
            + "\n\n".join(page_blocks)
        )

        try:
            raw = call_gemini(prompt, CLASSIFICATION_SYSTEM)
            classification = json.loads(raw)

            # Schema enforcement — fill defaults for any missing keys
            classification.setdefault("reflection", "")
            classification.setdefault("doc_type", "unknown")
            classification.setdefault("confidence", 0.0)
            classification.setdefault("primary_service_date", None)
            classification.setdefault("patient_info", {"name": None, "dob": None})
            classification.setdefault("facility_name", None)
            classification.setdefault("detected_dates", [])
            classification.setdefault("section_types", [])
            classification.setdefault("providers", [])
            classification.setdefault("requires_human_review", False)
            classification.setdefault("summary", "")

            # Force arrays (never null)
            for key in ("detected_dates", "section_types", "providers"):
                if classification[key] is None:
                    classification[key] = []

            # Confidence-action logic
            if classification["confidence"] < LOW_CONFIDENCE_THRESHOLD:
                classification["requires_human_review"] = True

            classification["page_range_start"] = chunk[0].page_number
            classification["page_range_end"] = chunk[-1].page_number
            classification["raw_text"] = "\n".join(p.text for p in chunk)
            results.append(classification)

        except json.JSONDecodeError as e:
            print(f"    ❌ JSON parse error pages {chunk[0].page_number}-{chunk[-1].page_number}: {e}")
            if 'raw' in dir():
                print(f"       Raw: {raw[:200]}...")
            results.append(_fallback(chunk, f"JSON parse: {e}"))
        except Exception as e:
            print(f"    ❌ Gemini error pages {chunk[0].page_number}-{chunk[-1].page_number}: {type(e).__name__}: {e}")
            results.append(_fallback(chunk, str(e)))

    return results


def _fallback(chunk, error_msg):
    return {
        "doc_type": "unknown", "confidence": 0.0, "primary_service_date": None,
        "detected_dates": [], "facility_name": None,
        "patient_info": {"name": None, "dob": None},
        "section_types": [], "providers": [],
        "requires_human_review": True,
        "reflection": f"Classification failed: {error_msg}",
        "summary": f"Error: {error_msg}",
        "page_range_start": chunk[0].page_number,
        "page_range_end": chunk[-1].page_number,
        "raw_text": "\n".join(p.text for p in chunk),
    }


# ═══════════════════════════════════════════════════════
# MERGE ADJACENT SAME-TYPE SEGMENTS
# ═══════════════════════════════════════════════════════

def merge_adjacent_segments(classifications: list) -> list:
    if not classifications:
        return []
    merged = [dict(classifications[0])]
    for curr in classifications[1:]:
        prev = merged[-1]
        if (curr["doc_type"] == prev["doc_type"]
                and curr["page_range_start"] == prev["page_range_end"] + 1):
            prev["page_range_end"] = curr["page_range_end"]
            prev["raw_text"] += "\n" + curr["raw_text"]
            # Merge dates (dedupe by date string)
            seen = {d["date"] if isinstance(d, dict) else d for d in prev["detected_dates"]}
            for d in curr["detected_dates"]:
                dv = d["date"] if isinstance(d, dict) else d
                if dv not in seen:
                    prev["detected_dates"].append(d)
                    seen.add(dv)
            prev["section_types"] = list(set(prev["section_types"] + curr.get("section_types", [])))
            # Merge providers
            seen_p = {p.get("name","") for p in prev.get("providers",[])}
            for p in curr.get("providers",[]):
                if p.get("name","") not in seen_p:
                    prev.setdefault("providers",[]).append(p)
            prev["confidence"] = min(prev["confidence"], curr["confidence"])
            if not prev.get("primary_service_date") and curr.get("primary_service_date"):
                prev["primary_service_date"] = curr["primary_service_date"]
            if curr.get("reflection"):
                prev["reflection"] = (prev.get("reflection","") + " | " + curr["reflection"]).strip(" | ")
        else:
            merged.append(dict(curr))
    return merged


# ═══════════════════════════════════════════════════════
# DATE NORMALIZATION
# ═══════════════════════════════════════════════════════

DATE_PATTERNS = [
    (r'(\d{1,2})/(\d{1,2})/(\d{4})',
     lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
    (r'(\d{4})-(\d{2})-(\d{2})', lambda m: m.group(0)),
]

def normalize_date_str(d):
    if not d: return d
    for pat, fmt in DATE_PATTERNS:
        m = re.search(pat, str(d))
        if m:
            try: return fmt(m)
            except: pass
    return str(d)

def normalize_dates_in_segment(cls):
    if cls.get("primary_service_date"):
        cls["primary_service_date"] = normalize_date_str(cls["primary_service_date"])
    normalized = []
    for d in cls.get("detected_dates", []):
        if isinstance(d, dict):
            d["date"] = normalize_date_str(d.get("date",""))
            normalized.append(d)
        elif isinstance(d, str):
            normalized.append({"date": normalize_date_str(d), "type": "other"})
    cls["detected_dates"] = normalized
    pi = cls.get("patient_info") or {}
    if pi.get("dob"):
        pi["dob"] = normalize_date_str(pi["dob"])
    return cls


# ═══════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════

def run_agent1(pdf_path: str) -> IngestionResult:
    start_time = time.time()
    pdf_path = str(pdf_path)
    filename = Path(pdf_path).name

    pages = extract_pages(pdf_path)
    if not pages:
        return IngestionResult(source_filename=filename, total_pages=0, segments=[])

    classifications = classify_pages(pages, filename)
    merged = merge_adjacent_segments(classifications)

    segments = []
    patient_name = None
    patient_dob = None
    facility = None

    for cls in merged:
        cls = normalize_dates_in_segment(cls)
        pi = cls.get("patient_info") or {}
        if pi.get("name") and not patient_name:
            patient_name = pi["name"]
        if pi.get("dob") and not patient_dob:
            patient_dob = pi["dob"]
        if cls.get("facility_name") and not facility:
            facility = cls["facility_name"]

        seg_id = hashlib.md5(
            f"{filename}-{cls['page_range_start']}-{cls['page_range_end']}".encode()
        ).hexdigest()[:12]

        try:
            DocType(cls["doc_type"])
            doc_type = cls["doc_type"]
        except (ValueError, KeyError):
            doc_type = "unknown"

        date_contexts = []
        for d in cls.get("detected_dates", []):
            if isinstance(d, dict):
                date_contexts.append(DateContext(d.get("date",""), d.get("type", d.get("context","other"))))
            elif isinstance(d, str):
                date_contexts.append(DateContext(d, "other"))

        segments.append(DocumentSegment(
            segment_id=seg_id, source_filename=filename, doc_type=doc_type,
            page_range_start=cls["page_range_start"], page_range_end=cls["page_range_end"],
            raw_text=cls["raw_text"], primary_service_date=cls.get("primary_service_date"),
            detected_dates=date_contexts, detected_facility=cls.get("facility_name"),
            section_types=cls.get("section_types",[]), confidence=cls.get("confidence",0.0),
            requires_human_review=cls.get("requires_human_review",False),
            reflection=cls.get("reflection"), patient_name=pi.get("name"),
            patient_dob=pi.get("dob"), providers=cls.get("providers",[]),
        ))

    elapsed = int((time.time() - start_time) * 1000)
    return IngestionResult(
        source_filename=filename, total_pages=len(pages), segments=segments,
        patient_name=patient_name, patient_dob=patient_dob,
        facility=facility, processing_time_ms=elapsed,
    )


def run_agent1_batch(pdf_paths: list) -> list:
    return [run_agent1(p) for p in pdf_paths]


# ═══════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    test_dir = Path(__file__).parent.parent / "test_data"
    pdfs = sorted(test_dir.glob("*.pdf"))
    pdfs = [p for p in pdfs if "COMPLETE" not in p.name]

    print(f"\n{'='*70}")
    print(f"  ChronoCare AI — Agent 1 v2 Test")
    print(f"  {len(pdfs)} PDFs in {test_dir}")
    print(f"{'='*70}")

    # Text extraction
    print("\n📄 TEXT EXTRACTION:")
    tp, tc = 0, 0
    for pdf in pdfs:
        pages = extract_pages(str(pdf))
        c = sum(p.char_count for p in pages)
        tp += len(pages); tc += c
        print(f"  {pdf.name}: {len(pages)} pages, {c:,} chars")
    print(f"  TOTAL: {tp} pages, {tc:,} chars  ✅")

    if not GEMINI_API_KEY:
        print(f"\n⚠️  Set GEMINI_API_KEY to test classification")
        print(f'  Windows:  set GEMINI_API_KEY=your_key')
        sys.exit(0)

    # API check
    print(f"\n🔑 API CHECK: model={GEMINI_MODEL}, key={GEMINI_API_KEY[:8]}...{GEMINI_API_KEY[-4:]}")
    try:
        r = call_gemini('Return exactly: {"status":"ok"}')
        json.loads(r)
        print(f"  ✅ API + JSON mode working")
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")
        sys.exit(1)

    # Full pipeline
    print(f"\n🤖 CLASSIFICATION:")
    all_segs = []
    all_dates = set()
    reviews = 0

    for pdf in pdfs:
        print(f"\n  📋 {pdf.name}")
        result = run_agent1(str(pdf))
        print(f"     {len(result.segments)} segments | {result.processing_time_ms}ms")
        if result.patient_name: print(f"     Patient: {result.patient_name}")
        if result.facility: print(f"     Facility: {result.facility}")

        for seg in result.segments:
            flag = " ⚠️REVIEW" if seg.requires_human_review else ""
            print(f"     [{seg.doc_type}] pp.{seg.page_range_start}-{seg.page_range_end} conf:{seg.confidence:.2f}{flag}")
            if seg.primary_service_date:
                print(f"       📌 Primary: {seg.primary_service_date}")
            if seg.detected_dates:
                ds = [f"{d.date}({d.context})" for d in seg.detected_dates]
                print(f"       📅 {', '.join(ds)}")
            if seg.reflection:
                print(f"       💭 {seg.reflection[:120]}")
            if seg.providers:
                ps = [f"{p.get('name','')}({p.get('role','')})" for p in seg.providers]
                print(f"       👨‍⚕️ {', '.join(ps)}")
            if seg.requires_human_review: reviews += 1
            all_segs.append(seg)
            for d in seg.detected_dates: all_dates.add(d.date)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    dtc = {}
    for s in all_segs: dtc[s.doc_type] = dtc.get(s.doc_type, 0) + 1
    print(f"  Segments: {len(all_segs)}")
    for dt, c in sorted(dtc.items()): print(f"    {dt}: {c}")
    print(f"  Timeline dates: {sorted(all_dates)}")
    print(f"  Need review: {reviews}/{len(all_segs)}")
    avg = sum(s.confidence for s in all_segs) / len(all_segs) if all_segs else 0
    print(f"  Avg confidence: {avg:.2f}")

    # Save results
    out = Path(__file__).parent.parent / "agent1_v2_results.json"
    data = {"model": GEMINI_MODEL, "segments": [s.to_dict() for s in all_segs], "dates": sorted(all_dates)}
    with open(out, "w") as f: json.dump(data, f, indent=2)
    print(f"  Saved: {out}")
    print(f"  ✅ Done!")
