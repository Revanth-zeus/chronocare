"""
ChronoCare AI — Agent 2: Event & Cost Extractor
=================================================
Takes classified segments from Agent 1 → extracts:
  - Clinical events (diagnoses, procedures, tests, treatments)
  - ICD-10 codes (extracted or inferred)
  - CPT/HCPCS codes
  - Cost line items (charges, totals, ambulance fees)
  - Provider details (name, specialty, NPI, signature)
  - Medications (name, dose, route, frequency)
  - Ambulance-specific data (interventions, transport details)

Uses doc-type-aware prompting: different extraction strategies
for billing statements vs. operative notes vs. lab reports.

Self-contained — no imports from rest of project.
"""
import json
import time
import re
import os
import hashlib
from pathlib import Path
from enum import Enum
from typing import Optional, List, Dict, Any

import requests


# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ═══════════════════════════════════════════════════════
# EXTRACTION SYSTEM PROMPT (The Core IP)
# ═══════════════════════════════════════════════════════

EXTRACTION_SYSTEM = """You are a medical claims data extractor for insurance audit timelines.

### YOUR ROLE ###
You receive a classified medical document segment (with doc_type, date context, and raw text)
and extract ALL structured clinical and financial data needed for an insurance audit timeline.

### EXTRACTION RULES ###

**CLINICAL EVENTS** — Extract every distinct medical event:
1. Each diagnosis, procedure, test, medication order, consultation, admission, discharge, 
   transfer, intervention, or significant clinical finding = one event.
2. For each event, identify the MOST SPECIFIC date. If no date is explicit, use the 
   segment's primary_service_date.
3. Assign an event_type from: diagnosis | procedure | test | lab_result | medication | 
   imaging | consultation | admission | discharge | transfer | ambulance_intervention | 
   vital_signs | assessment | follow_up | referral | other
4. Extract ICD-10-CM codes if explicitly stated in the text (e.g., "S72.001A").
   If a diagnosis is described but no code given, set icd_codes to [] — do NOT guess codes.
5. Extract CPT/HCPCS codes if explicitly stated. If a procedure is described but no code 
   given, set cpt_codes to [].

**COST ITEMS** — Extract every financial charge:
1. Each line item on a billing statement = one cost_item.
2. Categories: ambulance | emergency | room | surgery | anesthesia | medication | 
   lab | imaging | therapy | supply | professional_fee | facility_fee | other
3. Extract the exact dollar amount. If a range or estimate, use the stated amount.
4. Link to service_date if specified on the line item.
5. For billing statements, extract EVERY line item — do not summarize or group.

**MEDICATIONS** — Extract medication details:
1. Drug name (generic preferred, brand acceptable)
2. Dose and unit (e.g., "500mg", "2L")
3. Route (PO, IV, IM, SC, topical, inhaled, etc.)
4. Frequency (PRN, BID, TID, QID, daily, once, continuous, etc.)
5. If medication is in a procedure context (e.g., anesthesia drugs), still extract it.

**PROVIDERS** — Extract provider information:
1. Full name, specialty/role, NPI if present
2. Facility name and address if present
3. Whether a signature or attestation was detected
4. Distinguish: treating_provider | ordering_provider | consulting_provider | 
   referring_provider | supervising_provider | signing_provider

**DEDUPLICATION** — Medical records are repetitive (WITHIN-SEGMENT ONLY):
1. If the same diagnosis or procedure is mentioned multiple times within a single segment
   (e.g., in HPI, Assessment, and Plan), extract it as ONE event.
2. Consolidate all relevant details into that single event's description.
3. Use the most specific date, code, and provider from any of the mentions.
4. The source_quote should reference the most definitive mention (e.g., Assessment over HPI).
5. NOTE: If the same condition appears in DIFFERENT segments (different files), extract it
   in both. Agent 3 (Timeline Builder) handles cross-segment deduplication.

**NEGATION HANDLING** — Do NOT extract negated findings as positive events:
1. "Negative for fracture", "No history of diabetes", "Denies chest pain" = NOT events.
2. Only extract ACTIVE, CONFIRMED, or SUSPECTED findings.
3. If something is "ruled out" (e.g., "CT negative for PE"), extract the TEST (imaging)
   but NOT the diagnosis. Note the negative finding in the test event's description.

**UNITS ENFORCEMENT** — Always include units for measurable values:
1. Lab results: always include unit (mg/dL, mmol/L, cells/uL, etc.)
2. Vitals: always include unit (mmHg for BP, bpm for HR, °F/°C for temp, % for SpO2)
3. Medications: always include dose unit (mg, mcg, mL, units, etc.)
4. Do NOT return bare numbers without units.

**CODE VALIDATION** — Correct misplaced codes:
1. ICD-10 codes follow the pattern: letter + 2 digits + optional dot + more digits (e.g., S72.001A, E11.65)
2. CPT codes are 5-digit numbers (e.g., 99285, 27236)
3. If text labels a CPT code as "ICD-10" or vice versa, place it in the CORRECT field
   based on its FORMAT, not its label in the text.
4. Revenue codes are typically 3-4 digits (e.g., 0450, 0120).

**DOC-TYPE SPECIFIC RULES**:

- ambulance_report: Extract ALL interventions (CPR, oxygen, IV access, medications given 
  in transit, immobilization, monitoring). Extract dispatch/scene/transport times. 
  Extract mileage and transport level (ALS1, ALS2, BLS, SCT).
  
- emergency_department_note: Extract triage level (ESI 1-5), chief complaint, all 
  procedures performed in ED, disposition, time-critical events (door-to-CT, door-to-OR).

- operative_note: Extract procedure name, surgeon, assistant, anesthesia type and 
  provider, estimated blood loss (EBL), specimens sent, implants used, complications.

- progress_note: Extract daily assessments, plan changes, new orders, discontinued 
  medications, consultation requests, patient status changes.

- lab_report: Extract each test name, result value, unit, reference range, and 
  abnormal flag (H/L/critical). Group by specimen collection date.

- billing_statement: Extract EVERY line item with date, description, code, and amount.
  Extract admission/discharge dates, total charges, adjustments, insurance payments.

- discharge_summary: Extract admission diagnosis, discharge diagnosis, procedures 
  performed during stay, discharge medications, follow-up instructions.

- radiology_report: Extract modality, body part, findings, impression, critical 
  findings if any.

- pathology_report: Extract specimen type, gross findings, microscopic findings, 
  final diagnosis with staging if applicable.

- consultation_note: Extract requesting provider, reason for consult, consultant's 
  assessment, recommendations.

### NULL vs EMPTY RULES ###
- ALL array fields MUST be arrays. If empty, return []. NEVER return null for arrays.
- String fields: use null if information is not present. Do NOT hallucinate values.
- Numeric fields: use null if not stated. Do NOT estimate costs.

### CONFIDENCE ###
For each event and cost_item, assign a confidence score (0.0-1.0):
- 1.0: Explicitly stated in text with exact values
- 0.9: Clearly implied from context
- 0.7-0.8: Requires inference (e.g., ICD code from description)
- 0.5-0.6: Ambiguous or partially readable
- Below 0.5: Flag as uncertain

### OUTPUT SCHEMA ###
Return ONLY valid JSON matching this exact structure:
{
  "events": [
    {
      "event_id": "evt_001",
      "event_type": "diagnosis|procedure|test|lab_result|medication|imaging|consultation|admission|discharge|transfer|ambulance_intervention|vital_signs|assessment|follow_up|referral|other",
      "date": "YYYY-MM-DD",
      "description": "Clear description of the clinical event",
      "icd_codes": ["S72.001A"],
      "cpt_codes": ["27236"],
      "body_site": "left femur|chest|abdomen|null if not applicable",
      "provider_name": "Dr. Elena Vasquez",
      "provider_role": "treating_provider|ordering_provider|consulting_provider|referring_provider|supervising_provider|signing_provider",
      "confidence": 0.95,
      "source_quote": "Quote from text supporting this extraction (max 120 chars, capture Action + Value)"
    }
  ],
  "cost_items": [
    {
      "cost_id": "cost_001",
      "date": "YYYY-MM-DD",
      "category": "ambulance|emergency|room|surgery|anesthesia|medication|lab|imaging|therapy|supply|professional_fee|facility_fee|other",
      "description": "Line item description",
      "amount": 1234.56,
      "code": "CPT/HCPCS/revenue code if present",
      "linked_event_id": "evt_001 or null — links this charge to the clinical event it corresponds to",
      "confidence": 0.98
    }
  ],
  "medications": [
    {
      "med_id": "med_001",
      "name": "Drug name",
      "dose": "500mg",
      "route": "IV|PO|IM|SC|topical|inhaled|other",
      "frequency": "BID|TID|QID|daily|PRN|once|continuous|other",
      "date_started": "YYYY-MM-DD or null",
      "date_stopped": "YYYY-MM-DD or null",
      "context": "Brief context: why prescribed"
    }
  ],
  "providers": [
    {
      "name": "Full Name",
      "role": "treating_provider|ordering_provider|consulting_provider|referring_provider|supervising_provider|signing_provider",
      "specialty": "Specialty if stated",
      "npi": "NPI number or null",
      "facility": "Facility name or null",
      "signature_detected": true
    }
  ],
  "vitals_series": [
    {
      "label": "triage|pre-hospital|intra-op|post-op|daily|discharge|other",
      "date": "YYYY-MM-DD",
      "time": "HH:MM or null",
      "bp": "120/80 mmHg or null",
      "hr": "88 bpm or null",
      "rr": "16 breaths/min or null",
      "temp": "98.6°F or null",
      "spo2": "98% or null",
      "gcs": "15 or null",
      "pain_scale": "0-10 integer or 'X/10' format, or null"
    }
  ],
  "ambulance_details": {
    "dispatch_time": "HH:MM or null",
    "scene_time": "HH:MM or null",
    "transport_time": "HH:MM or null",
    "arrival_time": "HH:MM or null",
    "mileage": "number or null",
    "transport_level": "ALS1|ALS2|BLS|SCT|null",
    "origin": "scene address or null",
    "destination": "hospital name or null"
  },
  "lab_results": [
    {
      "test_name": "Test Name",
      "value": "result value",
      "unit": "unit",
      "reference_range": "range",
      "flag": "normal|high|low|critical|null",
      "specimen_date": "YYYY-MM-DD"
    }
  ],
  "extraction_notes": "Any issues, ambiguities, or observations about this extraction"
}

### CRITICAL REMINDERS ###
- Extract what IS there. Do not infer what ISN'T.
- ICD/CPT codes: ONLY extract codes explicitly written in text. Never generate codes.
- Costs: ONLY extract amounts explicitly stated. Never estimate.
- If the segment is flagged as unknown or multi_type_fragment, extract what you can but 
  note the ambiguity in extraction_notes.
- Preserve source_quote for auditability — every event should trace back to text.
"""


# ═══════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════

class ClinicalEvent:
    def __init__(self, event_id, event_type, date, description,
                 icd_codes=None, cpt_codes=None, body_site=None,
                 provider_name=None, provider_role=None,
                 confidence=0.0, source_quote=None):
        self.event_id = event_id
        self.event_type = event_type
        self.date = date
        self.description = description
        self.icd_codes = icd_codes or []
        self.cpt_codes = cpt_codes or []
        self.body_site = body_site
        self.provider_name = provider_name
        self.provider_role = provider_role
        self.confidence = confidence
        self.source_quote = source_quote

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "date": self.date,
            "description": self.description,
            "icd_codes": self.icd_codes,
            "cpt_codes": self.cpt_codes,
            "body_site": self.body_site,
            "provider_name": self.provider_name,
            "provider_role": self.provider_role,
            "confidence": self.confidence,
            "source_quote": self.source_quote,
        }


class CostItem:
    def __init__(self, cost_id, date, category, description,
                 amount, code=None, linked_event_id=None, confidence=0.0):
        self.cost_id = cost_id
        self.date = date
        self.category = category
        self.description = description
        self.amount = amount
        self.code = code
        self.linked_event_id = linked_event_id
        self.confidence = confidence

    def to_dict(self):
        return {
            "cost_id": self.cost_id,
            "date": self.date,
            "category": self.category,
            "description": self.description,
            "amount": self.amount,
            "code": self.code,
            "linked_event_id": self.linked_event_id,
            "confidence": self.confidence,
        }


class Medication:
    def __init__(self, med_id, name, dose=None, route=None,
                 frequency=None, date_started=None, date_stopped=None,
                 context=None):
        self.med_id = med_id
        self.name = name
        self.dose = dose
        self.route = route
        self.frequency = frequency
        self.date_started = date_started
        self.date_stopped = date_stopped
        self.context = context

    def to_dict(self):
        return {
            "med_id": self.med_id,
            "name": self.name,
            "dose": self.dose,
            "route": self.route,
            "frequency": self.frequency,
            "date_started": self.date_started,
            "date_stopped": self.date_stopped,
            "context": self.context,
        }


class ExtractionResult:
    """Output for a single segment extraction."""
    def __init__(self, segment_id, source_filename, doc_type,
                 events=None, cost_items=None, medications=None,
                 providers=None, vitals_series=None,
                 ambulance_details=None, lab_results=None,
                 extraction_notes=None, processing_time_ms=0,
                 raw_response=None):
        self.segment_id = segment_id
        self.source_filename = source_filename
        self.doc_type = doc_type
        self.events = events or []
        self.cost_items = cost_items or []
        self.medications = medications or []
        self.providers = providers or []
        self.vitals_series = vitals_series or []
        self.ambulance_details = ambulance_details
        self.lab_results = lab_results or []
        self.extraction_notes = extraction_notes
        self.processing_time_ms = processing_time_ms
        self.raw_response = raw_response

    def to_dict(self):
        return {
            "segment_id": self.segment_id,
            "source_filename": self.source_filename,
            "doc_type": self.doc_type,
            "events": [e.to_dict() if hasattr(e, 'to_dict') else e for e in self.events],
            "cost_items": [c.to_dict() if hasattr(c, 'to_dict') else c for c in self.cost_items],
            "medications": [m.to_dict() if hasattr(m, 'to_dict') else m for m in self.medications],
            "providers": self.providers,
            "vitals_series": self.vitals_series,
            "ambulance_details": self.ambulance_details,
            "lab_results": self.lab_results,
            "extraction_notes": self.extraction_notes,
            "processing_time_ms": self.processing_time_ms,
            "stats": self.stats(),
        }

    def stats(self):
        return {
            "event_count": len(self.events),
            "cost_count": len(self.cost_items),
            "medication_count": len(self.medications),
            "provider_count": len(self.providers),
            "lab_count": len(self.lab_results),
            "total_cost": sum(
                c.amount if hasattr(c, 'amount') else c.get('amount', 0)
                for c in self.cost_items
                if (c.amount if hasattr(c, 'amount') else c.get('amount')) is not None
            ),
        }


# ═══════════════════════════════════════════════════════
# GEMINI API CALL
# ═══════════════════════════════════════════════════════

def call_gemini_extraction(segment_context: str, raw_text: str,
                           api_key: str = None, model: str = None) -> dict:
    """Send segment to Gemini for structured extraction."""
    key = api_key or GEMINI_API_KEY
    mdl = model or GEMINI_MODEL
    if not key:
        raise ValueError("GEMINI_API_KEY not set")

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}"
           f":generateContent?key={key}")

    user_message = f"""### CLASSIFICATION_CONTEXT ###
{segment_context}

### AUDIT_REFERENCE_DATE ###
{time.strftime("%Y-%m-%d")}

### OCR_TEXT ###
{raw_text}

Extract all clinical events, cost items, medications, providers, vitals, 
ambulance details, and lab results. Return ONLY valid JSON."""

    payload = {
        "contents": [{"parts": [{"text": user_message}]}],
        "systemInstruction": {"parts": [{"text": EXTRACTION_SYSTEM}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
            "topP": 0.95,
        },
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


# ═══════════════════════════════════════════════════════
# SEGMENT CONTEXT BUILDER
# ═══════════════════════════════════════════════════════

def build_segment_context(segment: dict) -> str:
    """Build the context string from an Agent 1 segment."""
    ctx = {
        "segment_id": segment.get("segment_id"),
        "source_filename": segment.get("source_filename"),
        "doc_type": segment.get("doc_type"),
        "page_range": f"{segment.get('page_range_start')}-{segment.get('page_range_end')}",
        "primary_service_date": segment.get("primary_service_date"),
        "patient_name": segment.get("patient_name"),
        "patient_dob": segment.get("patient_dob"),
        "detected_facility": segment.get("detected_facility"),
        "confidence": segment.get("confidence"),
        "detected_dates": segment.get("detected_dates", []),
        "providers_from_classification": segment.get("providers", []),
    }
    return json.dumps(ctx, indent=2)


# ═══════════════════════════════════════════════════════
# EXTRACT FROM SINGLE SEGMENT
# ═══════════════════════════════════════════════════════

def extract_segment(segment: dict, raw_text: str,
                    api_key: str = None) -> ExtractionResult:
    """Extract all structured data from a single Agent 1 segment."""
    t0 = time.time()
    seg_id = segment.get("segment_id", "unknown")
    fname = segment.get("source_filename", "unknown")
    doc_type = segment.get("doc_type", "unknown")

    # Skip segments that are too fragmentary
    if segment.get("confidence", 0) < 0.15:
        return ExtractionResult(
            segment_id=seg_id,
            source_filename=fname,
            doc_type=doc_type,
            extraction_notes="Skipped: confidence too low for extraction",
            processing_time_ms=int((time.time() - t0) * 1000),
        )

    # Build context and call Gemini
    context = build_segment_context(segment)
    raw = call_gemini_extraction(context, raw_text, api_key=api_key)

    # Parse events
    events = []
    for e in raw.get("events", []):
        events.append(ClinicalEvent(
            event_id=e.get("event_id", f"evt_{len(events)+1:03d}"),
            event_type=e.get("event_type", "other"),
            date=e.get("date"),
            description=e.get("description", ""),
            icd_codes=e.get("icd_codes", []),
            cpt_codes=e.get("cpt_codes", []),
            body_site=e.get("body_site"),
            provider_name=e.get("provider_name"),
            provider_role=e.get("provider_role"),
            confidence=e.get("confidence", 0.5),
            source_quote=e.get("source_quote"),
        ))

    # Parse cost items
    cost_items = []
    for c in raw.get("cost_items", []):
        amt = c.get("amount")
        if amt is not None:
            try:
                amt = float(amt)
            except (ValueError, TypeError):
                amt = None
        cost_items.append(CostItem(
            cost_id=c.get("cost_id", f"cost_{len(cost_items)+1:03d}"),
            date=c.get("date"),
            category=c.get("category", "other"),
            description=c.get("description", ""),
            amount=amt,
            code=c.get("code"),
            linked_event_id=c.get("linked_event_id"),
            confidence=c.get("confidence", 0.5),
        ))

    # Parse medications
    medications = []
    for m in raw.get("medications", []):
        medications.append(Medication(
            med_id=m.get("med_id", f"med_{len(medications)+1:03d}"),
            name=m.get("name", "Unknown"),
            dose=m.get("dose"),
            route=m.get("route"),
            frequency=m.get("frequency"),
            date_started=m.get("date_started"),
            date_stopped=m.get("date_stopped"),
            context=m.get("context"),
        ))

    elapsed = int((time.time() - t0) * 1000)

    return ExtractionResult(
        segment_id=seg_id,
        source_filename=fname,
        doc_type=doc_type,
        events=events,
        cost_items=cost_items,
        medications=medications,
        providers=raw.get("providers", []),
        vitals_series=raw.get("vitals_series", []),
        ambulance_details=raw.get("ambulance_details"),
        lab_results=raw.get("lab_results", []),
        extraction_notes=raw.get("extraction_notes"),
        processing_time_ms=elapsed,
        raw_response=raw,
    )


# ═══════════════════════════════════════════════════════
# BATCH EXTRACTION (process all Agent 1 results)
# ═══════════════════════════════════════════════════════

def extract_all(agent1_results: list, page_texts: dict = None,
                api_key: str = None) -> List[ExtractionResult]:
    """
    Process all Agent 1 IngestionResults.
    
    agent1_results: list of IngestionResult.to_dict() outputs
    page_texts: dict mapping (filename, page_num) -> text
                If None, uses text_preview from segments
    """
    all_extractions = []

    for result in agent1_results:
        fname = result["source_filename"]
        for seg in result["segments"]:
            # Get the full raw text for this segment
            if page_texts:
                # Reconstruct from page texts
                pages = range(seg["page_range_start"],
                              seg["page_range_end"] + 1)
                raw = "\n\n".join(
                    page_texts.get((fname, p), "")
                    for p in pages
                )
            else:
                raw = seg.get("raw_text", seg.get("text_preview", ""))

            extraction = extract_segment(seg, raw, api_key=api_key)
            all_extractions.append(extraction)

    return all_extractions


# ═══════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import fitz  # PyMuPDF

    # ── 1. Locate Agent 1 results ──
    results_path = None
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
    else:
        for candidate in [
            Path("agent1_v2_results.json"),
            Path("chronocare/agent1_v2_results.json"),
        ]:
            if candidate.exists():
                results_path = candidate
                break

    if not results_path or not results_path.exists():
        print("Usage: python agent2_extractor.py [agent1_results.json]")
        sys.exit(1)

    with open(results_path) as f:
        agent1_data = json.load(f)

    # ── 2. API key (check hardcoded first, then env) ──
    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("❌ Set GEMINI_API_KEY environment variable or hardcode in file")
        sys.exit(1)

    # ── 3. Locate test_data PDFs ──
    test_data_dir = None
    for candidate in [
        Path("test_data"),
        Path("chronocare/test_data"),
        results_path.parent / "test_data",
    ]:
        if candidate.exists() and any(candidate.glob("*.pdf")):
            test_data_dir = candidate
            break

    # ── 4. Extract ALL page texts from PDFs ──
    # Build multiple lookup keys so we always find the text
    page_texts = {}  # (filename, page_num) -> text
    pdf_files = {}   # filename -> Path

    if test_data_dir:
        print(f"\n📂 Reading PDFs from: {test_data_dir}")
        for pdf_file in sorted(test_data_dir.glob("*.pdf")):
            pdf_files[pdf_file.name] = pdf_file
            doc = fitz.open(str(pdf_file))
            for i, page in enumerate(doc):
                page_texts[(pdf_file.name, i + 1)] = page.get_text()
            print(f"  {pdf_file.name}: {len(doc)} pages")
            doc.close()
        print(f"  TOTAL: {len(page_texts)} pages loaded ✅")
    else:
        print("⚠️  test_data/ not found — will use text_preview from Agent 1 (limited)")

    # ── 5. Parse Agent 1 output format ──
    # Agent 1 saves results in various formats. Handle all of them.
    # Format A: { "files": [ {source_filename, segments}, ... ] }
    # Format B: [ {source_filename, segments}, ... ]
    # Format C: { source_filename, segments }  (single file)
    # Format D: { "01_ambulance.pdf": {segments}, "02_ed.pdf": {segments} }

    agent1_list = []

    if isinstance(agent1_data, list):
        agent1_list = agent1_data
    elif isinstance(agent1_data, dict):
        if "files" in agent1_data:
            agent1_list = agent1_data["files"]
        elif "segments" in agent1_data:
            agent1_list = [agent1_data]
        else:
            # Format D: keys are filenames, values have segments
            for key, val in agent1_data.items():
                if isinstance(val, dict) and "segments" in val:
                    val["source_filename"] = val.get("source_filename", key)
                    agent1_list.append(val)

    # If still empty, just wrap the whole thing
    if not agent1_list:
        agent1_list = [agent1_data]

    # ── 6. Helper: find the right PDF filename for a segment ──
    def get_segment_text(seg, result_filename):
        """Get full text for a segment by re-reading PDFs."""
        # Try multiple filename sources
        seg_fname = seg.get("source_filename") or result_filename or ""

        # Try exact match first
        if page_texts:
            pages = range(
                seg.get("page_range_start", 1),
                seg.get("page_range_end", 1) + 1,
            )

            # Try exact filename
            text_parts = [page_texts.get((seg_fname, p), "") for p in pages]
            if any(text_parts):
                return "\n\n".join(text_parts)

            # Try matching by partial name (e.g., "01_ambulance" matches "01_ambulance_run_sheet.pdf")
            for pdf_name in pdf_files:
                if seg_fname in pdf_name or pdf_name in seg_fname:
                    text_parts = [page_texts.get((pdf_name, p), "") for p in pages]
                    if any(text_parts):
                        return "\n\n".join(text_parts)

            # Try matching by doc_type keyword
            doc_type = seg.get("doc_type", "")
            type_keywords = {
                "ambulance_report": "ambulance",
                "emergency_department_note": "emergency",
                "operative_note": "operative",
                "progress_note": "progress",
                "lab_report": "lab",
                "billing_statement": "billing",
                "discharge_summary": "discharge",
                "radiology_report": "radiology",
                "pathology_report": "pathology",
                "consultation_note": "consultation",
            }
            keyword = type_keywords.get(doc_type, "")
            if keyword:
                for pdf_name in pdf_files:
                    if keyword in pdf_name.lower():
                        text_parts = [page_texts.get((pdf_name, p), "") for p in pages]
                        if any(text_parts):
                            return "\n\n".join(text_parts)

        # Fallback: use whatever text is in the segment
        return seg.get("raw_text", seg.get("text_preview", ""))

    # ── 7. Run extraction ──
    print("\n" + "=" * 70)
    print("  ChronoCare AI — Agent 2: Event & Cost Extractor")
    print(f"  {len(agent1_list)} file(s) | {len(page_texts)} pages loaded")
    print("=" * 70)

    all_extractions = []
    total_events = 0
    total_costs = 0
    total_meds = 0
    total_cost_amount = 0.0

    for result in agent1_list:
        fname = result.get("source_filename", "unknown")
        segments = result.get("segments", [])
        print(f"\n📋 {fname} ({len(segments)} segments)")

        for seg in segments:
            doc_type = seg.get("doc_type", "unknown")
            conf = seg.get("confidence", 0)
            ps = seg.get("page_range_start", "?")
            pe = seg.get("page_range_end", "?")

            # Get full text for this segment
            raw = get_segment_text(seg, fname)
            text_len = len(raw.strip())

            print(f"\n  🔍 [{doc_type}] pp.{ps}-{pe} (conf:{conf}) [{text_len} chars]")

            if text_len < 20:
                print(f"     ⚠️  No text available — skipping")
                all_extractions.append({
                    "segment_id": seg.get("segment_id"),
                    "doc_type": doc_type,
                    "error": "No text available for extraction",
                })
                continue

            try:
                extraction = extract_segment(seg, raw, api_key=api_key)
                stats = extraction.stats()

                print(f"     ⏱️  {extraction.processing_time_ms}ms")
                print(f"     📊 {stats['event_count']} events | "
                      f"{stats['cost_count']} costs | "
                      f"{stats['medication_count']} meds | "
                      f"{stats['lab_count']} labs | "
                      f"${stats['total_cost']:,.2f}")

                # Show events
                for e in extraction.events[:5]:
                    e_dict = e.to_dict()
                    codes = e_dict['icd_codes'] + e_dict['cpt_codes']
                    code_str = f" [{', '.join(codes)}]" if codes else ""
                    print(f"     🏥 [{e_dict['event_type']}] {e_dict['date']}: "
                          f"{e_dict['description'][:60]}{code_str}")
                if len(extraction.events) > 5:
                    print(f"     ... +{len(extraction.events)-5} more events")

                # Show top costs
                for c in extraction.cost_items[:3]:
                    c_dict = c.to_dict()
                    if c_dict.get('amount') is not None:
                        print(f"     💰 {c_dict['date']}: ${c_dict['amount']:,.2f} "
                              f"({c_dict['category']}) {c_dict['description'][:40]}")
                if len(extraction.cost_items) > 3:
                    print(f"     ... +{len(extraction.cost_items)-3} more costs")

                # Show medications
                for m in extraction.medications[:3]:
                    m_dict = m.to_dict()
                    print(f"     💊 {m_dict['name']} {m_dict.get('dose','')} "
                          f"{m_dict.get('route','')} {m_dict.get('frequency','')}")
                if len(extraction.medications) > 3:
                    print(f"     ... +{len(extraction.medications)-3} more meds")

                # Show extraction notes
                if extraction.extraction_notes:
                    print(f"     📝 {extraction.extraction_notes[:120]}")

                total_events += stats['event_count']
                total_costs += stats['cost_count']
                total_meds += stats['medication_count']
                total_cost_amount += stats['total_cost']

                all_extractions.append(extraction.to_dict())

            except Exception as e:
                print(f"     ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                all_extractions.append({
                    "segment_id": seg.get("segment_id"),
                    "error": str(e),
                })

    # ── 8. Summary ──
    print("\n" + "=" * 70)
    print("  EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"  Segments processed: {len(all_extractions)}")
    print(f"  Total events:       {total_events}")
    print(f"  Total cost items:   {total_costs}")
    print(f"  Total medications:  {total_meds}")
    print(f"  Total charges:      ${total_cost_amount:,.2f}")

    # Save results
    output_path = results_path.parent / "agent2_extraction_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "agent": "agent2_extractor",
            "version": "v2",
            "input_file": str(results_path),
            "summary": {
                "segments_processed": len(all_extractions),
                "total_events": total_events,
                "total_cost_items": total_costs,
                "total_medications": total_meds,
                "total_charges": total_cost_amount,
            },
            "extractions": all_extractions,
        }, f, indent=2, default=str)

    print(f"\n  Saved: {output_path}")
    print("  ✅ Done!")
