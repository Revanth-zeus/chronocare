"""
ChronoCare AI — Agent 5: Narrative Generator
===============================================
Takes the unified timeline (Agent 3) + QA report (Agent 4) →
generates human-readable audit narratives using Gemini:
  - Episode summary (1-paragraph executive overview)
  - Date-by-date clinical narrative (what happened, why, outcome)
  - Cost justification narrative (links charges to clinical necessity)
  - QA findings narrative (plain-English audit concerns)
  - Medication reconciliation narrative
  - Provider attestation summary

This is the "final deliverable" — what a claims reviewer reads.

Uses Gemini 2.0 Flash for narrative generation.
Self-contained — no imports from rest of project.
"""
import json
import time
import re
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests


# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ═══════════════════════════════════════════════════════
# NARRATIVE SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════

EPISODE_SUMMARY_PROMPT = """You are a medical claims audit narrator. Generate a concise executive summary
of this episode of care for an insurance claims review.

### RULES ###
- Write in third person, past tense, strictly evidence-based medical-legal tone
- Current Date is {current_date}. Calculate the patient's age from their DOB.
- BEGIN the paragraph with the mechanism of injury and patient demographics
  (e.g., "This 46-year-old male presented following a high-speed motor vehicle collision...")
- Include: key diagnoses with ICD-10 codes, major procedures with CPT codes,
  length of stay, disposition, total charges
- Include COST DENSITY insight: what percentage of total charges occurred in the first 24 hours
- Show the CARE TRAJECTORY shift (e.g., "from trauma resuscitation to surgical recovery to rehabilitation")
- Use "the patient stabilized" not "the patient was lucky." Avoid subjective language.
- Keep to ONE paragraph, 5-7 sentences maximum
- Use exact dollar amounts and dates from the data
- Do NOT speculate or add information not in the data
- End with the total billed amount

### DATA ###
{episode_data}

Write the executive summary paragraph:"""


DATE_NARRATIVE_PROMPT = """You are a medical claims audit narrator. Generate a clinical narrative
for a single date of service within an inpatient episode.

### RULES ###
- Write in third person, past tense, strictly evidence-based medical-legal tone
- Describe what happened clinically: procedures, assessments, test results, medication changes
- Link charges to clinical events where possible (e.g., "A CT scan of the chest was performed ($1,420),
  revealing a left hemothorax")
- Include provider names and roles when available
- Show the CARE TRAJECTORY: how care shifted from the previous day (admission → stabilization → recovery)
  If a "previous_day_context" field is present in the data, use it to describe the transition
  (e.g., "Following ICU stabilization the prior day, the patient was transferred to step-down...")
- Keep to 3-5 sentences per date
- Use exact values from the data (vitals, lab values, dollar amounts)
- Do NOT speculate or infer clinical decisions not documented
- NEGATIVE CONSTRAINT: If a charge exists on this date but has no linked clinical event
  (linked_event_id is null or missing), explicitly state: "A financial charge for [description]
  ($X.XX) was noted without a corresponding clinical entry in the provided documentation."
  Do NOT invent clinical context for unlinked charges.

### DATE: {service_date} ###
{date_data}

Write the clinical narrative for this date:"""


QA_NARRATIVE_PROMPT = """You are a medical claims audit narrator. Translate these QA audit findings
into a professional narrative summary for a claims review committee.

### RULES ###
- Write in professional, objective, strictly evidence-based tone — no alarmist language
- BEGIN with the overall Risk Score (X/100) and whether the clinical documentation density
  supports the total billed amount. Justify the risk level in one sentence.
- DISTINGUISH between two categories of findings:
  1. TECHNICAL FLAWS (missing signatures, documentation gaps) — affect claim validity but
     may be correctable
  2. FINANCIAL RISKS (high-value charges, cost anomalies, potential overbilling) — affect
     reimbursement directly
- Group findings logically under these two categories
- For each finding, explain what was found and what it means for the claim
- Use specific dates, amounts, and provider names from the data
- Keep the total narrative to 2-3 paragraphs
- End with an overall assessment: "Based on the evidence reviewed, this claim [presents/does not present]
  significant concerns for [category]."

### QA FINDINGS ###
{qa_data}

### EPISODE CONTEXT ###
{episode_context}

Write the audit narrative:"""


MEDICATION_NARRATIVE_PROMPT = """You are a medical claims audit narrator. Generate a medication
reconciliation narrative for this episode of care.

### RULES ###
- Describe the MEDICATION TRAJECTORY through the care phases:
  Pre-hospital (EMS) → Emergency Department → Operating Room → ICU → Step-Down → Discharge
- Group by therapeutic purpose:
  1. PAIN MANAGEMENT: Describe the opioid ladder (IV fentanyl → PCA morphine → PO oxycodone)
     and the clinical rationale for each transition
  2. PROPHYLAXIS: DVT prevention, surgical antibiotic prophylaxis, post-splenectomy vaccines
  3. RESUSCITATION: IV fluids, blood products (pRBCs, FFP, platelets), vasopressors
  4. HOME MEDICATIONS: Which were held on admission and when they were resumed
- Explicitly state the total opioid count and whether the transitions follow standard
  trauma pain management protocols (IV→PCA→PO is appropriate; overlapping IV opioids is not)
- Use "the patient was transitioned to" not "the patient was given" — show clinical decision-making
- Keep to 2-3 paragraphs
- Use exact drug names, doses, and routes from the data

### MEDICATION DATA ###
{med_data}

### EPISODE CONTEXT ###
{episode_context}

Write the medication reconciliation narrative:"""


# ═══════════════════════════════════════════════════════
# GEMINI API CALLER
# ═══════════════════════════════════════════════════════

def call_gemini_narrative(prompt: str, api_key: str = None) -> str:
    """Call Gemini for narrative text generation."""
    key = api_key or GEMINI_API_KEY
    if not key:
        return "[ERROR: No API key]"

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={key}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2000,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "[No response]")
        )
    except Exception as e:
        return f"[Narrative generation error: {e}]"


# ═══════════════════════════════════════════════════════
# NARRATIVE BUILDERS
# ═══════════════════════════════════════════════════════

def generate_episode_summary(timeline: dict, api_key: str = None) -> str:
    """Generate executive summary of the entire episode."""
    episode = timeline.get("episode", {})
    patient = timeline.get("patient", {})

    # Calculate cost density (first 24h % of total)
    sorted_dates = sorted(timeline.get("timeline", {}).keys())
    first_day_cost = 0
    total_cost = episode.get("total_cost", 0)
    if sorted_dates:
        first_entry = timeline["timeline"].get(sorted_dates[0], {})
        first_day_cost = first_entry.get("total_cost", 0)
    cost_density_pct = (first_day_cost / total_cost * 100) if total_cost > 0 else 0

    # Build concise episode data for the prompt
    episode_data = {
        "patient": patient,
        "episode": episode,
        "cost_density": {
            "first_24h_cost": first_day_cost,
            "first_24h_percentage": round(cost_density_pct, 1),
        },
        "dates": [],
        "providers_count": len(timeline.get("providers", {})),
        "medications_count": len(timeline.get("medications", {})),
        "facilities": list(timeline.get("facilities", {}).keys()),
    }

    for date_str in sorted_dates:
        entry = timeline["timeline"][date_str]
        episode_data["dates"].append({
            "date": date_str,
            "event_count": entry.get("event_count", 0),
            "total_cost": entry.get("total_cost", 0),
            "top_events": [
                f"{e['event_type']}: {e['description'][:80]}"
                for e in entry.get("events", [])[:5]
            ],
        })

    prompt = EPISODE_SUMMARY_PROMPT.format(
        current_date=time.strftime("%Y-%m-%d"),
        episode_data=json.dumps(episode_data, indent=2, default=str),
    )
    return call_gemini_narrative(prompt, api_key)


def generate_date_narrative(date_str: str, entry: dict,
                           episode_context: dict,
                           previous_day_summary: str = None,
                           api_key: str = None) -> str:
    """Generate clinical narrative for a single date."""
    # Round all dollar amounts to 2 decimal places for precision
    def round_costs(items):
        rounded = []
        for item in items:
            item = dict(item)
            if item.get("amount") is not None:
                item["amount"] = round(float(item["amount"]), 2)
            # Slim down: only keep fields the narrative needs
            rounded.append({
                "description": item.get("description", ""),
                "amount": item.get("amount"),
                "category": item.get("category", ""),
                "code": item.get("code"),
                "linked_event_id": item.get("linked_event_id"),
            })
        return rounded

    # Slim down events: only narrative-relevant fields
    def slim_events(events):
        slimmed = []
        for e in events:
            slimmed.append({
                "event_id": e.get("event_id"),
                "event_type": e.get("event_type"),
                "description": e.get("description", "")[:120],
                "icd_codes": e.get("icd_codes", []),
                "cpt_codes": e.get("cpt_codes", []),
                "providers": e.get("providers", []),
            })
        return slimmed

    date_data = {
        "events": slim_events(entry.get("events", [])[:15]),
        "cost_items": round_costs(entry.get("cost_items", [])[:12]),
        "vitals": entry.get("vitals", []),
        "lab_results": entry.get("lab_results", [])[:10],
        "total_cost": round(entry.get("total_cost", 0), 2),
        "source_documents": entry.get("source_documents", []),
    }

    # Inject previous day context for trajectory continuity
    if previous_day_summary:
        date_data["previous_day_context"] = previous_day_summary

    prompt = DATE_NARRATIVE_PROMPT.format(
        service_date=date_str,
        date_data=json.dumps(date_data, indent=2, default=str),
    )
    return call_gemini_narrative(prompt, api_key)


def generate_qa_narrative(qa_report: dict, timeline: dict,
                         api_key: str = None) -> str:
    """Generate audit findings narrative."""
    episode_context = {
        "episode": timeline.get("episode", {}),
        "patient": timeline.get("patient", {}),
        "total_dates": len(timeline.get("timeline", {})),
    }

    # Build lean finding summaries instead of raw JSON (context window efficiency)
    finding_summaries = []
    for f in qa_report.get("all_findings", [])[:20]:
        summary = f"{f.get('severity','?').upper()} | {f.get('category','?')} | {f.get('title','?')}"
        if f.get("affected_date"):
            summary += f" [{f['affected_date']}]"
        if f.get("financial_impact"):
            summary += f" (${f['financial_impact']:,.2f})"
        summary += f" — {f.get('description', '')[:150]}"
        finding_summaries.append(summary)

    qa_summary = {
        "risk_score": qa_report.get("risk_score"),
        "total_findings": qa_report.get("total_findings"),
        "severity_counts": qa_report.get("summary"),
        "finding_summaries": finding_summaries,
    }

    prompt = QA_NARRATIVE_PROMPT.format(
        qa_data=json.dumps(qa_summary, indent=2, default=str),
        episode_context=json.dumps(episode_context, indent=2, default=str),
    )
    return call_gemini_narrative(prompt, api_key)


def generate_medication_narrative(timeline: dict,
                                 api_key: str = None) -> str:
    """Generate medication reconciliation narrative."""
    episode_context = {
        "episode": timeline.get("episode", {}),
        "patient": timeline.get("patient", {}),
    }

    med_data = timeline.get("medications", {})

    prompt = MEDICATION_NARRATIVE_PROMPT.format(
        med_data=json.dumps(med_data, indent=2, default=str),
        episode_context=json.dumps(episode_context, indent=2, default=str),
    )
    return call_gemini_narrative(prompt, api_key)


def generate_provider_summary(timeline: dict) -> str:
    """Generate provider attestation summary (no API needed — rule-based)."""
    providers = timeline.get("providers", {})
    facilities = timeline.get("facilities", {})

    lines = []
    lines.append("PROVIDER ATTESTATION SUMMARY")
    lines.append("=" * 40)

    # Treating providers
    treating = [
        p for p in providers.values()
        if "treating_provider" in (p.get("roles") or [])
    ]
    signing = [
        p for p in providers.values()
        if p.get("signature_detected")
    ]
    unsigned_treating = [
        p for p in treating if not p.get("signature_detected")
    ]

    lines.append(f"Total providers: {len(providers)}")
    lines.append(f"Treating providers: {len(treating)}")
    lines.append(f"Providers with signatures: {len(signing)}")
    if unsigned_treating:
        names = [p["name"] for p in unsigned_treating]
        lines.append(f"⚠ Unsigned treating providers: {', '.join(names)}")
    else:
        lines.append("✅ All treating providers have documented signatures")

    lines.append("")
    lines.append("Provider details:")
    for pname, pinfo in providers.items():
        roles = ", ".join(pinfo.get("roles", []))
        sig = "✅ signed" if pinfo.get("signature_detected") else "❌ unsigned"
        docs = len(pinfo.get("source_segments", []))
        lines.append(f"  • {pinfo['name']} — {pinfo.get('specialty', 'N/A')} "
                     f"[{roles}] {sig} ({docs} docs)")

    if facilities:
        lines.append("")
        lines.append("Facilities:")
        for fname, finfo in facilities.items():
            lines.append(f"  🏨 {finfo['name']} ({finfo.get('type', 'facility')})")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# FULL NARRATIVE REPORT
# ═══════════════════════════════════════════════════════

def generate_full_report(timeline: dict, qa_report: dict,
                        api_key: str = None) -> dict:
    """Generate the complete narrative audit report.
    
    Section order (per Gemini recommendation):
      I.   Executive Summary (Episode)
      II.  Audit Findings & Risk Score (QA)
      III. Medication Reconciliation (Med Ledger)
      IV.  Detailed Clinical Timeline (Date-by-date)
      V.   Provider Attestation Summary
      VI.  Itemized Cost Summary
    """
    t0 = time.time()
    report = {
        "agent": "agent5_narrative",
        "version": "v2",
        "sections": {},
        "section_order": [
            "I_executive_summary",
            "II_audit_findings",
            "III_medication_reconciliation",
            "IV_clinical_timeline",
            "V_provider_attestation",
            "VI_cost_summary",
        ],
        "generation_time_ms": 0,
    }

    # ── I. Executive Summary ──
    print("\n  📝 I. Generating executive summary...")
    report["sections"]["I_executive_summary"] = generate_episode_summary(
        timeline, api_key
    )

    # ── II. Audit Findings & Risk Score ──
    print("  📝 II. Generating audit findings narrative...")
    report["sections"]["II_audit_findings"] = generate_qa_narrative(
        qa_report, timeline, api_key
    )

    # ── III. Medication Reconciliation ──
    print("  📝 III. Generating medication reconciliation...")
    report["sections"]["III_medication_reconciliation"] = generate_medication_narrative(
        timeline, api_key
    )

    # ── IV. Detailed Clinical Timeline (iterative with day chaining) ──
    print("  📝 IV. Generating date-by-date clinical timeline...")
    date_narratives = {}
    episode_ctx = {
        "episode": timeline.get("episode", {}),
        "patient": timeline.get("patient", {}),
    }
    previous_day_summary = None  # Chain for trajectory continuity
    for date_str in sorted(timeline.get("timeline", {}).keys()):
        entry = timeline["timeline"][date_str]
        print(f"     → {date_str}...")
        narrative = generate_date_narrative(
            date_str, entry, episode_ctx,
            previous_day_summary=previous_day_summary,
            api_key=api_key,
        )
        date_narratives[date_str] = narrative
        # Pass this narrative as context for the next day (first 300 chars)
        previous_day_summary = narrative[:300]
    report["sections"]["IV_clinical_timeline"] = date_narratives

    # ── V. Provider Attestation Summary (rule-based) ──
    print("  📝 V. Generating provider attestation...")
    report["sections"]["V_provider_attestation"] = generate_provider_summary(timeline)

    # ── VI. Itemized Cost Summary (rule-based) ──
    print("  📝 VI. Building cost summary...")
    report["sections"]["VI_cost_summary"] = build_cost_summary(timeline)

    report["generation_time_ms"] = int((time.time() - t0) * 1000)
    return report


def build_cost_summary(timeline: dict) -> str:
    """Build a structured cost summary (no API needed)."""
    lines = []
    lines.append("ITEMIZED COST SUMMARY")
    lines.append("=" * 50)

    episode = timeline.get("episode", {})
    lines.append(f"Episode: {episode.get('start_date')} → {episode.get('end_date')} "
                f"({episode.get('duration_days', '?')} days)")
    lines.append(f"Total Billed: ${episode.get('total_cost', 0):,.2f}")
    lines.append("")

    # By date
    lines.append("DAILY BREAKDOWN:")
    lines.append("-" * 50)
    for date_str in sorted(timeline.get("timeline", {}).keys()):
        entry = timeline["timeline"][date_str]
        tc = entry.get("total_cost", 0)
        cc = entry.get("cost_count", 0)
        lines.append(f"  {date_str}: ${tc:,.2f} ({cc} line items)")

    # By category
    lines.append("")
    lines.append("CATEGORY BREAKDOWN:")
    lines.append("-" * 50)
    cat_totals = {}
    for date_str, entry in timeline.get("timeline", {}).items():
        for cost in entry.get("cost_items", []):
            cat = cost.get("category", "other")
            amt = cost.get("amount") or 0
            cat_totals[cat] = cat_totals.get(cat, 0) + amt

    for cat in sorted(cat_totals.keys(), key=lambda c: -cat_totals[c]):
        pct = (cat_totals[cat] / episode.get("total_cost", 1)) * 100
        lines.append(f"  {cat:25s} ${cat_totals[cat]:>12,.2f}  ({pct:.1f}%)")

    # Billing cycles
    lines.append("")
    lines.append("BILLING CYCLES:")
    lines.append("-" * 50)
    for cycle_key in sorted(timeline.get("billing_cycles", {}).keys()):
        cycle = timeline["billing_cycles"][cycle_key]
        lines.append(f"  {cycle_key}: ${cycle['total_cost']:,.2f} "
                    f"({cycle['cost_count']} charges, {len(cycle['dates'])} dates)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # ── 1. Locate inputs ──
    timeline_path = None
    qa_path = None

    if len(sys.argv) > 2:
        timeline_path = Path(sys.argv[1])
        qa_path = Path(sys.argv[2])
    elif len(sys.argv) > 1:
        timeline_path = Path(sys.argv[1])
    else:
        for candidate in [
            Path("agent3_timeline.json"),
            Path("chronocare/agent3_timeline.json"),
        ]:
            if candidate.exists():
                timeline_path = candidate
                break

    if not timeline_path or not timeline_path.exists():
        print("Usage: python agent5_narrative.py <agent3_timeline.json> [agent4_qa_report.json]")
        sys.exit(1)

    # Auto-find QA report
    if not qa_path:
        qa_candidate = timeline_path.parent / "agent4_qa_report.json"
        if qa_candidate.exists():
            qa_path = qa_candidate

    with open(timeline_path) as f:
        timeline_data = json.load(f)

    qa_data = {}
    if qa_path and qa_path.exists():
        with open(qa_path) as f:
            qa_data = json.load(f)

    # ── 2. API key ──
    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("❌ Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    print("=" * 70)
    print("  ChronoCare AI — Agent 5: Narrative Generator")
    print(f"  Timeline: {timeline_path}")
    print(f"  QA Report: {qa_path or 'not found'}")
    print("=" * 70)

    # Generate
    report = generate_full_report(timeline_data, qa_data, api_key)

    # Display
    print("\n" + "=" * 70)
    print("  NARRATIVE AUDIT REPORT")
    print("=" * 70)

    section_titles = {
        "I_executive_summary": "I. EXECUTIVE SUMMARY",
        "II_audit_findings": "II. AUDIT FINDINGS & RISK SCORE",
        "III_medication_reconciliation": "III. MEDICATION RECONCILIATION",
        "IV_clinical_timeline": "IV. DETAILED CLINICAL TIMELINE",
        "V_provider_attestation": "V. PROVIDER ATTESTATION",
        "VI_cost_summary": "VI. ITEMIZED COST SUMMARY",
    }

    for section_key in report.get("section_order", report["sections"].keys()):
        content = report["sections"].get(section_key)
        if content is None:
            continue

        title = section_titles.get(section_key, section_key.replace("_", " ").upper())
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")

        if isinstance(content, dict):
            # Date narratives
            for date_str, narrative in content.items():
                print(f"\n  📅 {date_str}:")
                # Word wrap
                words = narrative.split()
                line = "     "
                for word in words:
                    if len(line) + len(word) + 1 > 100:
                        print(line)
                        line = "     " + word
                    else:
                        line += " " + word if line.strip() else "     " + word
                if line.strip():
                    print(line)
        else:
            # Single narrative
            for raw_line in content.split("\n"):
                if not raw_line.strip():
                    print()
                    continue
                words = raw_line.split()
                line = "  "
                for word in words:
                    if len(line) + len(word) + 1 > 100:
                        print(line)
                        line = "  " + word
                    else:
                        line += " " + word if line.strip() else "  " + word
                if line.strip():
                    print(line)

    print(f"\n⏱️  Generated in {report['generation_time_ms']}ms")

    # Save
    output_path = timeline_path.parent / "agent5_narrative_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  Saved: {output_path}")
    print("  ✅ Done!")
