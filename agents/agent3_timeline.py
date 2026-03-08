"""
ChronoCare AI — Agent 3: Timeline Builder
============================================
Takes extraction results from Agent 2 → builds a unified date-keyed timeline:
  - Cross-segment event deduplication (same diagnosis from ambulance + ED + billing)
  - Medication deduplication (Fentanyl in ambulance AND billing = one dose + its charge)
  - Date normalization and conflict resolution
  - Billing cycle grouping (1st-15th, 16th-end)
  - Cost aggregation per date and per cycle
  - Provider consolidation across documents
  - Source document linking for audit trail
  - The "Golden Link" — cost items linked to clinical events

The output is the core "click any date, see everything" structure.

Self-contained — no imports from rest of project.
"""
import json
import time
import re
import os
import hashlib
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple


# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

# Billing cycle split: day 1-15 = cycle 1, day 16-end = cycle 2
CYCLE_SPLIT_DAY = 15


# ═══════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════

class TimelineEvent:
    """A deduplicated clinical event on the timeline."""
    def __init__(self, event_id, event_type, date, description,
                 icd_codes=None, cpt_codes=None, body_site=None,
                 providers=None, confidence=0.0, source_quotes=None,
                 source_segments=None, linked_cost_ids=None):
        self.event_id = event_id
        self.event_type = event_type
        self.date = date
        self.description = description
        self.icd_codes = icd_codes or []
        self.cpt_codes = cpt_codes or []
        self.body_site = body_site
        self.providers = providers or []
        self.confidence = confidence
        self.source_quotes = source_quotes or []
        self.source_segments = source_segments or []
        self.linked_cost_ids = linked_cost_ids or []

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "date": self.date,
            "description": self.description,
            "icd_codes": self.icd_codes,
            "cpt_codes": self.cpt_codes,
            "body_site": self.body_site,
            "providers": self.providers,
            "confidence": self.confidence,
            "source_quotes": self.source_quotes,
            "source_segments": self.source_segments,
            "linked_cost_ids": self.linked_cost_ids,
        }


class TimelineCost:
    """A cost item on the timeline."""
    def __init__(self, cost_id, date, category, description,
                 amount, code=None, linked_event_id=None,
                 source_segment=None, confidence=0.0):
        self.cost_id = cost_id
        self.date = date
        self.category = category
        self.description = description
        self.amount = amount
        self.code = code
        self.linked_event_id = linked_event_id
        self.source_segment = source_segment
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
            "source_segment": self.source_segment,
            "confidence": self.confidence,
        }


class TimelineMedication:
    """A deduplicated medication entry."""
    def __init__(self, med_id, name, dose=None, route=None,
                 frequency=None, date_first_seen=None,
                 date_last_seen=None, contexts=None,
                 source_segments=None):
        self.med_id = med_id
        self.name = name
        self.dose = dose
        self.route = route
        self.frequency = frequency
        self.date_first_seen = date_first_seen
        self.date_last_seen = date_last_seen
        self.contexts = contexts or []
        self.source_segments = source_segments or []

    def to_dict(self):
        return {
            "med_id": self.med_id,
            "name": self.name,
            "dose": self.dose,
            "route": self.route,
            "frequency": self.frequency,
            "date_first_seen": self.date_first_seen,
            "date_last_seen": self.date_last_seen,
            "contexts": self.contexts,
            "source_segments": self.source_segments,
        }


class DateEntry:
    """Everything that happened on a single date — the core 'click to expand' unit."""
    def __init__(self, service_date, billing_cycle=None):
        self.service_date = service_date
        self.billing_cycle = billing_cycle
        self.events = []          # TimelineEvent list
        self.cost_items = []      # TimelineCost list
        self.medications = []     # TimelineMedication list
        self.providers = []       # consolidated provider list
        self.vitals = []          # vitals_series entries for this date
        self.lab_results = []     # lab results for this date
        self.source_documents = []  # which files contributed to this date
        self.total_cost = 0.0
        self.summary = None       # AI-generated narrative (Agent 5)

    def to_dict(self):
        return {
            "service_date": self.service_date,
            "billing_cycle": self.billing_cycle,
            "total_cost": self.total_cost,
            "event_count": len(self.events),
            "cost_count": len(self.cost_items),
            "med_count": len(self.medications),
            "events": [e.to_dict() for e in self.events],
            "cost_items": [c.to_dict() for c in self.cost_items],
            "medications": [m.to_dict() for m in self.medications],
            "providers": self.providers,
            "vitals": self.vitals,
            "lab_results": self.lab_results,
            "source_documents": self.source_documents,
            "summary": self.summary,
        }


class Timeline:
    """The complete date-keyed timeline for an episode of care."""
    def __init__(self):
        self.patient_name = None
        self.patient_dob = None
        self.episode_start = None
        self.episode_end = None
        self.date_entries = {}       # date_str -> DateEntry
        self.billing_cycles = {}     # cycle_key -> {dates, total_cost}
        self.all_providers = {}      # provider_name -> provider_info
        self.facilities = {}         # facility_name -> facility_info
        self.all_medications = {}    # med_key -> TimelineMedication
        self.total_cost = 0.0
        self.total_events = 0
        self.source_files = []
        self.event_id_map = {}       # Agent2 ID -> Agent3 ID mapping
        self.build_time_ms = 0

    def to_dict(self):
        sorted_dates = sorted(self.date_entries.keys())
        return {
            "patient": {
                "name": self.patient_name,
                "dob": self.patient_dob,
            },
            "episode": {
                "start_date": self.episode_start,
                "end_date": self.episode_end,
                "duration_days": self._duration_days(),
                "total_cost": self.total_cost,
                "total_events": self.total_events,
                "total_dates": len(sorted_dates),
            },
            "timeline": {
                d: self.date_entries[d].to_dict() for d in sorted_dates
            },
            "billing_cycles": self.billing_cycles,
            "providers": self.all_providers,
            "facilities": self.facilities,
            "medications": {
                k: v.to_dict() for k, v in self.all_medications.items()
            },
            "source_files": self.source_files,
            "event_id_map": self.event_id_map,
            "build_time_ms": self.build_time_ms,
        }

    def _duration_days(self):
        if self.episode_start and self.episode_end:
            try:
                s = datetime.strptime(self.episode_start, "%Y-%m-%d")
                e = datetime.strptime(self.episode_end, "%Y-%m-%d")
                return (e - s).days
            except ValueError:
                pass
        return None


# ═══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════

def get_billing_cycle(date_str: str) -> str:
    """Assign a billing cycle key: 'YYYY-MM-cycle1' or 'YYYY-MM-cycle2'."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        cycle = "cycle1" if dt.day <= CYCLE_SPLIT_DAY else "cycle2"
        return f"{dt.strftime('%Y-%m')}-{cycle}"
    except (ValueError, TypeError):
        return "unknown-cycle"


def normalize_date(date_str: str) -> Optional[str]:
    """Normalize date to YYYY-MM-DD, filtering out DOBs and far-future dates."""
    if not date_str or not isinstance(date_str, str):
        return None
    # Try parsing
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d"]:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def is_service_date(date_str: str, patient_dob: str = None) -> bool:
    """Filter out DOB and obviously non-service dates."""
    if not date_str:
        return False
    if patient_dob and date_str == patient_dob:
        return False
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        if dt.year < 2020 or dt.year > 2027:
            return False
    except ValueError:
        return False
    return True


def event_dedup_key(event: dict) -> str:
    """Generate a dedup key for cross-segment event matching."""
    etype = event.get("event_type", "").lower()
    date = event.get("date", "")
    desc = event.get("description", "").lower()

    # For diagnoses: key on ICD code if available, else description keywords
    if etype == "diagnosis":
        codes = event.get("icd_codes", [])
        if codes:
            return f"dx|{date}|{sorted(codes)[0]}"
        # Extract key terms from description
        keywords = re.sub(r'[^a-z0-9\s]', '', desc).split()[:4]
        return f"dx|{date}|{'_'.join(keywords)}"

    # For procedures: key on CPT code if available
    if etype == "procedure":
        codes = event.get("cpt_codes", [])
        if codes:
            return f"proc|{date}|{sorted(codes)[0]}"
        keywords = re.sub(r'[^a-z0-9\s]', '', desc).split()[:4]
        return f"proc|{date}|{'_'.join(keywords)}"

    # For medications: key on drug name + date
    if etype == "medication":
        drug = re.sub(r'[^a-z0-9]', '', desc.split()[0].lower()) if desc else "unknown"
        return f"med|{date}|{drug}"

    # For labs: key on test name + date
    if etype in ("test", "lab_result"):
        keywords = re.sub(r'[^a-z0-9\s]', '', desc).split()[:3]
        return f"lab|{date}|{'_'.join(keywords)}"

    # For admissions/discharges: one per date
    if etype in ("admission", "discharge", "transfer"):
        return f"{etype}|{date}"

    # Default: type + date + first 5 description words
    keywords = re.sub(r'[^a-z0-9\s]', '', desc).split()[:5]
    return f"{etype}|{date}|{'_'.join(keywords)}"


def med_dedup_key(med: dict) -> str:
    """Generate a dedup key for medication matching."""
    name = re.sub(r'[^a-z0-9]', '', (med.get("name") or "").lower())
    dose = re.sub(r'[^a-z0-9]', '', (med.get("dose") or "").lower())
    route = (med.get("route") or "").lower()
    return f"{name}|{dose}|{route}"


def merge_providers(existing: dict, new_provider: dict) -> dict:
    """Merge provider info, keeping the most complete version."""
    merged = dict(existing)
    for key in ["specialty", "npi", "facility"]:
        if new_provider.get(key) and not merged.get(key):
            merged[key] = new_provider[key]
    # Merge roles
    existing_roles = set(merged.get("roles", []))
    new_role = new_provider.get("role") or new_provider.get("provider_role")
    if new_role:
        existing_roles.add(new_role)
    merged["roles"] = list(existing_roles)
    # Signature: true if any source detected it
    if new_provider.get("signature_detected"):
        merged["signature_detected"] = True
    # Keep the longer/more complete name
    if len(new_provider.get("name", "")) > len(merged.get("name", "")):
        merged["name"] = new_provider["name"]
    # Track source segments
    sources = set(merged.get("source_segments", []))
    if new_provider.get("source_segment"):
        sources.add(new_provider["source_segment"])
    merged["source_segments"] = list(sources)
    return merged


def normalize_provider_name(name: str) -> str:
    """Normalize provider name for dedup matching."""
    n = name.lower().strip()
    # Strip common prefixes
    n = re.sub(r'^(dr\.?\s*|mr\.?\s*|ms\.?\s*|mrs\.?\s*)', '', n)
    # Strip common suffixes
    n = re.sub(r',?\s*(md|do|rn|np|pa|emt-[bp]|emt|phd|dpm|dds)\s*$', '', n)
    return n.strip()


def get_last_name(name: str) -> str:
    """Extract last name from a provider name."""
    parts = name.strip().split()
    if len(parts) >= 2:
        # Skip middle initials (single letter or letter with period)
        non_initials = [p for p in parts if len(p.replace('.', '')) > 1]
        if non_initials:
            return non_initials[-1].lower()
    return name.lower().strip()


# Facility keywords — these are NOT providers
FACILITY_KEYWORDS = [
    "medical center", "hospital", "clinic", "laboratory", "lab",
    "diagnostic", "urgent care", "ambulance", "ems", "health system",
    "pharmacy", "imaging center", "rehab", "nursing", "lone star",
    "meridian", "regional", "memorial", "community", "university",
]

def is_facility(name: str) -> bool:
    """Check if a 'provider' entry is actually a facility."""
    n = name.lower()
    return any(kw in n for kw in FACILITY_KEYWORDS)


# ═══════════════════════════════════════════════════════
# TIMELINE BUILDER
# ═══════════════════════════════════════════════════════

def build_timeline(agent2_results: dict) -> Timeline:
    """
    Build a unified timeline from Agent 2 extraction results.
    
    agent2_results: the full agent2_extraction_results.json content
    """
    t0 = time.time()
    timeline = Timeline()

    extractions = agent2_results.get("extractions", [])

    # ── Pass 1: Collect all events, costs, meds, vitals, labs ──
    all_events = []       # (dedup_key, event_dict, source_segment)
    all_costs = []        # (cost_dict, source_segment)
    all_meds = []         # (dedup_key, med_dict, source_segment)
    all_vitals = []       # (vital_dict, source_segment)
    all_labs = []         # (lab_dict, source_segment)
    all_providers = []    # (provider_dict, source_segment)

    for ext in extractions:
        if ext.get("error"):
            continue

        seg_id = ext.get("segment_id", "unknown")
        seg_fname = ext.get("source_filename", "unknown")
        doc_type = ext.get("doc_type", "unknown")
        source_ref = f"{seg_fname}|{doc_type}|{seg_id}"

        # Track source files
        if seg_fname not in timeline.source_files and seg_fname != "unknown":
            timeline.source_files.append(seg_fname)

        # Events
        for evt in ext.get("events", []):
            key = event_dedup_key(evt)
            evt["_source_segment"] = source_ref
            evt["_source_doc_type"] = doc_type
            all_events.append((key, evt, source_ref))

        # Costs
        for cost in ext.get("cost_items", []):
            cost["_source_segment"] = source_ref
            all_costs.append((cost, source_ref))

        # Medications
        for med in ext.get("medications", []):
            key = med_dedup_key(med)
            med["_source_segment"] = source_ref
            all_meds.append((key, med, source_ref))

        # Vitals
        for vital in ext.get("vitals_series", []):
            vital["_source_segment"] = source_ref
            all_vitals.append((vital, source_ref))

        # Labs
        for lab in ext.get("lab_results", []):
            lab["_source_segment"] = source_ref
            all_labs.append((lab, source_ref))

        # Providers
        for prov in ext.get("providers", []):
            prov["source_segment"] = source_ref
            all_providers.append((prov, source_ref))

    # ── Pass 2: Deduplicate events ──
    deduped_events = {}  # dedup_key -> TimelineEvent
    event_id_counter = 0
    # Map Agent 2 event IDs → Agent 3 timeline event IDs (the Golden Link bridge)
    event_id_map = {}  # "evt_002" -> "tl_evt_0001"

    for key, evt, source_ref in all_events:
        old_id = evt.get("event_id")  # Agent 2's ID
        if key in deduped_events:
            # Merge into existing event
            existing = deduped_events[key]
            # Map old ID to existing timeline ID
            if old_id:
                event_id_map[old_id] = existing.event_id
            # Add source
            if source_ref not in existing.source_segments:
                existing.source_segments.append(source_ref)
            # Merge codes
            for code in evt.get("icd_codes", []):
                if code not in existing.icd_codes:
                    existing.icd_codes.append(code)
            for code in evt.get("cpt_codes", []):
                if code not in existing.cpt_codes:
                    existing.cpt_codes.append(code)
            # Keep highest confidence
            conf = evt.get("confidence", 0)
            if conf > existing.confidence:
                existing.confidence = conf
                existing.description = evt.get("description", existing.description)
            # Add source quote
            sq = evt.get("source_quote")
            if sq and sq not in existing.source_quotes:
                existing.source_quotes.append(sq)
            # Add provider
            pname = evt.get("provider_name")
            if pname and pname not in existing.providers:
                existing.providers.append(pname)
        else:
            event_id_counter += 1
            new_id = f"tl_evt_{event_id_counter:04d}"
            # Map old ID to new timeline ID
            if old_id:
                event_id_map[old_id] = new_id
            te = TimelineEvent(
                event_id=new_id,
                event_type=evt.get("event_type", "other"),
                date=normalize_date(evt.get("date")),
                description=evt.get("description", ""),
                icd_codes=evt.get("icd_codes", []),
                cpt_codes=evt.get("cpt_codes", []),
                body_site=evt.get("body_site"),
                providers=[evt.get("provider_name")] if evt.get("provider_name") else [],
                confidence=evt.get("confidence", 0.5),
                source_quotes=[evt.get("source_quote")] if evt.get("source_quote") else [],
                source_segments=[source_ref],
            )
            deduped_events[key] = te

    # ── Pass 3: Deduplicate medications ──
    deduped_meds = {}  # dedup_key -> TimelineMedication
    med_id_counter = 0

    for key, med, source_ref in all_meds:
        if key in deduped_meds:
            existing = deduped_meds[key]
            if source_ref not in existing.source_segments:
                existing.source_segments.append(source_ref)
            ctx = med.get("context")
            if ctx and ctx not in existing.contexts:
                existing.contexts.append(ctx)
            # Update date range
            ds = normalize_date(med.get("date_started"))
            if ds:
                if not existing.date_first_seen or ds < existing.date_first_seen:
                    existing.date_first_seen = ds
                if not existing.date_last_seen or ds > existing.date_last_seen:
                    existing.date_last_seen = ds
        else:
            med_id_counter += 1
            tm = TimelineMedication(
                med_id=f"tl_med_{med_id_counter:03d}",
                name=med.get("name", "Unknown"),
                dose=med.get("dose"),
                route=med.get("route"),
                frequency=med.get("frequency"),
                date_first_seen=normalize_date(med.get("date_started")),
                date_last_seen=normalize_date(med.get("date_stopped")),
                contexts=[med.get("context")] if med.get("context") else [],
                source_segments=[source_ref],
            )
            deduped_meds[key] = tm

    timeline.all_medications = deduped_meds

    # ── Pass 4: Consolidate providers (fuzzy match + facility separation) ──
    provider_map = {}    # normalized_name -> provider_info
    facility_map = {}    # facility_name -> facility_info
    last_name_index = {} # last_name -> normalized_name (for fuzzy matching)

    for prov, source_ref in all_providers:
        name = (prov.get("name") or "").strip()
        if not name:
            continue

        # Separate facilities from providers
        if is_facility(name):
            fn = name.lower().strip()
            if fn not in facility_map:
                facility_map[fn] = {
                    "name": name,
                    "type": prov.get("specialty") or prov.get("role", "facility"),
                    "source_segments": [source_ref],
                }
            else:
                if source_ref not in facility_map[fn]["source_segments"]:
                    facility_map[fn]["source_segments"].append(source_ref)
            continue

        norm_name = normalize_provider_name(name)

        # Try exact normalized match first
        matched_key = None
        if norm_name in provider_map:
            matched_key = norm_name
        else:
            # Try last-name + specialty fuzzy match
            # e.g., "Dr. Sharma" matches "Priya Sharma" if specialty overlaps
            last = get_last_name(norm_name)
            if last in last_name_index:
                candidate_key = last_name_index[last]
                candidate = provider_map[candidate_key]
                # Match if specialties overlap or one is a subset
                cand_spec = (candidate.get("specialty") or "").lower()
                new_spec = (prov.get("specialty") or "").lower()
                if cand_spec and new_spec:
                    # Check for keyword overlap
                    cand_words = set(cand_spec.split())
                    new_words = set(new_spec.split())
                    if cand_words & new_words:
                        matched_key = candidate_key
                elif cand_spec or new_spec:
                    # One has specialty, one doesn't — still merge on last name
                    matched_key = candidate_key

        if matched_key:
            provider_map[matched_key] = merge_providers(
                provider_map[matched_key], prov
            )
        else:
            provider_map[norm_name] = {
                "name": name,
                "specialty": prov.get("specialty"),
                "npi": prov.get("npi"),
                "facility": prov.get("facility"),
                "roles": [prov.get("role")] if prov.get("role") else [],
                "signature_detected": prov.get("signature_detected", False),
                "source_segments": [source_ref],
            }
            # Index by last name for fuzzy matching
            last = get_last_name(norm_name)
            if last not in last_name_index:
                last_name_index[last] = norm_name

    timeline.all_providers = provider_map
    timeline.facilities = facility_map
    timeline.event_id_map = event_id_map

    # ── Pass 5: Build date entries ──
    # Determine patient info from extractions
    for ext in extractions:
        if ext.get("error"):
            continue
        pname = ext.get("patient_name")
        if pname and pname != "Unknown":
            timeline.patient_name = pname
        pdob = ext.get("patient_dob")
        if pdob:
            timeline.patient_dob = pdob
        if timeline.patient_name and timeline.patient_dob:
            break

    # Populate date entries with events
    for key, te in deduped_events.items():
        d = te.date
        if not d or not is_service_date(d):
            continue
        if d not in timeline.date_entries:
            timeline.date_entries[d] = DateEntry(
                service_date=d,
                billing_cycle=get_billing_cycle(d),
            )
        timeline.date_entries[d].events.append(te)

    # Add cost items to date entries
    cost_id_counter = 0
    for cost, source_ref in all_costs:
        cost_id_counter += 1
        d = normalize_date(cost.get("date"))
        if not d or not is_service_date(d):
            continue
        if d not in timeline.date_entries:
            timeline.date_entries[d] = DateEntry(
                service_date=d,
                billing_cycle=get_billing_cycle(d),
            )

        amt = cost.get("amount")
        if amt is not None:
            try:
                amt = float(amt)
            except (ValueError, TypeError):
                amt = None

        # Remap linked_event_id from Agent 2 namespace to Agent 3 namespace
        raw_linked = cost.get("linked_event_id")
        remapped_linked = event_id_map.get(raw_linked, raw_linked) if raw_linked else None

        tc = TimelineCost(
            cost_id=f"tl_cost_{cost_id_counter:04d}",
            date=d,
            category=cost.get("category", "other"),
            description=cost.get("description", ""),
            amount=amt,
            code=cost.get("code"),
            linked_event_id=remapped_linked,
            source_segment=source_ref,
            confidence=cost.get("confidence", 0.5),
        )
        timeline.date_entries[d].cost_items.append(tc)

    # Build reverse links: event → cost_ids
    for d, entry in timeline.date_entries.items():
        event_lookup = {e.event_id: e for e in entry.events}
        for cost in entry.cost_items:
            if cost.linked_event_id and cost.linked_event_id in event_lookup:
                evt = event_lookup[cost.linked_event_id]
                if cost.cost_id not in evt.linked_cost_ids:
                    evt.linked_cost_ids.append(cost.cost_id)

    # Add vitals to date entries
    for vital, source_ref in all_vitals:
        d = normalize_date(vital.get("date"))
        if not d:
            continue
        if d in timeline.date_entries:
            timeline.date_entries[d].vitals.append(vital)

    # Add labs to date entries
    for lab, source_ref in all_labs:
        d = normalize_date(lab.get("specimen_date"))
        if not d:
            continue
        if d not in timeline.date_entries:
            timeline.date_entries[d] = DateEntry(
                service_date=d,
                billing_cycle=get_billing_cycle(d),
            )
        timeline.date_entries[d].lab_results.append(lab)

    # ── Pass 6: Aggregate costs and build billing cycles ──
    cycle_totals = defaultdict(lambda: {"dates": [], "total_cost": 0.0, "cost_count": 0})

    for d, entry in timeline.date_entries.items():
        # Sum costs
        entry.total_cost = sum(
            c.amount for c in entry.cost_items if c.amount is not None
        )
        timeline.total_cost += entry.total_cost

        # Count events
        timeline.total_events += len(entry.events)

        # Track source documents
        source_docs = set()
        for evt in entry.events:
            for src in evt.source_segments:
                source_docs.add(src.split("|")[0])
        entry.source_documents = list(source_docs)

        # Billing cycle aggregation
        cycle_key = entry.billing_cycle
        if cycle_key:
            if d not in cycle_totals[cycle_key]["dates"]:
                cycle_totals[cycle_key]["dates"].append(d)
            cycle_totals[cycle_key]["total_cost"] += entry.total_cost
            cycle_totals[cycle_key]["cost_count"] += len(entry.cost_items)

    # Sort dates within each cycle
    for key in cycle_totals:
        cycle_totals[key]["dates"] = sorted(cycle_totals[key]["dates"])

    timeline.billing_cycles = dict(cycle_totals)

    # ── Pass 7: Determine episode boundaries ──
    sorted_dates = sorted(timeline.date_entries.keys())
    if sorted_dates:
        timeline.episode_start = sorted_dates[0]
        timeline.episode_end = sorted_dates[-1]

    timeline.build_time_ms = int((time.time() - t0) * 1000)
    return timeline


# ═══════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # ── 1. Locate Agent 2 results ──
    results_path = None
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
    else:
        for candidate in [
            Path("agent2_extraction_results.json"),
            Path("chronocare/agent2_extraction_results.json"),
        ]:
            if candidate.exists():
                results_path = candidate
                break

    if not results_path or not results_path.exists():
        print("Usage: python agent3_timeline.py [agent2_extraction_results.json]")
        sys.exit(1)

    with open(results_path) as f:
        agent2_data = json.load(f)

    print("=" * 70)
    print("  ChronoCare AI — Agent 3: Timeline Builder")
    print(f"  Input: {results_path}")
    print("=" * 70)

    # Build timeline
    timeline = build_timeline(agent2_data)
    tl = timeline.to_dict()

    # Display
    print(f"\n📅 EPISODE: {tl['episode']['start_date']} → {tl['episode']['end_date']}"
          f" ({tl['episode']['duration_days']} days)")
    print(f"💰 TOTAL COST: ${tl['episode']['total_cost']:,.2f}")
    print(f"📊 {tl['episode']['total_events']} events across "
          f"{tl['episode']['total_dates']} dates")
    print(f"⏱️  Built in {tl['build_time_ms']}ms")

    # Timeline dates
    print(f"\n{'─' * 70}")
    print("  DATE-BY-DATE TIMELINE")
    print(f"{'─' * 70}")

    for date_str in sorted(tl["timeline"].keys()):
        entry = tl["timeline"][date_str]
        cycle = entry.get("billing_cycle", "")
        cost = entry.get("total_cost", 0)
        evt_count = entry.get("event_count", 0)
        cost_count = entry.get("cost_count", 0)
        med_count = entry.get("med_count", 0)
        sources = entry.get("source_documents", [])

        print(f"\n  📅 {date_str} [{cycle}]")
        print(f"     {evt_count} events | {cost_count} costs | "
              f"{med_count} meds | ${cost:,.2f}")
        print(f"     Sources: {', '.join(sources) if sources else 'n/a'}")

        # Top events
        for evt in entry.get("events", [])[:4]:
            codes = evt.get("icd_codes", []) + evt.get("cpt_codes", [])
            code_str = f" [{', '.join(codes)}]" if codes else ""
            src_count = len(evt.get("source_segments", []))
            multi = f" (×{src_count} docs)" if src_count > 1 else ""
            print(f"     🏥 [{evt['event_type']}] {evt['description'][:55]}"
                  f"{code_str}{multi}")
        remaining = evt_count - 4
        if remaining > 0:
            print(f"     ... +{remaining} more events")

        # Top costs
        for c in entry.get("cost_items", [])[:3]:
            if c.get("amount") is not None:
                linked = f" → {c['linked_event_id']}" if c.get("linked_event_id") else ""
                print(f"     💰 ${c['amount']:,.2f} ({c['category']}) "
                      f"{c['description'][:35]}{linked}")
        remaining_costs = cost_count - 3
        if remaining_costs > 0:
            print(f"     ... +{remaining_costs} more charges")

    # Billing cycles
    print(f"\n{'─' * 70}")
    print("  BILLING CYCLES")
    print(f"{'─' * 70}")
    for cycle_key in sorted(tl.get("billing_cycles", {}).keys()):
        cycle = tl["billing_cycles"][cycle_key]
        print(f"  {cycle_key}: {len(cycle['dates'])} dates | "
              f"{cycle['cost_count']} charges | ${cycle['total_cost']:,.2f}")

    # Provider roster
    print(f"\n{'─' * 70}")
    print("  PROVIDER ROSTER")
    print(f"{'─' * 70}")
    for pname, pinfo in tl.get("providers", {}).items():
        roles = ", ".join(pinfo.get("roles", []))
        spec = pinfo.get("specialty", "") or ""
        sig = "✅" if pinfo.get("signature_detected") else "❌"
        sources = len(pinfo.get("source_segments", []))
        print(f"  {pinfo['name']} — {spec} [{roles}] sig:{sig} ({sources} docs)")

    # Facilities
    facilities = tl.get("facilities", {})
    if facilities:
        print(f"\n{'─' * 70}")
        print("  FACILITIES")
        print(f"{'─' * 70}")
        for fname, finfo in facilities.items():
            ftype = finfo.get("type", "facility")
            sources = len(finfo.get("source_segments", []))
            print(f"  🏨 {finfo['name']} ({ftype}) — {sources} docs")

    # Medications
    print(f"\n{'─' * 70}")
    print("  MEDICATION LEDGER (deduplicated)")
    print(f"{'─' * 70}")
    for mkey, med in tl.get("medications", {}).items():
        sources = len(med.get("source_segments", []))
        multi = f" (×{sources} docs)" if sources > 1 else ""
        ctx = f" — {', '.join(med.get('contexts', []))}" if med.get("contexts") else ""
        print(f"  💊 {med['name']} {med.get('dose','')} {med.get('route','')} "
              f"{med.get('frequency','')}{multi}{ctx}")

    # Dedup stats
    input_events = agent2_data.get("summary", {}).get("total_events", 0)
    input_meds = agent2_data.get("summary", {}).get("total_medications", 0)
    deduped_events = tl["episode"]["total_events"]
    deduped_meds = len(tl.get("medications", {}))

    print(f"\n{'─' * 70}")
    print("  DEDUPLICATION STATS")
    print(f"{'─' * 70}")
    print(f"  Events:      {input_events} input → {deduped_events} unique "
          f"({input_events - deduped_events} duplicates removed)")
    print(f"  Medications: {input_meds} input → {deduped_meds} unique "
          f"({input_meds - deduped_meds} duplicates removed)")

    # Save
    output_path = results_path.parent / "agent3_timeline.json"
    with open(output_path, "w") as f:
        json.dump(tl, f, indent=2, default=str)

    print(f"\n  Saved: {output_path}")
    print("  ✅ Done!")
