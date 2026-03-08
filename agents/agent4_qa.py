"""
ChronoCare AI — Agent 4: QA & Anomaly Detector
=================================================
Takes the unified timeline from Agent 3 → runs audit checks:
  - Charge-without-clinical-support (billing line with no matching event)
  - Clinical-event-without-charge (procedure documented but not billed)
  - Duplicate/overlapping charges on same date
  - Timeline inconsistency detection (discharge before admission, etc.)
  - High-value charge flags (above threshold per category)
  - Missing documentation flags (unsigned notes, missing consults)
  - Medication safety flags (duplicate meds, high-risk combos)
  - Provider coverage gaps (dates with no attending documented)
  - Billing code validation (CPT vs service mismatch)

No API calls — pure rule-based logic. Runs in milliseconds.
Self-contained — no imports from rest of project.
"""
import json
import time
import re
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple


# ═══════════════════════════════════════════════════════
# AUDIT RULE CONFIGURATION
# ═══════════════════════════════════════════════════════

# High-value charge thresholds per category
HIGH_VALUE_THRESHOLDS = {
    "room": 6000.00,
    "surgery": 15000.00,
    "anesthesia": 3000.00,
    "ambulance": 3000.00,
    "imaging": 2000.00,
    "lab": 500.00,
    "medication": 500.00,
    "therapy": 1000.00,
    "supply": 1000.00,
    "professional_fee": 2000.00,
    "facility_fee": 5000.00,
    "emergency": 5000.00,
    "other": 2000.00,
}

# Expected daily charges for inpatient stay
EXPECTED_DAILY_MIN = 500.00
EXPECTED_DAILY_MAX = 30000.00

# Severity levels
class Severity:
    CRITICAL = "critical"    # Likely error or fraud indicator
    HIGH = "high"            # Significant concern, needs review
    MEDIUM = "medium"        # Worth investigating
    LOW = "low"              # Informational finding
    INFO = "info"            # Contextual note


# ═══════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════

class AuditFinding:
    """A single QA/audit finding."""
    def __init__(self, finding_id, rule_id, severity, category,
                 title, description, affected_date=None,
                 affected_items=None, recommendation=None,
                 financial_impact=None):
        self.finding_id = finding_id
        self.rule_id = rule_id
        self.severity = severity
        self.category = category
        self.title = title
        self.description = description
        self.affected_date = affected_date
        self.affected_items = affected_items or []
        self.recommendation = recommendation
        self.financial_impact = financial_impact

    def to_dict(self):
        return {
            "finding_id": self.finding_id,
            "rule_id": self.rule_id,
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "affected_date": self.affected_date,
            "affected_items": self.affected_items,
            "recommendation": self.recommendation,
            "financial_impact": self.financial_impact,
        }


class AuditReport:
    """Complete audit report for an episode of care."""
    def __init__(self):
        self.findings = []
        self.summary = {}
        self.risk_score = 0.0       # 0-100 overall risk
        self.total_financial_impact = 0.0
        self.rules_executed = 0
        self.processing_time_ms = 0

    def add(self, finding: AuditFinding):
        self.findings.append(finding)

    def to_dict(self):
        # Group by severity
        by_severity = defaultdict(list)
        for f in self.findings:
            by_severity[f.severity].append(f.to_dict())

        # Group by category
        by_category = defaultdict(list)
        for f in self.findings:
            by_category[f.category].append(f.to_dict())

        return {
            "agent": "agent4_qa",
            "version": "v1",
            "risk_score": round(self.risk_score, 1),
            "total_findings": len(self.findings),
            "total_financial_impact": self.total_financial_impact,
            "rules_executed": self.rules_executed,
            "processing_time_ms": self.processing_time_ms,
            "summary": {
                "critical": len(by_severity.get(Severity.CRITICAL, [])),
                "high": len(by_severity.get(Severity.HIGH, [])),
                "medium": len(by_severity.get(Severity.MEDIUM, [])),
                "low": len(by_severity.get(Severity.LOW, [])),
                "info": len(by_severity.get(Severity.INFO, [])),
            },
            "findings_by_severity": dict(by_severity),
            "findings_by_category": dict(by_category),
            "all_findings": [f.to_dict() for f in self.findings],
        }


# ═══════════════════════════════════════════════════════
# AUDIT RULES
# ═══════════════════════════════════════════════════════

def _next_id(counter: list) -> str:
    counter[0] += 1
    return f"QA_{counter[0]:04d}"


# ── Rule 1: Charges Without Clinical Support ──
def rule_charge_without_clinical(timeline: dict, report: AuditReport, counter: list):
    """Flag billing charges that have no linked clinical event."""
    for date_str, entry in timeline.get("timeline", {}).items():
        costs = entry.get("cost_items", [])
        events = entry.get("events", [])
        event_ids = {e.get("event_id") for e in events}

        for cost in costs:
            linked = cost.get("linked_event_id")
            if linked and linked not in event_ids:
                # Linked event doesn't exist on this date
                report.add(AuditFinding(
                    finding_id=_next_id(counter),
                    rule_id="R001",
                    severity=Severity.HIGH,
                    category="billing_integrity",
                    title="Charge references non-existent clinical event",
                    description=(
                        f"Cost '{cost.get('description', '')}' (${cost.get('amount', 0):,.2f}) "
                        f"links to event '{linked}' which was not found on {date_str}. "
                        f"The clinical documentation may be missing or the link is incorrect."
                    ),
                    affected_date=date_str,
                    affected_items=[cost.get("cost_id")],
                    recommendation="Verify clinical documentation exists for this charge.",
                    financial_impact=cost.get("amount"),
                ))

        # Check for costs with no linked event at all (on surgical/procedure dates)
        procedure_events = [e for e in events if e.get("event_type") in
                          ("procedure", "surgery", "imaging")]
        procedure_dates = bool(procedure_events)

        for cost in costs:
            if not cost.get("linked_event_id") and cost.get("category") in (
                "surgery", "anesthesia", "imaging", "procedure"
            ):
                amt = cost.get("amount", 0) or 0
                if amt > 500:
                    report.add(AuditFinding(
                        finding_id=_next_id(counter),
                        rule_id="R001b",
                        severity=Severity.MEDIUM,
                        category="billing_integrity",
                        title="High-value procedural charge without clinical link",
                        description=(
                            f"Cost '{cost.get('description', '')}' (${amt:,.2f}) "
                            f"on {date_str} has no linked clinical event. "
                            f"Category: {cost.get('category')}."
                        ),
                        affected_date=date_str,
                        affected_items=[cost.get("cost_id")],
                        recommendation="Link this charge to the corresponding clinical event.",
                        financial_impact=amt,
                    ))


# ── Rule 2: Clinical Events Without Charges ──
def rule_clinical_without_charge(timeline: dict, report: AuditReport, counter: list):
    """Flag documented procedures that have no corresponding charge."""
    billable_types = {"procedure", "imaging", "consultation"}

    for date_str, entry in timeline.get("timeline", {}).items():
        events = entry.get("events", [])
        costs = entry.get("cost_items", [])
        cost_descriptions = " ".join(
            (c.get("description", "") + " " + (c.get("code") or "")).lower()
            for c in costs
        )

        for evt in events:
            if evt.get("event_type") not in billable_types:
                continue
            # Check if any cost item seems to match this event
            evt_desc = evt.get("description", "").lower()
            cpt_codes = evt.get("cpt_codes", [])

            has_matching_cost = False
            # Check by CPT code
            for code in cpt_codes:
                if code in cost_descriptions:
                    has_matching_cost = True
                    break
            # Check by keyword overlap
            if not has_matching_cost:
                keywords = re.sub(r'[^a-z0-9\s]', '', evt_desc).split()
                significant = [w for w in keywords if len(w) > 3]
                for kw in significant[:3]:
                    if kw in cost_descriptions:
                        has_matching_cost = True
                        break

            if not has_matching_cost and cpt_codes:
                report.add(AuditFinding(
                    finding_id=_next_id(counter),
                    rule_id="R002",
                    severity=Severity.MEDIUM,
                    category="underbilling",
                    title="Documented procedure with no matching charge",
                    description=(
                        f"Event '{evt.get('description', '')[:60]}' "
                        f"(CPT: {', '.join(cpt_codes)}) on {date_str} "
                        f"has no corresponding billing charge."
                    ),
                    affected_date=date_str,
                    affected_items=[evt.get("event_id")],
                    recommendation="Verify if this procedure should be billed separately.",
                ))


# ── Rule 3: High-Value Charge Flags ──
def rule_high_value_charges(timeline: dict, report: AuditReport, counter: list):
    """Flag charges that exceed category thresholds."""
    for date_str, entry in timeline.get("timeline", {}).items():
        for cost in entry.get("cost_items", []):
            cat = cost.get("category", "other")
            amt = cost.get("amount") or 0
            threshold = HIGH_VALUE_THRESHOLDS.get(cat, 2000)

            if amt > threshold:
                report.add(AuditFinding(
                    finding_id=_next_id(counter),
                    rule_id="R003",
                    severity=Severity.LOW,
                    category="high_value",
                    title=f"High-value {cat} charge",
                    description=(
                        f"${amt:,.2f} for '{cost.get('description', '')}' on {date_str} "
                        f"exceeds the {cat} threshold of ${threshold:,.2f}."
                    ),
                    affected_date=date_str,
                    affected_items=[cost.get("cost_id")],
                    recommendation="Verify charge is appropriate for services rendered.",
                    financial_impact=amt,
                ))


# ── Rule 4: Duplicate/Similar Charges ──
def rule_duplicate_charges(timeline: dict, report: AuditReport, counter: list):
    """Flag potential duplicate charges on the same date."""
    for date_str, entry in timeline.get("timeline", {}).items():
        costs = entry.get("cost_items", [])
        seen = {}  # (category, amount, code) -> list of cost_ids

        for cost in costs:
            key = (
                cost.get("category"),
                cost.get("amount"),
                cost.get("code"),
            )
            if key[1] is None or key[1] == 0:
                continue
            if key not in seen:
                seen[key] = []
            seen[key].append(cost)

        for key, dupes in seen.items():
            if len(dupes) > 1:
                cat, amt, code = key
                descriptions = [d.get("description", "") for d in dupes]
                # Check if descriptions are actually different services
                unique_desc = set(d.lower()[:30] for d in descriptions)
                if len(unique_desc) == 1:
                    # Same description, same amount, same code = likely duplicate
                    report.add(AuditFinding(
                        finding_id=_next_id(counter),
                        rule_id="R004",
                        severity=Severity.HIGH,
                        category="duplicate_charges",
                        title=f"Potential duplicate {cat} charge",
                        description=(
                            f"{len(dupes)} identical charges of ${amt:,.2f} "
                            f"(code: {code or 'N/A'}) on {date_str}: "
                            f"'{descriptions[0][:50]}'"
                        ),
                        affected_date=date_str,
                        affected_items=[d.get("cost_id") for d in dupes],
                        recommendation="Verify these are distinct services, not duplicates.",
                        financial_impact=amt * (len(dupes) - 1),
                    ))


# ── Rule 5: Timeline Consistency ──
def rule_timeline_consistency(timeline: dict, report: AuditReport, counter: list):
    """Check for logical timeline issues."""
    dates = sorted(timeline.get("timeline", {}).keys())
    if not dates:
        return

    episode = timeline.get("episode", {})
    start = episode.get("start_date")
    end = episode.get("end_date")

    # Check for admission event
    first_entry = timeline["timeline"].get(dates[0], {})
    has_admission = any(
        e.get("event_type") == "admission"
        for e in first_entry.get("events", [])
    )
    if not has_admission:
        report.add(AuditFinding(
            finding_id=_next_id(counter),
            rule_id="R005a",
            severity=Severity.MEDIUM,
            category="timeline_integrity",
            title="No admission event on first date",
            description=(
                f"Episode starts on {dates[0]} but no admission event found. "
                f"Verify admission documentation is complete."
            ),
            affected_date=dates[0],
            recommendation="Ensure admission note and orders are documented.",
        ))

    # Check for discharge event
    last_entry = timeline["timeline"].get(dates[-1], {})
    has_discharge = any(
        e.get("event_type") == "discharge"
        for e in last_entry.get("events", [])
    )
    if not has_discharge and len(dates) > 1:
        report.add(AuditFinding(
            finding_id=_next_id(counter),
            rule_id="R005b",
            severity=Severity.MEDIUM,
            category="timeline_integrity",
            title="No discharge event on last date",
            description=(
                f"Episode ends on {dates[-1]} but no discharge event found. "
                f"Verify discharge documentation is complete."
            ),
            affected_date=dates[-1],
            recommendation="Ensure discharge summary is documented.",
        ))

    # Check for date gaps (missing days in inpatient stay)
    for i in range(len(dates) - 1):
        d1 = datetime.strptime(dates[i], "%Y-%m-%d")
        d2 = datetime.strptime(dates[i + 1], "%Y-%m-%d")
        gap = (d2 - d1).days
        if gap > 1:
            missing = [(d1 + timedelta(days=j)).strftime("%Y-%m-%d")
                      for j in range(1, gap)]
            report.add(AuditFinding(
                finding_id=_next_id(counter),
                rule_id="R005c",
                severity=Severity.HIGH,
                category="timeline_integrity",
                title=f"{gap - 1}-day gap in timeline",
                description=(
                    f"No data between {dates[i]} and {dates[i+1]}. "
                    f"Missing dates: {', '.join(missing)}. "
                    f"This may indicate missing documentation or unbilled days."
                ),
                affected_date=dates[i],
                recommendation="Obtain progress notes and charges for missing dates.",
                financial_impact=EXPECTED_DAILY_MIN * (gap - 1),
            ))

    # Check for charges on dates outside episode
    for date_str, entry in timeline.get("timeline", {}).items():
        if start and date_str < start:
            report.add(AuditFinding(
                finding_id=_next_id(counter),
                rule_id="R005d",
                severity=Severity.HIGH,
                category="timeline_integrity",
                title="Charges before admission date",
                description=(
                    f"Found {entry.get('cost_count', 0)} charges on {date_str}, "
                    f"which is before the episode start date of {start}."
                ),
                affected_date=date_str,
                recommendation="Verify admission date or pre-admission charges.",
                financial_impact=entry.get("total_cost", 0),
            ))


# ── Rule 6: Daily Cost Anomalies ──
def rule_daily_cost_anomalies(timeline: dict, report: AuditReport, counter: list):
    """Flag dates with abnormally high or low costs."""
    daily_costs = []
    for date_str, entry in timeline.get("timeline", {}).items():
        tc = entry.get("total_cost", 0)
        daily_costs.append((date_str, tc))

    if len(daily_costs) < 2:
        return

    # Calculate mean and flag outliers
    costs = [c for _, c in daily_costs if c > 0]
    if not costs:
        return
    avg_cost = sum(costs) / len(costs)

    for date_str, tc in daily_costs:
        if tc > avg_cost * 3 and tc > 10000:
            report.add(AuditFinding(
                finding_id=_next_id(counter),
                rule_id="R006",
                severity=Severity.LOW,
                category="cost_anomaly",
                title=f"High-cost day: ${tc:,.2f}",
                description=(
                    f"{date_str} has charges of ${tc:,.2f}, which is "
                    f"{tc/avg_cost:.1f}x the average daily cost of ${avg_cost:,.2f}. "
                    f"This may be expected (e.g., surgery day) or may warrant review."
                ),
                affected_date=date_str,
                recommendation="Verify all charges are supported by documentation.",
                financial_impact=tc,
            ))


# ── Rule 7: Provider Coverage Gaps ──
def rule_provider_coverage(timeline: dict, report: AuditReport, counter: list):
    """Check for dates without documented attending provider."""
    for date_str, entry in timeline.get("timeline", {}).items():
        events = entry.get("events", [])
        providers_on_date = set()
        for evt in events:
            for p in evt.get("providers", []):
                if p:
                    providers_on_date.add(p)

        if not providers_on_date:
            report.add(AuditFinding(
                finding_id=_next_id(counter),
                rule_id="R007",
                severity=Severity.MEDIUM,
                category="documentation",
                title="No provider documented for date",
                description=(
                    f"No attending/treating provider documented in events for {date_str}. "
                    f"All inpatient days should have an identifiable attending physician."
                ),
                affected_date=date_str,
                recommendation="Verify progress note with provider signature exists.",
            ))


# ── Rule 8: Medication Safety Flags ──
def rule_medication_safety(timeline: dict, report: AuditReport, counter: list):
    """Flag potential medication concerns."""
    medications = timeline.get("medications", {})

    # Check for opioid combinations
    opioid_names = {"fentanyl", "morphine", "oxycodone", "hydrocodone",
                    "hydromorphone", "methadone", "oxymorphone", "tramadol"}
    active_opioids = []
    for mkey, med in medications.items():
        name = (med.get("name") or "").lower()
        if any(op in name for op in opioid_names):
            active_opioids.append(med)

    if len(active_opioids) > 2:
        names = [m.get("name") for m in active_opioids]
        report.add(AuditFinding(
            finding_id=_next_id(counter),
            rule_id="R008a",
            severity=Severity.MEDIUM,
            category="medication_safety",
            title=f"Multiple opioids prescribed ({len(active_opioids)})",
            description=(
                f"Patient has {len(active_opioids)} opioid medications documented: "
                f"{', '.join(names)}. While this may be appropriate in a trauma setting "
                f"with transitions from IV to PO, verify no overlap in active orders."
            ),
            recommendation="Review medication reconciliation for appropriate opioid transitions.",
        ))

    # Check for high-risk drug interactions
    med_names = [(med.get("name") or "").lower() for med in medications.values()]

    # Anticoagulant + NSAID
    has_anticoag = any("enoxaparin" in n or "heparin" in n or "warfarin" in n
                      for n in med_names)
    has_nsaid = any("ibuprofen" in n or "naproxen" in n or "ketorolac" in n
                   for n in med_names)
    if has_anticoag and has_nsaid:
        report.add(AuditFinding(
            finding_id=_next_id(counter),
            rule_id="R008b",
            severity=Severity.HIGH,
            category="medication_safety",
            title="Anticoagulant + NSAID combination",
            description=(
                "Patient has both an anticoagulant and an NSAID documented. "
                "This combination increases bleeding risk significantly."
            ),
            recommendation="Verify clinical appropriateness or if one has been discontinued.",
        ))


# ── Rule 9: Billing Code Validation ──
def rule_billing_codes(timeline: dict, report: AuditReport, counter: list):
    """Validate billing codes against service categories."""
    # Common CPT range checks
    cpt_ranges = {
        "emergency": (99281, 99285),
        "surgery": (10000, 69999),
        "anesthesia": (100, 1999),
        "radiology": (70010, 79999),
        "lab": (80000, 89999),
        "evaluation": (99200, 99499),
    }

    for date_str, entry in timeline.get("timeline", {}).items():
        for cost in entry.get("cost_items", []):
            code = cost.get("code")
            cat = cost.get("category")
            if not code or not cat:
                continue

            # Try to parse as numeric CPT
            code_num = None
            code_clean = re.sub(r'[^0-9]', '', code)
            if len(code_clean) == 5:
                try:
                    code_num = int(code_clean)
                except ValueError:
                    pass

            if code_num:
                # Check if code matches expected range for category
                expected = cpt_ranges.get(cat)
                if expected:
                    low, high = expected
                    if code_num < low or code_num > high:
                        report.add(AuditFinding(
                            finding_id=_next_id(counter),
                            rule_id="R009",
                            severity=Severity.LOW,
                            category="code_validation",
                            title=f"CPT code may not match category",
                            description=(
                                f"Code {code} on {date_str} is categorized as '{cat}' "
                                f"but falls outside expected CPT range "
                                f"({low}-{high}) for that category. "
                                f"Charge: '{cost.get('description', '')[:40]}'"
                            ),
                            affected_date=date_str,
                            affected_items=[cost.get("cost_id")],
                            recommendation="Verify code-to-category mapping.",
                        ))


# ── Rule 10: Episode-Level Checks ──
def rule_episode_checks(timeline: dict, report: AuditReport, counter: list):
    """Episode-level validation."""
    episode = timeline.get("episode", {})
    total_cost = episode.get("total_cost", 0)
    duration = episode.get("duration_days")

    if duration and duration > 0:
        avg_daily = total_cost / (duration + 1)  # +1 for day of admission
        if avg_daily > 25000:
            report.add(AuditFinding(
                finding_id=_next_id(counter),
                rule_id="R010a",
                severity=Severity.LOW,
                category="episode_review",
                title=f"High average daily cost: ${avg_daily:,.2f}",
                description=(
                    f"Total episode cost ${total_cost:,.2f} over {duration + 1} days "
                    f"= ${avg_daily:,.2f}/day average. This is above typical inpatient rates "
                    f"and may warrant itemized review."
                ),
                recommendation="Review for bundled charges or facility fee issues.",
                financial_impact=total_cost,
            ))

    # Check for missing provider signatures
    providers = timeline.get("providers", {})
    unsigned = [
        p.get("name") for p in providers.values()
        if not p.get("signature_detected") and
        any(r in (p.get("roles") or []) for r in
            ["treating_provider", "signing_provider", "supervising_provider"])
    ]
    if unsigned:
        report.add(AuditFinding(
            finding_id=_next_id(counter),
            rule_id="R010b",
            severity=Severity.MEDIUM,
            category="documentation",
            title=f"{len(unsigned)} provider(s) without detected signature",
            description=(
                f"The following treating/signing providers have no detected signature: "
                f"{', '.join(unsigned)}. Unsigned notes may not be billable."
            ),
            recommendation="Verify all clinical notes have appropriate provider signatures.",
        ))


# ═══════════════════════════════════════════════════════
# RISK SCORE CALCULATION
# ═══════════════════════════════════════════════════════

def calculate_risk_score(report: AuditReport) -> float:
    """Calculate overall risk score (0-100) from findings."""
    weights = {
        Severity.CRITICAL: 25,
        Severity.HIGH: 10,
        Severity.MEDIUM: 5,
        Severity.LOW: 2,
        Severity.INFO: 0,
    }

    score = 0
    for f in report.findings:
        score += weights.get(f.severity, 0)

    # Cap at 100
    return min(score, 100.0)


# ═══════════════════════════════════════════════════════
# MAIN AUDIT PIPELINE
# ═══════════════════════════════════════════════════════

ALL_RULES = [
    ("R001: Charge Without Clinical Support", rule_charge_without_clinical),
    ("R002: Clinical Without Charge", rule_clinical_without_charge),
    ("R003: High-Value Charges", rule_high_value_charges),
    ("R004: Duplicate Charges", rule_duplicate_charges),
    ("R005: Timeline Consistency", rule_timeline_consistency),
    ("R006: Daily Cost Anomalies", rule_daily_cost_anomalies),
    ("R007: Provider Coverage", rule_provider_coverage),
    ("R008: Medication Safety", rule_medication_safety),
    ("R009: Billing Codes", rule_billing_codes),
    ("R010: Episode Checks", rule_episode_checks),
]


def run_audit(timeline_data: dict) -> AuditReport:
    """Run all audit rules against a timeline."""
    t0 = time.time()
    report = AuditReport()
    counter = [0]

    for rule_name, rule_func in ALL_RULES:
        try:
            rule_func(timeline_data, report, counter)
            report.rules_executed += 1
        except Exception as e:
            report.add(AuditFinding(
                finding_id=_next_id(counter),
                rule_id="ERROR",
                severity=Severity.INFO,
                category="system",
                title=f"Rule execution error: {rule_name}",
                description=str(e),
            ))

    # Calculate totals
    report.risk_score = calculate_risk_score(report)
    report.total_financial_impact = sum(
        f.financial_impact for f in report.findings
        if f.financial_impact is not None
    )
    report.processing_time_ms = int((time.time() - t0) * 1000)

    return report


# ═══════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # ── 1. Locate Agent 3 timeline ──
    results_path = None
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
    else:
        for candidate in [
            Path("agent3_timeline.json"),
            Path("chronocare/agent3_timeline.json"),
        ]:
            if candidate.exists():
                results_path = candidate
                break

    if not results_path or not results_path.exists():
        print("Usage: python agent4_qa.py [agent3_timeline.json]")
        sys.exit(1)

    with open(results_path) as f:
        timeline_data = json.load(f)

    print("=" * 70)
    print("  ChronoCare AI — Agent 4: QA & Anomaly Detector")
    print(f"  Input: {results_path}")
    print("=" * 70)

    # Run audit
    report = run_audit(timeline_data)
    result = report.to_dict()

    # ── Display ──
    severity_icons = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🔵",
        "info": "⚪",
    }

    risk = result["risk_score"]
    if risk >= 70:
        risk_label = "🔴 HIGH RISK"
    elif risk >= 40:
        risk_label = "🟠 MODERATE RISK"
    elif risk >= 15:
        risk_label = "🟡 LOW RISK"
    else:
        risk_label = "🟢 MINIMAL RISK"

    print(f"\n📊 RISK SCORE: {risk}/100 — {risk_label}")
    print(f"💰 FINANCIAL IMPACT: ${result['total_financial_impact']:,.2f}")
    print(f"📋 {result['total_findings']} findings from {result['rules_executed']} rules")
    print(f"⏱️  {result['processing_time_ms']}ms")

    # Summary
    print(f"\n{'─' * 70}")
    print("  FINDING SUMMARY")
    print(f"{'─' * 70}")
    summary = result["summary"]
    print(f"  🔴 Critical: {summary['critical']}")
    print(f"  🟠 High:     {summary['high']}")
    print(f"  🟡 Medium:   {summary['medium']}")
    print(f"  🔵 Low:      {summary['low']}")
    print(f"  ⚪ Info:     {summary['info']}")

    # All findings grouped by severity
    for severity in ["critical", "high", "medium", "low", "info"]:
        findings = result["findings_by_severity"].get(severity, [])
        if not findings:
            continue

        icon = severity_icons.get(severity, "⚪")
        print(f"\n{'─' * 70}")
        print(f"  {icon} {severity.upper()} FINDINGS ({len(findings)})")
        print(f"{'─' * 70}")

        for f in findings:
            date_str = f" [{f['affected_date']}]" if f.get("affected_date") else ""
            impact = f" (${f['financial_impact']:,.2f})" if f.get("financial_impact") else ""
            print(f"\n  {icon} {f['rule_id']}: {f['title']}{date_str}{impact}")
            print(f"     {f['description'][:120]}")
            if f.get("recommendation"):
                print(f"     💡 {f['recommendation']}")

    # Category summary
    print(f"\n{'─' * 70}")
    print("  FINDINGS BY CATEGORY")
    print(f"{'─' * 70}")
    for cat, findings in sorted(result["findings_by_category"].items()):
        total_impact = sum(
            f.get("financial_impact", 0) or 0 for f in findings
        )
        impact_str = f" — ${total_impact:,.2f}" if total_impact > 0 else ""
        print(f"  {cat}: {len(findings)} findings{impact_str}")

    # Save
    output_path = results_path.parent / "agent4_qa_report.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n  Saved: {output_path}")
    print("  ✅ Done!")
