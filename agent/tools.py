"""
agent/tools.py
--------------
The four core tools of NetGuardAgent:
  Tool 1 — Log Analyzer
  Tool 2 — Threat Classifier
  Tool 3 — MITRE ATT&CK RAG
  Tool 4 — Report Generator
"""

import os
import json
import pandas as pd
from langchain_groq import ChatGroq
from agent.mitre_rag import get_rag

# ── Feature columns used from CICIDS-2017 ─────────────────────────────────────
FEATURE_COLS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "SYN Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
    "URG Flag Count",
]

ATTACK_CATEGORIES = [
    "BENIGN",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Heartbleed",
    "DDoS",
    "PortScan",
    "FTP-Patator",
    "SSH-Patator",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - Sql Injection",
    "Infiltration",
    "Bot",
]

SEVERITY_MAP = {
    "BENIGN": "None",
    "DoS Hulk": "High",
    "DoS GoldenEye": "High",
    "DoS slowloris": "Medium",
    "DoS Slowhttptest": "Medium",
    "Heartbleed": "Critical",
    "DDoS": "Critical",
    "PortScan": "Medium",
    "FTP-Patator": "High",
    "SSH-Patator": "High",
    "Web Attack - Brute Force": "High",
    "Web Attack - XSS": "High",
    "Web Attack - Sql Injection": "Critical",
    "Infiltration": "Critical",
    "Bot": "High",
}


def get_llm():
    """Returns a configured Groq LLM instance."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "and add it to your .env file."
        )
    return ChatGroq(
        model="llama3-8b-8192",
        temperature=0.1,
        max_tokens=1500,
        groq_api_key=api_key,
    )


# ── Tool 1: Log Analyzer ──────────────────────────────────────────────────────

def tool_log_analyzer(row: pd.Series) -> dict:
    """
    Parses a raw CICIDS-2017 network flow record into a structured
    natural language description and key statistics.

    Returns:
        dict with keys: 'text' (NL description), 'stats' (dict of values),
        'flags' (list of anomaly flags detected)
    """
    available = [c for c in FEATURE_COLS if c in row.index]
    stats = {}
    for col in available:
        val = row[col]
        try:
            stats[col] = float(val) if not pd.isna(val) else 0.0
        except Exception:
            stats[col] = 0.0

    # Build natural language description
    lines = ["=== Network Flow Analysis ==="]

    if "Flow Duration" in stats:
        dur_ms = stats["Flow Duration"] / 1000
        lines.append(f"Flow Duration: {dur_ms:.2f} ms")
    if "Total Fwd Packets" in stats and "Total Backward Packets" in stats:
        fwd = int(stats["Total Fwd Packets"])
        bwd = int(stats["Total Backward Packets"])
        ratio = fwd / max(bwd, 1)
        lines.append(f"Packets — Forward: {fwd}, Backward: {bwd}, Ratio: {ratio:.2f}")
    if "Total Length of Fwd Packets" in stats:
        lines.append(f"Total Fwd Bytes: {stats['Total Length of Fwd Packets']:.0f}")
    if "Flow Bytes/s" in stats:
        bps = stats["Flow Bytes/s"]
        lines.append(f"Flow Bytes/s: {bps:.2f}")
    if "Flow Packets/s" in stats:
        pps = stats["Flow Packets/s"]
        lines.append(f"Flow Packets/s: {pps:.2f}")
    if "Flow IAT Mean" in stats:
        lines.append(f"Mean Inter-Arrival Time: {stats['Flow IAT Mean']:.2f} µs")
    if "Fwd Packet Length Mean" in stats:
        lines.append(f"Mean Fwd Packet Length: {stats['Fwd Packet Length Mean']:.2f} bytes")

    # TCP flags
    flag_values = {
        "SYN": stats.get("SYN Flag Count", 0),
        "ACK": stats.get("ACK Flag Count", 0),
        "PSH": stats.get("PSH Flag Count", 0),
        "FIN": stats.get("FIN Flag Count", 0),
        "RST": stats.get("RST Flag Count", 0),
    }
    flag_str = ", ".join(f"{k}={int(v)}" for k, v in flag_values.items() if v > 0)
    if flag_str:
        lines.append(f"TCP Flags: {flag_str}")

    # Anomaly detection heuristics
    flags = []
    bps = stats.get("Flow Bytes/s", 0)
    pps = stats.get("Flow Packets/s", 0)
    syn = stats.get("SYN Flag Count", 0)
    fwd = stats.get("Total Fwd Packets", 0)
    bwd = stats.get("Total Backward Packets", 0)
    iat = stats.get("Flow IAT Mean", 999999)
    dur = stats.get("Flow Duration", 999999)

    if bps > 1e8:
        flags.append("EXTREME_THROUGHPUT: Possible flood attack (>100 MB/s)")
    elif bps > 1e6:
        flags.append("HIGH_THROUGHPUT: Above normal baseline (>1 MB/s)")
    if syn > 100:
        flags.append(f"HIGH_SYN: Abnormal SYN count ({int(syn)}) — possible SYN flood or scan")
    if fwd > 0 and bwd == 0:
        flags.append("ONE_DIRECTIONAL: No backward traffic — possible scan or DoS")
    if iat < 10 and pps > 1000:
        flags.append("RAPID_FIRE: Very short inter-arrival time with high packet rate")
    if dur < 1000 and fwd > 500:
        flags.append("BURST: Short flow with very high packet count")

    if flags:
        lines.append("\nAnomaly Indicators:")
        for f in flags:
            lines.append(f"  ⚠ {f}")

    return {
        "text": "\n".join(lines),
        "stats": stats,
        "flags": flags,
    }


# ── Tool 2: Threat Classifier ─────────────────────────────────────────────────

def tool_threat_classifier(parsed_log_text: str) -> dict:
    """
    Uses Llama 3 via Groq to classify the network flow as an attack type
    or benign. Returns label, confidence, and justification.
    """
    llm = get_llm()
    categories_str = "\n".join(f"  - {c}" for c in ATTACK_CATEGORIES)

    prompt = f"""You are a senior network security analyst. Analyze this network traffic flow and classify it.

{parsed_log_text}

Based on the traffic characteristics, classify this flow as EXACTLY ONE of these categories:
{categories_str}

Rules:
- If throughput is extremely high (>100MB/s) with one-directional traffic, consider DoS/DDoS
- If SYN count is very high with many unique destination ports, consider PortScan
- If many repeated login attempts are seen on ports 21/22, consider FTP-Patator or SSH-Patator
- If traffic appears completely normal with balanced flow, classify as BENIGN

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{"label": "<exact category from list>", "confidence": "<High|Medium|Low>", "reason": "<1-2 sentences explaining key indicators>"}}"""

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        # Strip any markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        # Validate label
        if result.get("label") not in ATTACK_CATEGORIES:
            result["label"] = "BENIGN"
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        match = re.search(r'\{.*?\}', response.content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except Exception:
                result = {"label": "BENIGN", "confidence": "Low", "reason": "Could not parse LLM response."}
        else:
            result = {"label": "BENIGN", "confidence": "Low", "reason": "LLM response parsing failed."}
    except Exception as e:
        result = {"label": "BENIGN", "confidence": "Low", "reason": f"Classification error: {str(e)}"}

    result["severity"] = SEVERITY_MAP.get(result.get("label", "BENIGN"), "Unknown")
    return result


# ── Tool 3: MITRE ATT&CK RAG ─────────────────────────────────────────────────

def tool_mitre_rag(attack_label: str) -> list[dict]:
    """
    Retrieves top-3 MITRE ATT&CK techniques for the given attack label.
    Returns a list of technique dicts with id, name, tactic, description,
    detection, remediation, and similarity_score.
    """
    rag = get_rag()
    return rag.retrieve(attack_label, top_k=3)


# ── Tool 4: Report Generator ──────────────────────────────────────────────────

def tool_report_generator(
    parsed_log_text: str,
    classification: dict,
    mitre_techniques: list[dict],
) -> dict:
    """
    Uses Llama 3 to synthesize the log, classification, and MITRE TTPs
    into a structured incident report.

    Returns dict with keys: summary, behavior, mitre_mapping,
    recommended_actions, full_report
    """
    llm = get_llm()
    label = classification.get("label", "BENIGN")
    severity = classification.get("severity", "Unknown")
    confidence = classification.get("confidence", "Unknown")
    reason = classification.get("reason", "")

    if mitre_techniques:
        mitre_text = "\n".join(
            f"- {t['name']} ({t['id']}) | Tactic: {t['tactic']}\n"
            f"  Description: {t['description'][:200]}..."
            for t in mitre_techniques
        )
    else:
        mitre_text = "N/A — Traffic classified as benign."

    prompt = f"""You are a senior SOC analyst. Write a formal incident report for this detected network event.

NETWORK TRAFFIC ANALYSIS:
{parsed_log_text}

CLASSIFICATION:
- Attack Type: {label}
- Severity: {severity}
- Confidence: {confidence}
- Analyst Note: {reason}

MITRE ATT&CK CONTEXT:
{mitre_text}

Write a structured report with EXACTLY these four sections. Be specific and professional.

## 1. Attack Summary
[One paragraph: attack type, severity, what it means for the organization]

## 2. Observed Behavior
[One paragraph: specific traffic anomalies from the flow data above, what they indicate]

## 3. MITRE ATT&CK Mapping
[List each relevant technique with its ID, tactic, and 1-sentence relevance explanation]

## 4. Recommended Actions
[Numbered list of 5-7 specific, actionable response steps. Be concrete — include specific thresholds, tools, or commands where appropriate]

Keep the total report under 500 words. Use professional security analyst language."""

    try:
        response = llm.invoke(prompt)
        full_report = response.content.strip()
    except Exception as e:
        full_report = f"Report generation failed: {str(e)}"

    # Parse sections for structured display
    sections = {"summary": "", "behavior": "", "mitre_mapping": "", "recommended_actions": ""}
    current = None
    lines = full_report.split("\n")
    for line in lines:
        if "## 1." in line:
            current = "summary"
        elif "## 2." in line:
            current = "behavior"
        elif "## 3." in line:
            current = "mitre_mapping"
        elif "## 4." in line:
            current = "recommended_actions"
        elif current:
            sections[current] += line + "\n"

    return {
        "full_report": full_report,
        "summary": sections["summary"].strip(),
        "behavior": sections["behavior"].strip(),
        "mitre_mapping": sections["mitre_mapping"].strip(),
        "recommended_actions": sections["recommended_actions"].strip(),
    }
