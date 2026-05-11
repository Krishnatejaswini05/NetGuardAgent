"""
agent/tools.py
--------------
The four core tools of NetGuardAgent:
  Tool 1 — Log Analyzer       (rule-based heuristic pre-classification)
  Tool 2 — Threat Classifier  (LLM with strong attack-biased prompting)
  Tool 3 — MITRE ATT&CK RAG  (FAISS semantic retrieval)
  Tool 4 — Report Generator   (structured 4-section incident report)
"""

import os
import json
import re
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from agent.mitre_rag import get_rag

FEATURE_COLS = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Fwd IAT Mean", "Bwd IAT Mean", "SYN Flag Count", "ACK Flag Count",
    "PSH Flag Count", "FIN Flag Count", "RST Flag Count", "URG Flag Count",
]

ATTACK_CATEGORIES = [
    "BENIGN", "DoS Hulk", "DoS GoldenEye", "DoS slowloris",
    "DoS Slowhttptest", "Heartbleed", "DDoS", "PortScan",
    "FTP-Patator", "SSH-Patator", "Web Attack - Brute Force",
    "Web Attack - XSS", "Web Attack - Sql Injection", "Infiltration", "Bot",
]

SEVERITY_MAP = {
    "BENIGN": "None", "DoS Hulk": "High", "DoS GoldenEye": "High",
    "DoS slowloris": "Medium", "DoS Slowhttptest": "Medium",
    "Heartbleed": "Critical", "DDoS": "Critical", "PortScan": "Medium",
    "FTP-Patator": "High", "SSH-Patator": "High",
    "Web Attack - Brute Force": "High", "Web Attack - XSS": "High",
    "Web Attack - Sql Injection": "Critical", "Infiltration": "Critical",
    "Bot": "High",
}


def get_llm():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.0,
                    max_tokens=2000, groq_api_key=api_key)


def rule_based_hint(stats: dict) -> str:
    bps      = stats.get("Flow Bytes/s", 0)
    pps      = stats.get("Flow Packets/s", 0)
    fwd_pkts = stats.get("Total Fwd Packets", 0)
    bwd_pkts = stats.get("Total Backward Packets", 0)
    syn      = stats.get("SYN Flag Count", 0)
    fin      = stats.get("FIN Flag Count", 0)
    rst      = stats.get("RST Flag Count", 0)
    iat_mean = stats.get("Flow IAT Mean", 999999)
    fwd_len  = stats.get("Fwd Packet Length Mean", 0)
    duration = stats.get("Flow Duration", 999999)
    total_fwd_bytes = stats.get("Total Length of Fwd Packets", 0)

    if bps > 500000 and fwd_pkts > 100 and iat_mean < 500 and bwd_pkts < fwd_pkts * 0.1:
        return "DoS Hulk"
    if bps > 2000000 and syn > 50:
        return "DDoS"
    if pps > 1000 and duration < 10000 and fwd_pkts > 50 and bwd_pkts < 10:
        return "DoS GoldenEye"
    if duration > 1000000 and fwd_pkts < 20 and bps < 1000 and fin == 0:
        return "DoS slowloris"
    if duration > 500000 and fwd_pkts < 30 and bps < 5000 and rst > 0:
        return "DoS Slowhttptest"
    if syn > 20 and bwd_pkts == 0 and duration < 5000 and fwd_pkts < 5:
        return "PortScan"
    if fwd_len > 10000 and fwd_pkts < 20 and total_fwd_bytes > 50000:
        return "Heartbleed"
    if fwd_pkts > 10 and bwd_pkts > 5 and fwd_len < 100 and bps < 50000 and duration < 100000:
        return "FTP-Patator"
    if bps > 200000 or pps > 500 or syn > 10:
        return "UNKNOWN"
    if bps < 100000 and pps < 200 and bwd_pkts > 0 and iat_mean > 1000:
        return "BENIGN"
    return "UNKNOWN"


def tool_log_analyzer(row: pd.Series) -> dict:
    available = [c for c in FEATURE_COLS if c in row.index]
    stats = {}
    for col in available:
        try:
            val = float(row[col])
            stats[col] = 0.0 if (np.isnan(val) or np.isinf(val)) else val
        except Exception:
            stats[col] = 0.0

    dur          = stats.get("Flow Duration", 0)
    fwd          = stats.get("Total Fwd Packets", 0)
    bwd          = stats.get("Total Backward Packets", 0)
    ratio        = fwd / max(bwd, 1)
    fwd_bytes    = stats.get("Total Length of Fwd Packets", 0)
    bwd_bytes    = stats.get("Total Length of Bwd Packets", 0)
    bps          = stats.get("Flow Bytes/s", 0)
    pps          = stats.get("Flow Packets/s", 0)
    iat          = stats.get("Flow IAT Mean", 0)
    iat_std      = stats.get("Flow IAT Std", 0)
    fwd_len_mean = stats.get("Fwd Packet Length Mean", 0)
    fwd_len_max  = stats.get("Fwd Packet Length Max", 0)
    bwd_len_mean = stats.get("Bwd Packet Length Mean", 0)
    syn = stats.get("SYN Flag Count", 0)
    ack = stats.get("ACK Flag Count", 0)
    psh = stats.get("PSH Flag Count", 0)
    fin = stats.get("FIN Flag Count", 0)
    rst = stats.get("RST Flag Count", 0)

    lines = [
        "=== NETWORK FLOW ANALYSIS REPORT ===",
        f"Flow Duration      : {dur:.0f} µs ({dur/1000:.2f} ms)",
        f"Packets Fwd/Bwd    : {int(fwd)} / {int(bwd)}  (ratio {ratio:.2f})",
        f"Bytes  Fwd/Bwd     : {fwd_bytes:.0f} / {bwd_bytes:.0f}",
        f"Throughput         : {bps:.2f} bytes/s  |  {pps:.2f} packets/s",
        f"Inter-Arrival Time : mean={iat:.2f} µs  std={iat_std:.2f}",
        f"Fwd Pkt Length     : mean={fwd_len_mean:.2f}  max={fwd_len_max:.2f}",
        f"Bwd Pkt Length     : mean={bwd_len_mean:.2f}",
        f"TCP Flags          : SYN={int(syn)} ACK={int(ack)} PSH={int(psh)} FIN={int(fin)} RST={int(rst)}",
    ]

    flags = []
    if bps > 2_000_000:
        flags.append(f"EXTREME THROUGHPUT: {bps:.0f} bytes/s — DDoS/DoS flood indicator")
    elif bps > 500_000:
        flags.append(f"HIGH THROUGHPUT: {bps:.0f} bytes/s — above normal baseline")
    if pps > 5000:
        flags.append(f"EXTREME PACKET RATE: {pps:.0f} pps — DoS attack indicator")
    elif pps > 1000:
        flags.append(f"HIGH PACKET RATE: {pps:.0f} pps")
    if syn > 50:
        flags.append(f"EXCESSIVE SYN FLAGS: {int(syn)} — SYN flood or port scan")
    if bwd == 0 and fwd > 20:
        flags.append("UNIDIRECTIONAL FLOW: Zero backward packets — DoS or scan likely")
    if iat < 100 and pps > 500:
        flags.append(f"RAPID-FIRE: IAT={iat:.1f}µs with {pps:.0f} pps — automated attack pattern")
    if dur > 1_000_000 and fwd < 20:
        flags.append(f"SLOW ATTACK: Long duration ({dur:.0f}µs) few packets — possible slowloris")
    if fwd_len_mean > 10000:
        flags.append(f"OVERSIZED PACKETS: mean fwd length {fwd_len_mean:.0f} bytes — possible Heartbleed")
    if ratio > 50:
        flags.append(f"EXTREME FWD/BWD RATIO: {ratio:.0f} — heavily unidirectional")

    if flags:
        lines.append("\n--- ANOMALY INDICATORS DETECTED ---")
        for f in flags:
            lines.append(f"  ⚠ {f}")
    else:
        lines.append("\n--- No strong anomaly indicators detected ---")

    hint = rule_based_hint(stats)
    lines.append(f"\n--- RULE-BASED PRE-CLASSIFICATION: {hint} ---")

    return {"text": "\n".join(lines), "stats": stats, "flags": flags, "hint": hint}


def tool_threat_classifier(parsed_log_text: str, hint: str = "UNKNOWN") -> dict:
    llm = get_llm()

    if hint not in ("UNKNOWN", "BENIGN"):
        hint_instruction = (
            f"IMPORTANT: Rule-based analysis STRONGLY indicates this is a {hint} attack. "
            f"Only classify as BENIGN if you have overwhelming evidence of normal traffic."
        )
    elif hint == "BENIGN":
        hint_instruction = (
            "Rule-based analysis suggests BENIGN traffic. "
            "Confirm only if all indicators show normal bidirectional communication."
        )
    else:
        hint_instruction = (
            "Rule-based analysis was inconclusive. "
            "Do NOT default to BENIGN if any anomaly flags are present."
        )

    categories_str = "\n".join(f"  - {c}" for c in ATTACK_CATEGORIES)

    prompt = f"""You are an expert network intrusion detection system analyzing CICIDS-2017 traffic.

NETWORK FLOW DATA:
{parsed_log_text}

{hint_instruction}

CLASSIFICATION DECISION RULES:
- DoS Hulk        : bytes/s >500K, many fwd pkts, tiny IAT (<500µs), almost no backward traffic
- DoS GoldenEye   : pps >1000, short duration, HTTP flood, few backward pkts
- DoS slowloris   : very long duration (>1M µs), very few pkts (<20), low throughput, no FIN
- DoS Slowhttptest: long duration, few pkts, low throughput, some RST flags
- DDoS            : bytes/s >2MB, high SYN count (>50)
- PortScan        : high SYN (>20), ZERO backward pkts, very short flows
- Heartbleed      : oversized fwd packets (mean >10KB), small packet count
- FTP/SSH Patator : many small packets, repeated short flows, brute-force pattern
- BENIGN          : ONLY if bidirectional, low rates (<100KB/s), normal IAT (>1000µs), NO anomaly flags

CATEGORIES:
{categories_str}

Respond ONLY with valid JSON — no markdown, no text outside the JSON:
{{"label": "<exact category from list above>", "confidence": "<High|Medium|Low>", "reason": "<2-3 sentences citing specific numeric values from the flow data>"}}"""

    try:
        response = llm.invoke(prompt)
        text = re.sub(r"```json|```", "", response.content.strip()).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        result = json.loads(match.group() if match else text)
        if result.get("label") not in ATTACK_CATEGORIES:
            result["label"] = hint if hint != "UNKNOWN" else "BENIGN"
            result["confidence"] = "Low"
    except Exception as e:
        fallback = hint if hint != "UNKNOWN" else "BENIGN"
        result = {
            "label": fallback,
            "confidence": "Medium" if hint not in ("UNKNOWN", "BENIGN") else "Low",
            "reason": f"Rule-based classification used (LLM error): {str(e)[:100]}"
        }

    result["severity"] = SEVERITY_MAP.get(result.get("label", "BENIGN"), "Unknown")
    return result


def tool_mitre_rag(attack_label: str) -> list:
    rag = get_rag()
    return rag.retrieve(attack_label, top_k=3)


def tool_report_generator(parsed_log_text: str, classification: dict, mitre_techniques: list) -> dict:
    label      = classification.get("label", "BENIGN")
    severity   = classification.get("severity", "Unknown")
    confidence = classification.get("confidence", "Unknown")
    reason     = classification.get("reason", "")

    if label == "BENIGN":
        return {
            "full_report": (
                "## 1. Attack Summary\nNo attack detected. Traffic classified as BENIGN.\n\n"
                "## 2. Observed Behavior\nNormal bidirectional communication with no anomalies.\n\n"
                "## 3. MITRE ATT&CK Mapping\nNo malicious techniques identified.\n\n"
                "## 4. Recommended Actions\n1. No immediate action required.\n"
                "2. Continue routine network monitoring.\n3. Retain flow logs for baseline analysis."
            ),
            "summary": "No attack detected. Traffic classified as BENIGN.",
            "behavior": "Normal bidirectional communication with no anomalies.",
            "mitre_mapping": "No malicious techniques identified.",
            "recommended_actions": "1. No action required.\n2. Continue monitoring.",
        }

    llm = get_llm()

    if mitre_techniques:
        mitre_text = "\n".join(
            f"- {t['name']} ({t['id']}) | Tactic: {t['tactic']}\n  {t['description'][:200]}..."
            for t in mitre_techniques
        )
        remediation_list = []
        for t in mitre_techniques:
            remediation_list.extend(t.get("remediation", []))
        remediation_text = "\n".join(f"- {r}" for r in remediation_list[:8])
    else:
        mitre_text = "No specific techniques retrieved."
        remediation_text = "Follow general incident response procedures."

    prompt = f"""You are a senior SOC analyst. Write a detailed professional incident report for this CONFIRMED attack.
This is NOT benign traffic. Write with urgency and precision.

=== CONFIRMED ATTACK ===
Type      : {label}
Severity  : {severity}
Confidence: {confidence}
Evidence  : {reason}

=== FLOW EVIDENCE ===
{parsed_log_text}

=== MITRE ATT&CK INTELLIGENCE ===
{mitre_text}

=== KNOWLEDGE BASE REMEDIATIONS ===
{remediation_text}

Write ALL FOUR sections. Be specific. Use exact numbers from the flow data.
Do NOT say traffic "might be" anything — it IS a {label} attack.

## 1. Attack Summary
2-3 sentences: confirmed attack type, severity ({severity}), immediate risk.

## 2. Observed Behavior
2-3 sentences: cite SPECIFIC numbers from the flow (bytes/s, packet counts, flag values) that confirm {label}.

## 3. MITRE ATT&CK Mapping
For each technique: [ID] Technique Name — how it applies to this {label} attack.

## 4. Recommended Actions
6 numbered specific steps with tool names, thresholds, commands where possible.

Total: 350-450 words. Professional security analyst language."""

    try:
        response = llm.invoke(prompt)
        full_report = response.content.strip()
    except Exception as e:
        full_report = (
            f"## 1. Attack Summary\nA confirmed {label} attack of {severity} severity was detected.\n\n"
            f"## 2. Observed Behavior\n{reason}\n\n"
            f"## 3. MITRE ATT&CK Mapping\n{mitre_text}\n\n"
            f"## 4. Recommended Actions\n"
            f"1. Block source IP at perimeter firewall immediately.\n"
            f"2. Enable rate limiting on affected interfaces.\n"
            f"3. Capture packet data for forensic analysis.\n"
            f"4. Escalate to Tier 2 SOC analyst.\n"
            f"5. Review and update IDS/IPS signatures.\n"
            f"6. Document incident in ticketing system.\n[LLM error: {str(e)[:80]}]"
        )

    sections = {"summary": "", "behavior": "", "mitre_mapping": "", "recommended_actions": ""}
    current = None
    for line in full_report.split("\n"):
        if "## 1." in line:   current = "summary"
        elif "## 2." in line: current = "behavior"
        elif "## 3." in line: current = "mitre_mapping"
        elif "## 4." in line: current = "recommended_actions"
        elif current:         sections[current] += line + "\n"

    if not any(sections.values()):
        sections["summary"] = full_report

    return {
        "full_report": full_report,
        "summary": sections["summary"].strip(),
        "behavior": sections["behavior"].strip(),
        "mitre_mapping": sections["mitre_mapping"].strip(),
        "recommended_actions": sections["recommended_actions"].strip(),
    }