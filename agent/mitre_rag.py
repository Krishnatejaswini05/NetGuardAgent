"""
agent/mitre_rag.py
------------------
Builds and queries a FAISS vector store from MITRE ATT&CK technique descriptions.
Uses sentence-transformers (all-MiniLM-L6-v2) for 384-dim embeddings.
"""

import os
import pickle
import numpy as np
from pathlib import Path

# ── MITRE ATT&CK technique knowledge base ─────────────────────────────────────
# In a production system, download the full STIX dataset from:
# https://github.com/mitre/cti
# Here we include a comprehensive subset covering CICIDS-2017 attack categories.

MITRE_TECHNIQUES = [
    {
        "id": "T1498", "name": "Network Denial of Service",
        "tactic": "Impact",
        "description": (
            "Adversaries may perform Network Denial of Service (DoS) attacks to degrade "
            "or block the availability of targeted resources to users. Network DoS can be "
            "performed by exhausting the network bandwidth services rely on. Example "
            "resources include specific websites, email services, DNS, and web-based "
            "applications. Adversaries have been observed conducting network DoS attacks "
            "for political purposes and to support other malicious activities including "
            "distraction, hacktivism, and extortion."
        ),
        "detection": (
            "Monitor for anomalous network traffic that may indicate DoS attack, such as "
            "a large number of packets destined to a single address or from a single source. "
            "Track bandwidth utilization per interface and alert on sudden spikes."
        ),
        "remediation": [
            "Enable ingress rate limiting on upstream firewall (max 500 req/s per IP)",
            "Contact upstream ISP for traffic scrubbing or null-routing of attack source",
            "Deploy a cloud-based DDoS mitigation service (Cloudflare, AWS Shield)",
            "Enable SYN cookie protection on affected servers",
            "Block identified attack source IP ranges at the perimeter",
        ]
    },
    {
        "id": "T1498.001", "name": "Direct Network Flood",
        "tactic": "Impact",
        "description": (
            "Adversaries may attempt to cause a Denial of Service (DoS) by directly "
            "sending a high-volume of network traffic to a target. Direct Network Floods "
            "are when one or more systems are used to send a high-volume of network packets "
            "towards the targeted service's network. Almost any network protocol may be used "
            "for flooding. Stateless protocols such as UDP or ICMP are commonly used but "
            "stateful protocols such as TCP can be used as well."
        ),
        "detection": (
            "Monitor network packet rates and set alerting thresholds for unusual spikes. "
            "Look for high volumes of SYN packets, ICMP packets, or UDP floods from "
            "single or distributed sources."
        ),
        "remediation": [
            "Implement rate-limiting rules: block IPs exceeding 1000 packets/second",
            "Configure TCP SYN flood protection (SYN cookies) on all public-facing servers",
            "Deploy CAPTCHA or proof-of-work challenges on web entry points",
            "Enable automatic IP blacklisting via fail2ban or equivalent",
        ]
    },
    {
        "id": "T1499", "name": "Endpoint Denial of Service",
        "tactic": "Impact",
        "description": (
            "Adversaries may perform Endpoint Denial of Service (DoS) attacks to degrade "
            "or block the availability of services to users. Endpoint DoS can be performed "
            "by exhausting the system resources those services are hosted on or exploiting "
            "the system to cause a sustained crash/reboot. OS/Application vulnerability "
            "exploitation for DoS can be performed against a web server for example to "
            "crash the application. This applies to Slowloris and Slowhttptest attacks "
            "that exhaust server connection pools."
        ),
        "detection": (
            "Monitor for system crashes, service restarts, and high CPU/memory utilization. "
            "Check event logs for unexpected service terminations. Track active connection "
            "counts and alert when connection pools near exhaustion."
        ),
        "remediation": [
            "Configure server connection timeout (recommend 10-30 seconds for incomplete requests)",
            "Set maximum concurrent connections per IP address",
            "Enable request rate limiting at the web application layer (nginx/Apache)",
            "Deploy a reverse proxy or WAF to filter slow-rate attacks",
            "Increase server connection pool size as a temporary mitigation",
        ]
    },
    {
        "id": "T1046", "name": "Network Service Discovery",
        "tactic": "Discovery",
        "description": (
            "Adversaries may attempt to get a listing of services running on remote hosts "
            "and local network infrastructure devices, including those that may be vulnerable "
            "to remote exploitation through services. Common methods to acquire this "
            "information include port scans and vulnerability scans using tools such as Nmap, "
            "Masscan, or Zmap that are brought onto a system. Within cloud environments, "
            "adversaries may attempt to discover services running on other cloud instances."
        ),
        "detection": (
            "System and network discovery techniques normally occur throughout an operation "
            "as an adversary learns the environment. Monitor for high volumes of connection "
            "attempts to many ports on a single host, or sweeps across multiple hosts. "
            "Alert on sequential port scanning behavior."
        ),
        "remediation": [
            "Block all non-essential ports at the perimeter firewall",
            "Deploy network intrusion detection (Snort/Suricata) with port-scan rules",
            "Implement port knocking or firewall geo-blocking for sensitive services",
            "Alert on source IPs attempting more than 100 unique ports within 60 seconds",
            "Review and minimize exposed attack surface by disabling unnecessary services",
        ]
    },
    {
        "id": "T1595", "name": "Active Scanning",
        "tactic": "Reconnaissance",
        "description": (
            "Adversaries may execute active reconnaissance scans to gather information that "
            "can be used during targeting. Active scans are those where the adversary probes "
            "victim infrastructure via network traffic, as opposed to other forms of "
            "reconnaissance that do not involve direct interaction. Adversaries may perform "
            "different forms of active scanning depending on information they seek to gather."
        ),
        "detection": (
            "Monitor for suspicious network traffic that could be indicative of scanning "
            "activity, such as unusual ICMP traffic or high volumes of SYN packets to many "
            "ports. Correlate multiple connection failures from a single source IP."
        ),
        "remediation": [
            "Enable firewall logging and alert on sequential scanning patterns",
            "Implement automatic IP blocking after N failed connection attempts",
            "Use honeypot ports to detect and capture scanning activity",
            "Deploy deception technology (fake open ports) to mislead attackers",
        ]
    },
    {
        "id": "T1110", "name": "Brute Force",
        "tactic": "Credential Access",
        "description": (
            "Adversaries may use brute force techniques to gain access to accounts when "
            "passwords are unknown or when password hashes are obtained. Without knowledge "
            "of the password for an account or set of accounts, an adversary may "
            "systematically guess the password using a repetitive or iterative mechanism. "
            "Credential dumping may also be used to obtain password hashes from a system "
            "to brute force offline."
        ),
        "detection": (
            "Monitor authentication logs for system and application login failures of valid "
            "accounts. If authentication failures are high, there may be a brute force "
            "attempt to gain access using legitimate credentials. Alert on more than 10 "
            "failed login attempts per minute from a single IP."
        ),
        "remediation": [
            "Enable account lockout after 5-10 failed login attempts",
            "Implement multi-factor authentication (MFA) on all remote access points",
            "Block source IP after repeated failures using fail2ban or equivalent",
            "Deploy CAPTCHA challenges after 3 consecutive failures",
            "Audit and rotate credentials for accounts targeted by brute force",
        ]
    },
    {
        "id": "T1110.001", "name": "Password Guessing",
        "tactic": "Credential Access",
        "description": (
            "Adversaries with no prior knowledge of legitimate credentials within the system "
            "or environment may guess passwords to attempt access to accounts. Without "
            "knowledge of the password for an account, an adversary may opt to guess the "
            "password using a repetitive or iterative mechanism. Adversaries may guess login "
            "credentials without prior knowledge of system or environment passwords during "
            "an operation, such as FTP and SSH brute-force attacks."
        ),
        "detection": (
            "Monitor authentication logs for large numbers of failed login attempts. Track "
            "login attempts per IP per time window and alert on thresholds."
        ),
        "remediation": [
            "Enforce strong password policy (minimum 12 chars, complexity requirements)",
            "Implement progressive delays between login attempts",
            "Monitor and alert on credential stuffing patterns across accounts",
            "Disable password-based authentication for SSH; use key-based auth only",
        ]
    },
    {
        "id": "T1190", "name": "Exploit Public-Facing Application",
        "tactic": "Initial Access",
        "description": (
            "Adversaries may attempt to exploit a weakness in an Internet-facing host or "
            "system to initially access a network. The weakness in the system can be a "
            "software bug, a temporary glitch, or a misconfiguration. With internet-facing "
            "systems that are compromised there is not typically an opportunity for "
            "second-stage payloads. Web applications and web shells are common examples "
            "of exploitation targets including SQL injection, XSS, and RCE."
        ),
        "detection": (
            "Monitor application logs for abnormal behavior that may indicate attempted or "
            "successful exploitation. Use deep packet inspection to look for artifacts of "
            "common exploit traffic such as SQL injection strings or shell commands in "
            "HTTP parameters."
        ),
        "remediation": [
            "Apply all available security patches immediately for the affected application",
            "Deploy a Web Application Firewall (WAF) with OWASP CRS rules",
            "Conduct an immediate vulnerability scan on the affected application",
            "Review and sanitize all input validation on web endpoints",
            "Enable detailed request logging and monitor for exploitation artifacts",
        ]
    },
    {
        "id": "T1059", "name": "Command and Scripting Interpreter",
        "tactic": "Execution",
        "description": (
            "Adversaries may abuse command and script interpreters to execute commands, "
            "scripts, or binaries. These interfaces and languages provide ways of interacting "
            "with computer systems and are a common feature across many different platforms. "
            "Most systems come with some built-in command-line interface and scripting "
            "capabilities."
        ),
        "detection": (
            "Command line logging and monitoring of process execution. Look for unusual "
            "parent-child process relationships. Monitor for command-line arguments "
            "referencing known malicious tools or techniques."
        ),
        "remediation": [
            "Enable command-line audit logging on all endpoints",
            "Restrict execution of scripting engines (PowerShell, Python) via AppLocker",
            "Monitor process creation events for anomalous parent-child chains",
            "Implement application whitelisting to prevent unauthorized executable launch",
        ]
    },
    {
        "id": "T1071", "name": "Application Layer Protocol",
        "tactic": "Command and Control",
        "description": (
            "Adversaries may communicate using OSI application layer protocols to avoid "
            "detection/network filtering by blending in with existing traffic. Commands to "
            "the remote system, and often the results of those commands, will be embedded "
            "within the protocol traffic between the client and server. Adversaries may "
            "utilize many different protocols, including those used for web browsing, "
            "transferring files, electronic mail, or DNS."
        ),
        "detection": (
            "Analyze network data for uncommon data flows. Processes utilizing the network "
            "that do not normally have network communication or have never been seen before "
            "are suspicious. Analyze packet contents to detect application layer protocols "
            "that do not follow the expected protocol standards."
        ),
        "remediation": [
            "Implement deep packet inspection (DPI) on all egress traffic",
            "Deploy DNS filtering to block known malicious command-and-control domains",
            "Monitor for unusual beaconing patterns (regular intervals, unusual hours)",
            "Segment network to limit lateral movement opportunity",
        ]
    },
    {
        "id": "T1078", "name": "Valid Accounts",
        "tactic": "Initial Access",
        "description": (
            "Adversaries may obtain and abuse credentials of existing accounts as a means "
            "of gaining Initial Access, Persistence, Privilege Escalation, or Defense "
            "Evasion. Compromised credentials may be used to bypass access controls placed "
            "on various resources on systems within the network and may even be used for "
            "persistent access to remote systems and externally available services."
        ),
        "detection": (
            "Monitor for account logins at atypical hours or unusual locations. Track "
            "impossible travel events (login from two geographically distant locations "
            "within a short timeframe). Alert on first-time logins from new devices."
        ),
        "remediation": [
            "Immediately reset credentials for compromised accounts",
            "Enable multi-factor authentication (MFA) across all accounts",
            "Implement user and entity behavior analytics (UEBA) to detect anomalous logins",
            "Audit account privileges and apply principle of least privilege",
            "Review active sessions and terminate any unauthorized sessions immediately",
        ]
    },
    {
        "id": "T1133", "name": "External Remote Services",
        "tactic": "Initial Access",
        "description": (
            "Adversaries may leverage external-facing remote services to initially access "
            "and/or persist within a network. Remote services such as VPNs, Citrix, RDP, "
            "and other access mechanisms allow users to connect to internal enterprise "
            "network resources from external locations. There are often remote service gateways "
            "that manage connections and credential authentication for these services. "
            "Adversaries with access to valid accounts may use these services to gain access."
        ),
        "detection": (
            "Monitor for user accounts logged into systems they would not normally access "
            "or at abnormal times. Track VPN and RDP connection logs for unusual patterns."
        ),
        "remediation": [
            "Restrict remote service access by IP allowlist (known corporate IPs only)",
            "Enable MFA on all VPN and remote access gateways",
            "Audit remote access logs for suspicious authentication patterns",
            "Disable remote services not in active use (RDP, Telnet, FTP)",
        ]
    },
]


class MitreRAG:
    """
    MITRE ATT&CK Retrieval-Augmented Generation module.
    Builds a FAISS index from technique descriptions and retrieves
    top-k most semantically similar techniques for a given attack label.
    """

    def __init__(self, cache_path: str = "models/mitre_faiss.pkl"):
        self.cache_path = cache_path
        self._store = None

    def _load_or_build(self):
        if self._store is not None:
            return

        if Path(self.cache_path).exists():
            with open(self.cache_path, "rb") as f:
                self._store = pickle.load(f)
            return

        self._build_index()

    def _build_index(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        docs = []
        for t in MITRE_TECHNIQUES:
            text = (
                f"{t['name']} ({t['id']}) - Tactic: {t['tactic']}. "
                f"{t['description']} Detection: {t['detection']}"
            )
            docs.append(text)

        embeddings = embedder.encode(docs, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self._store = {
            "index": index,
            "docs": docs,
            "techniques": MITRE_TECHNIQUES,
            "embedder": embedder,
        }

        with open(self.cache_path, "wb") as f:
            pickle.dump(self._store, f)

    def retrieve(self, attack_label: str, top_k: int = 3) -> list[dict]:
        """Returns top-k MITRE techniques as a list of dicts."""
        import faiss

        self._load_or_build()

        if attack_label.upper() == "BENIGN":
            return []

        embedder = self._store["embedder"]
        index = self._store["index"]
        techniques = self._store["techniques"]

        query_vec = embedder.encode([attack_label]).astype("float32")
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            t = techniques[idx].copy()
            t["similarity_score"] = float(score)
            results.append(t)

        return results

    def retrieve_as_text(self, attack_label: str, top_k: int = 3) -> str:
        """Returns retrieved techniques formatted as a text block."""
        techniques = self.retrieve(attack_label, top_k)
        if not techniques:
            return "No malicious TTPs detected — traffic classified as benign."

        lines = []
        for i, t in enumerate(techniques, 1):
            lines.append(
                f"[{i}] {t['name']} ({t['id']})\n"
                f"    Tactic: {t['tactic']}\n"
                f"    Description: {t['description'][:250]}...\n"
                f"    Detection: {t['detection']}"
            )
        return "\n\n".join(lines)


# Singleton
_rag_instance = None

def get_rag() -> MitreRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = MitreRAG()
    return _rag_instance
