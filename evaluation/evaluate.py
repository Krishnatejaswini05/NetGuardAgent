"""
evaluation/evaluate.py
----------------------
Random Forest baseline training and NetGuardAgent evaluation utilities.
Generates classification reports, confusion matrices, and report quality scores.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import io
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

from agent.tools import FEATURE_COLS, tool_mitre_rag


def load_and_sample(csv_path: str, n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df[df["Label"].notna()].copy()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    n = min(n_samples, len(df))
    df_sample = df.groupby("Label", group_keys=False).apply(
        lambda x: x.sample(
            min(len(x), max(1, int(n * len(x) / len(df)))),
            random_state=seed
        )
    ).reset_index(drop=True)
    return df_sample


def create_synthetic_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Creates synthetic CICIDS-2017-like dataset with REALISTIC feature values
    matching actual attack signatures from the real dataset.
    """
    np.random.seed(seed)
    classes = [
        "BENIGN", "DoS Hulk", "DoS GoldenEye",
        "DoS slowloris", "DoS Slowhttptest", "Heartbleed"
    ]
    weights = [0.15, 0.40, 0.15, 0.10, 0.10, 0.10]
    labels = np.random.choice(classes, size=n, p=weights)

    records = []
    for label in labels:

        if label == "BENIGN":
            r = {
                "Flow Duration": np.random.randint(50000, 5000000),
                "Total Fwd Packets": np.random.randint(3, 25),
                "Total Backward Packets": np.random.randint(3, 25),
                "Total Length of Fwd Packets": np.random.randint(500, 10000),
                "Total Length of Bwd Packets": np.random.randint(500, 10000),
                "Fwd Packet Length Max": np.random.randint(100, 1500),
                "Fwd Packet Length Mean": np.random.uniform(100, 800),
                "Bwd Packet Length Max": np.random.randint(100, 1500),
                "Bwd Packet Length Mean": np.random.uniform(100, 800),
                "Flow Bytes/s": np.random.uniform(500, 80000),
                "Flow Packets/s": np.random.uniform(1, 150),
                "Flow IAT Mean": np.random.uniform(5000, 200000),
                "Flow IAT Std": np.random.uniform(1000, 50000),
                "Fwd IAT Mean": np.random.uniform(5000, 200000),
                "Bwd IAT Mean": np.random.uniform(5000, 200000),
                "SYN Flag Count": np.random.randint(0, 2),
                "ACK Flag Count": np.random.randint(1, 8),
                "PSH Flag Count": np.random.randint(0, 4),
                "FIN Flag Count": np.random.randint(0, 2),
                "RST Flag Count": 0,
                "URG Flag Count": 0,
            }

        elif label == "DoS Hulk":
            # Real CICIDS-2017 DoS Hulk: millions of bytes/s, thousands of packets
            # extremely short IAT, almost no backward traffic
            r = {
                "Flow Duration": np.random.randint(1, 500),
                "Total Fwd Packets": np.random.randint(3000, 15000),
                "Total Backward Packets": np.random.randint(0, 3),
                "Total Length of Fwd Packets": np.random.randint(3000000, 15000000),
                "Total Length of Bwd Packets": np.random.randint(0, 500),
                "Fwd Packet Length Max": 1500,
                "Fwd Packet Length Mean": np.random.uniform(1200, 1500),
                "Bwd Packet Length Max": 0,
                "Bwd Packet Length Mean": 0,
                "Flow Bytes/s": np.random.uniform(5_000_000, 50_000_000),
                "Flow Packets/s": np.random.uniform(10000, 80000),
                "Flow IAT Mean": np.random.uniform(0.01, 20),
                "Flow IAT Std": np.random.uniform(0.01, 5),
                "Fwd IAT Mean": np.random.uniform(0.01, 20),
                "Bwd IAT Mean": 0,
                "SYN Flag Count": np.random.randint(500, 3000),
                "ACK Flag Count": np.random.randint(0, 3),
                "PSH Flag Count": 0,
                "FIN Flag Count": 0,
                "RST Flag Count": np.random.randint(0, 10),
                "URG Flag Count": 0,
            }

        elif label == "DoS GoldenEye":
            # High pps, short duration, HTTP flood, minimal backward
            r = {
                "Flow Duration": np.random.randint(100, 5000),
                "Total Fwd Packets": np.random.randint(200, 2000),
                "Total Backward Packets": np.random.randint(0, 5),
                "Total Length of Fwd Packets": np.random.randint(100000, 2000000),
                "Total Length of Bwd Packets": np.random.randint(0, 1000),
                "Fwd Packet Length Max": 1500,
                "Fwd Packet Length Mean": np.random.uniform(800, 1500),
                "Bwd Packet Length Max": 0,
                "Bwd Packet Length Mean": 0,
                "Flow Bytes/s": np.random.uniform(1_000_000, 10_000_000),
                "Flow Packets/s": np.random.uniform(2000, 20000),
                "Flow IAT Mean": np.random.uniform(0.1, 100),
                "Flow IAT Std": np.random.uniform(0.1, 50),
                "Fwd IAT Mean": np.random.uniform(0.1, 100),
                "Bwd IAT Mean": 0,
                "SYN Flag Count": np.random.randint(100, 1000),
                "ACK Flag Count": np.random.randint(0, 5),
                "PSH Flag Count": 0,
                "FIN Flag Count": 0,
                "RST Flag Count": np.random.randint(0, 20),
                "URG Flag Count": 0,
            }

        elif label == "DoS slowloris":
            # Very long duration, very few packets, extremely low throughput, no FIN
            r = {
                "Flow Duration": np.random.randint(2000000, 10000000),
                "Total Fwd Packets": np.random.randint(3, 15),
                "Total Backward Packets": np.random.randint(0, 3),
                "Total Length of Fwd Packets": np.random.randint(100, 2000),
                "Total Length of Bwd Packets": np.random.randint(0, 500),
                "Fwd Packet Length Max": np.random.randint(100, 500),
                "Fwd Packet Length Mean": np.random.uniform(50, 300),
                "Bwd Packet Length Max": np.random.randint(0, 200),
                "Bwd Packet Length Mean": np.random.uniform(0, 100),
                "Flow Bytes/s": np.random.uniform(10, 500),
                "Flow Packets/s": np.random.uniform(0.1, 5),
                "Flow IAT Mean": np.random.uniform(500000, 2000000),
                "Flow IAT Std": np.random.uniform(100000, 500000),
                "Fwd IAT Mean": np.random.uniform(500000, 2000000),
                "Bwd IAT Mean": np.random.uniform(500000, 2000000),
                "SYN Flag Count": np.random.randint(1, 5),
                "ACK Flag Count": np.random.randint(1, 5),
                "PSH Flag Count": np.random.randint(0, 2),
                "FIN Flag Count": 0,
                "RST Flag Count": 0,
                "URG Flag Count": 0,
            }

        elif label == "DoS Slowhttptest":
            # Similar to slowloris but with RST flags
            r = {
                "Flow Duration": np.random.randint(1000000, 8000000),
                "Total Fwd Packets": np.random.randint(5, 25),
                "Total Backward Packets": np.random.randint(0, 5),
                "Total Length of Fwd Packets": np.random.randint(200, 5000),
                "Total Length of Bwd Packets": np.random.randint(0, 1000),
                "Fwd Packet Length Max": np.random.randint(100, 600),
                "Fwd Packet Length Mean": np.random.uniform(50, 400),
                "Bwd Packet Length Max": np.random.randint(0, 300),
                "Bwd Packet Length Mean": np.random.uniform(0, 150),
                "Flow Bytes/s": np.random.uniform(50, 3000),
                "Flow Packets/s": np.random.uniform(0.5, 10),
                "Flow IAT Mean": np.random.uniform(200000, 1500000),
                "Flow IAT Std": np.random.uniform(50000, 300000),
                "Fwd IAT Mean": np.random.uniform(200000, 1500000),
                "Bwd IAT Mean": np.random.uniform(200000, 1500000),
                "SYN Flag Count": np.random.randint(1, 5),
                "ACK Flag Count": np.random.randint(1, 5),
                "PSH Flag Count": np.random.randint(0, 3),
                "FIN Flag Count": 0,
                "RST Flag Count": np.random.randint(2, 15),
                "URG Flag Count": 0,
            }

        else:  # Heartbleed
            # Oversized fwd packets, small count, SSL-like
            r = {
                "Flow Duration": np.random.randint(10000, 500000),
                "Total Fwd Packets": np.random.randint(5, 20),
                "Total Backward Packets": np.random.randint(5, 20),
                "Total Length of Fwd Packets": np.random.randint(100000, 600000),
                "Total Length of Bwd Packets": np.random.randint(50000, 300000),
                "Fwd Packet Length Max": 65535,
                "Fwd Packet Length Mean": np.random.uniform(15000, 60000),
                "Bwd Packet Length Max": 65535,
                "Bwd Packet Length Mean": np.random.uniform(10000, 40000),
                "Flow Bytes/s": np.random.uniform(50000, 500000),
                "Flow Packets/s": np.random.uniform(10, 200),
                "Flow IAT Mean": np.random.uniform(1000, 50000),
                "Flow IAT Std": np.random.uniform(500, 10000),
                "Fwd IAT Mean": np.random.uniform(1000, 50000),
                "Bwd IAT Mean": np.random.uniform(1000, 50000),
                "SYN Flag Count": np.random.randint(1, 4),
                "ACK Flag Count": np.random.randint(2, 8),
                "PSH Flag Count": np.random.randint(1, 4),
                "FIN Flag Count": np.random.randint(0, 2),
                "RST Flag Count": 0,
                "URG Flag Count": 0,
            }

        r["Label"] = label
        records.append(r)

    return pd.DataFrame(records)


def train_random_forest(df: pd.DataFrame):
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["Label"].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )
    accuracy = accuracy_score(y_test, y_pred)

    return rf, le, X_test, y_test, y_pred, report, accuracy


def plot_confusion_matrix(y_test, y_pred, class_names: list) -> bytes:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Random Forest — Confusion Matrix (CICIDS-2017)", fontsize=13, pad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


def plot_class_distribution(df: pd.DataFrame) -> bytes:
    counts = df["Label"].value_counts()
    colors = ["#2196F3" if "BENIGN" in c else "#F44336" for c in counts.index]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Attack Class", fontsize=11)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.set_title("CICIDS-2017 — Class Distribution", fontsize=12, pad=8)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, str(val),
                ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


def score_report(report_text: str, true_label: str, pred_label: str, mitre_techniques: list) -> dict:
    sections = ["## 1.", "## 2.", "## 3.", "## 4."]
    found = sum(1 for s in sections if s in report_text)
    completeness = 3 if found == 4 else (2 if found >= 2 else 1)

    if pred_label == true_label:
        correctness = 3
    elif pred_label.split()[0] == true_label.split()[0]:
        correctness = 2
    else:
        correctness = 1

    if mitre_techniques and any(t["id"] in report_text for t in mitre_techniques):
        relevance = 3
    elif any(kw in report_text.lower() for kw in ["tactic", "technique", "mitre", "T1"]):
        relevance = 2
    else:
        relevance = 1

    action_kws = [
        "block", "rate limit", "firewall", "monitor", "patch",
        "escalate", "disable", "capture", "notify", "isolate",
        "reset", "enable mfa", "fail2ban", "alert", "restrict"
    ]
    kw_count = sum(1 for kw in action_kws if kw in report_text.lower())
    actionability = min(3, max(1, kw_count // 2 + 1))

    return {
        "completeness": completeness,
        "correctness": correctness,
        "relevance": relevance,
        "actionability": actionability,
        "total": completeness + correctness + relevance + actionability,
    }


def plot_report_quality(scores_list: list) -> bytes:
    dims = ["Completeness", "Correctness", "Relevance", "Actionability"]
    avgs = [np.mean([s[d.lower()] for s in scores_list]) for d in dims]
    colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(dims, avgs, color=colors, edgecolor="white", width=0.5)
    ax.set_ylim(0, 3.5)
    ax.set_ylabel("Average Score (max 3)", fontsize=11)
    ax.set_title("NetGuardAgent — Incident Report Quality Scores", fontsize=12, pad=8)
    ax.axhline(3, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Max (3)")
    for bar, v in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.06, f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()