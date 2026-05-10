"""
app.py — NetGuardAgent Streamlit Dashboard
Run with: streamlit run app.py
"""

import os
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="NetGuardAgent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Sora:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
    .ng-header { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); border: 1px solid #21262d; border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 1rem; }
    .ng-title { font-size: 1.6rem; font-weight: 600; color: #e6edf3; margin: 0; }
    .ng-subtitle { font-size: 0.82rem; color: #8b949e; margin: 0; }
    .ng-badge { margin-left: auto; background: #1f6feb22; border: 1px solid #1f6feb55; border-radius: 20px; padding: 0.3rem 0.9rem; font-size: 0.75rem; color: #58a6ff; font-family: 'JetBrains Mono', monospace; }
    .metric-card { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 1.1rem 1.2rem; text-align: center; }
    .metric-val { font-size: 1.8rem; font-weight: 600; color: #e6edf3; font-family: 'JetBrains Mono', monospace; }
    .metric-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.2rem; }
    .sev-critical { background: #490202; color: #ffa198; border: 1px solid #f85149; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 600; }
    .sev-high { background: #3d1f00; color: #ffa657; border: 1px solid #d29922; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 600; }
    .sev-medium { background: #2d2a00; color: #e3b341; border: 1px solid #9e6a03; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 600; }
    .sev-low { background: #0d2a0f; color: #56d364; border: 1px solid #238636; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 600; }
    .sev-none { background: #161b22; color: #8b949e; border: 1px solid #30363d; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; }
    .mitre-card { background: #0d1117; border: 1px solid #1f6feb44; border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; }
    .mitre-id { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #58a6ff; font-weight: 500; }
    .mitre-name { font-size: 0.92rem; font-weight: 500; color: #e6edf3; }
    .mitre-tactic { font-size: 0.75rem; color: #8b949e; margin-top: 0.2rem; }
    .log-box { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #c9d1d9; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }
    .anomaly-flag { background: #3d1a00; border: 1px solid #d29922; border-radius: 6px; padding: 0.4rem 0.8rem; font-size: 0.8rem; color: #ffa657; margin-bottom: 0.4rem; font-family: 'JetBrains Mono', monospace; }
    hr { border-color: #21262d; }
    .stButton > button { background: #1f6feb; color: white; border: none; border-radius: 8px; padding: 0.6rem 1.4rem; font-family: 'Sora', sans-serif; font-weight: 500; width: 100%; }
    .stButton > button:hover { background: #388bfd; border: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def severity_badge(sev):
    cls = f"sev-{sev.lower()}" if sev.lower() in ["critical","high","medium","low"] else "sev-none"
    return f'<span class="{cls}">{sev}</span>'

def confidence_color(conf):
    return {"High": "#56d364", "Medium": "#e3b341", "Low": "#f85149"}.get(conf, "#8b949e")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ NetGuardAgent")
    st.markdown("<p style='font-size:0.75rem;color:#8b949e;'>CS 6349.501 · UT Dallas</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**API Configuration**")
    api_key_input = st.text_input("Groq API Key", value=os.environ.get("GROQ_API_KEY",""), type="password", help="Get a FREE key at console.groq.com", placeholder="gsk_...")
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input
    api_ok = bool(os.environ.get("GROQ_API_KEY","").startswith("gsk_"))
    if api_ok:
        st.success("API key ready", icon="✅")
    else:
        st.warning("Add your Groq key above", icon="⚠️")
    st.divider()
    st.markdown("**Navigation**")
    page = st.radio("Go to", ["🔍 Analyze Traffic", "📊 Evaluation", "ℹ️ About"], label_visibility="collapsed")
    st.divider()
    st.markdown("<p style='font-size:0.72rem;color:#8b949e;line-height:1.6;'><b style='color:#c9d1d9;'>Stack</b><br>LangGraph · Llama 3.3 · Groq<br>MITRE ATT&CK · FAISS<br>sentence-transformers<br>CICIDS-2017 dataset</p>", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ng-header">
  <div style="font-size:2.2rem;">🛡️</div>
  <div>
    <p class="ng-title">NetGuardAgent</p>
    <p class="ng-subtitle">Agentic GenAI · Network Intrusion Detection · Incident Report Generation</p>
  </div>
  <div class="ng-badge">LangGraph + MITRE ATT&CK + RAG</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYZE TRAFFIC
# ══════════════════════════════════════════════════════════════════════════════
if "🔍" in page:
    st.markdown("### Analyze Network Traffic")
    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
        st.markdown("**Input Method**")
        input_method = st.radio("Choose input", ["📁 Upload CSV", "🎲 Random Sample (demo)", "✏️ Manual Entry"], label_visibility="collapsed")
        st.divider()

        row_data = None
        true_label = None

        if "Upload" in input_method:
            uploaded = st.file_uploader("Upload CICIDS-2017 CSV", type=["csv"])
            if uploaded:
                with st.spinner("Loading dataset..."):
                    try:
                        df = pd.read_csv(uploaded)
                        df.columns = df.columns.str.strip()
                        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
                        st.session_state["dataset"] = df
                        st.success(f"Loaded {len(df):,} rows · {df['Label'].nunique()} classes")
                    except Exception as e:
                        st.error(f"Error loading CSV: {e}")
            if "dataset" in st.session_state:
                df = st.session_state["dataset"]
                classes = df["Label"].unique().tolist()
                selected_class = st.selectbox("Filter by class", ["All"] + sorted(classes))
                filtered = df if selected_class == "All" else df[df["Label"] == selected_class]
                idx = st.slider("Row index", 0, len(filtered)-1, 0)
                row = filtered.iloc[idx]
                true_label = row.get("Label", "Unknown")
                row_data = row.drop("Label", errors="ignore")
                st.info(f"Ground truth: **{true_label}**")

        elif "Random" in input_method:
            if st.button("🎲 Generate New Sample"):
                st.session_state.pop("sample_row", None)
            if "sample_row" not in st.session_state:
                from evaluation.evaluate import create_synthetic_dataset
                demo_df = create_synthetic_dataset(n=200)
                attack_df = demo_df[demo_df["Label"] != "BENIGN"]
                sample = attack_df.sample(1).iloc[0]
                st.session_state["sample_row"] = sample
            sample = st.session_state["sample_row"]
            true_label = sample["Label"]
            row_data = sample.drop("Label", errors="ignore")
            st.info(f"Sample ground truth: **{true_label}**")
            st.caption("Using synthetic CICIDS-2017-like data. Upload a real CSV for full evaluation.")

        else:
            st.markdown("**Key flow features**")
            from agent.tools import FEATURE_COLS
            manual = {}
            manual["Flow Duration"] = st.number_input("Flow Duration (µs)", value=50000, min_value=0)
            manual["Total Fwd Packets"] = st.number_input("Total Fwd Packets", value=10, min_value=0)
            manual["Total Backward Packets"] = st.number_input("Total Bwd Packets", value=8, min_value=0)
            manual["Flow Bytes/s"] = st.number_input("Flow Bytes/s", value=5000.0, min_value=0.0, format="%.2f")
            manual["Flow Packets/s"] = st.number_input("Flow Packets/s", value=50.0, min_value=0.0, format="%.2f")
            manual["SYN Flag Count"] = st.number_input("SYN Flag Count", value=1, min_value=0)
            manual["ACK Flag Count"] = st.number_input("ACK Flag Count", value=3, min_value=0)
            manual["Fwd Packet Length Mean"] = st.number_input("Fwd Pkt Length Mean", value=200.0, format="%.2f")
            manual["Flow IAT Mean"] = st.number_input("Flow IAT Mean (µs)", value=10000.0, format="%.2f")
            for c in FEATURE_COLS:
                if c not in manual:
                    manual[c] = 0.0
            row_data = pd.Series(manual)
            true_label = "Manual entry"

        st.divider()
        run_btn = st.button("🚀 Run NetGuardAgent Pipeline", disabled=row_data is None or not api_ok)

    with col_right:
        if row_data is None:
            st.markdown("""
            <div style='background:#161b22;border:1px solid #21262d;border-radius:12px;padding:3rem;text-align:center;margin-top:2rem;'>
              <div style='font-size:3rem;margin-bottom:1rem;'>🔍</div>
              <p style='color:#8b949e;font-size:0.95rem;'>Select a traffic sample on the left, then click<br><b style='color:#e6edf3;'>Run NetGuardAgent Pipeline</b> to analyze it.</p>
            </div>
            """, unsafe_allow_html=True)

        elif run_btn:
            from agent.graph import run_agent
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                st.markdown("**Pipeline running...**")
                prog = st.progress(0)
                status_text = st.empty()
            start_time = time.time()
            try:
                status_text.markdown("🔎 **Log Analyzer** — Parsing network flow features...")
                prog.progress(10)
                result = run_agent(row_data)
                prog.progress(40); time.sleep(0.3)
                status_text.markdown("🧠 **Threat Classifier** — Classification complete")
                prog.progress(65); time.sleep(0.3)
                status_text.markdown("🗂️ **MITRE RAG** — TTPs retrieved")
                prog.progress(85); time.sleep(0.3)
                status_text.markdown("📝 **Report Generator** — Report generated")
                prog.progress(100); time.sleep(0.3)
                st.session_state["last_result"] = result
                st.session_state["last_true_label"] = true_label
                progress_placeholder.empty()
            except Exception as e:
                progress_placeholder.empty()
                st.error(f"Pipeline error: {e}")
                st.info("Make sure your Groq API key is valid and you have internet access.")

        # ── Display results ────────────────────────────────────────────────────
        if "last_result" in st.session_state and row_data is not None:
            result   = st.session_state["last_result"]
            clf      = result.get("classification", {})
            mitre    = result.get("mitre_techniques", [])
            report   = result.get("report", {})
            parsed   = result.get("parsed_log", {})
            true_lbl = st.session_state.get("last_true_label", "—")
            label    = clf.get("label", "Unknown")
            severity = clf.get("severity", "None")
            confidence = clf.get("confidence", "Unknown")

            # Metric cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{label.split()[0] if label != "BENIGN" else "✓"}</div><div class="metric-label">Detection</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{confidence_color(confidence)};font-size:1.2rem;">{severity}</div><div class="metric-label">Severity</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1.1rem;color:{confidence_color(confidence)};">{confidence}</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1.1rem;">{len(mitre)}</div><div class="metric-label">MITRE TTPs</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            tab1, tab2, tab3, tab4 = st.tabs(["📋 Incident Report", "🗂️ MITRE ATT&CK", "🔎 Log Analysis", "📄 Raw JSON"])

            # ── Tab 1: Incident Report ─────────────────────────────────────────
            with tab1:
                if label != "BENIGN":
                    st.markdown(f"**Detected:** {label} &nbsp;|&nbsp; **Severity:** {severity_badge(severity)} &nbsp;|&nbsp; **True Label:** `{true_lbl}`", unsafe_allow_html=True)
                else:
                    st.success("✅ Traffic classified as BENIGN — no malicious activity detected.")
                st.markdown(f"**Classifier reasoning:** {clf.get('reason','N/A')}")
                st.divider()

                sections = [
                    ("📌 Attack Summary",       report.get("summary", "")),
                    ("👁️ Observed Behavior",    report.get("behavior", "")),
                    ("🗂️ MITRE ATT&CK Mapping", report.get("mitre_mapping", "")),
                    ("✅ Recommended Actions",   report.get("recommended_actions", "")),
                ]
                for title, content in sections:
                    if content:
                        with st.expander(title, expanded=True):
                            st.markdown(content)
                    else:
                        full = report.get("full_report", "")
                        if full and sections.index((title, content)) == 0:
                            st.markdown(full)
                        break

                # ── Download buttons ───────────────────────────────────────────
                st.divider()
                st.markdown("**📥 Export Incident Report**")

                mitre_lines = "\n".join(
                    f"  - {t['name']} ({t['id']}) | Tactic: {t['tactic']}"
                    for t in mitre
                ) if mitre else "  No techniques identified."

                txt_report = (
                    "NETGUARDAGENT — INCIDENT REPORT\n"
                    "================================\n"
                    "Generated by: NetGuardAgent (LangGraph + MITRE ATT&CK + RAG)\n"
                    "Course: CS 6349.501 Network Security — UT Dallas\n"
                    "Author: Krishna Tejaswini Paleti\n\n"
                    "DETECTION SUMMARY\n"
                    "-----------------\n"
                    f"Attack Type  : {label}\n"
                    f"Severity     : {severity}\n"
                    f"Confidence   : {confidence}\n"
                    f"True Label   : {true_lbl}\n\n"
                    "MITRE ATT&CK TECHNIQUES\n"
                    "------------------------\n"
                    f"{mitre_lines}\n\n"
                    "FULL INCIDENT REPORT\n"
                    "--------------------\n"
                    f"{report.get('full_report', 'No report generated.')}\n\n"
                    "NETWORK FLOW LOG\n"
                    "----------------\n"
                    f"{parsed.get('text', 'No log data.')}\n"
                )

                md_report = (
                    "# NetGuardAgent — Incident Report\n\n"
                    f"**Attack Type:** {label}  \n"
                    f"**Severity:** {severity}  \n"
                    f"**Confidence:** {confidence}  \n"
                    f"**True Label:** {true_lbl}  \n\n"
                    "---\n\n"
                    "## MITRE ATT&CK Techniques\n"
                    f"{mitre_lines}\n\n"
                    "---\n\n"
                    f"{report.get('full_report', 'No report generated.')}\n\n"
                    "---\n\n"
                    "## Network Flow Log\n"
                    "```\n"
                    f"{parsed.get('text', 'No log data.')}\n"
                    "```\n\n"
                    "---\n"
                    "*Generated by NetGuardAgent — CS 6349.501 Network Security — UT Dallas*  \n"
                    "*Author: Krishna Tejaswini Paleti*\n"
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    st.download_button(
                        label="📄 Download as TXT",
                        data=txt_report,
                        file_name=f"incident_report_{label.replace(' ','_')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with col_b:
                    st.download_button(
                        label="📝 Download as Markdown",
                        data=md_report,
                        file_name=f"incident_report_{label.replace(' ','_')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

            # ── Tab 2: MITRE ATT&CK ───────────────────────────────────────────
            with tab2:
                if mitre:
                    for t in mitre:
                        remediation = t.get("remediation", [])
                        st.markdown(f"""
                        <div class="mitre-card">
                          <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
                            <span class="mitre-id">{t['id']}</span>
                            <span class="mitre-name">{t['name']}</span>
                          </div>
                          <div class="mitre-tactic">Tactic: {t['tactic']} &nbsp;·&nbsp; Similarity: {t.get('similarity_score',0):.3f}</div>
                          <p style="font-size:0.83rem;color:#8b949e;margin-top:8px;line-height:1.6;">{t['description'][:300]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                        if remediation:
                            with st.expander(f"Remediation steps for {t['id']}"):
                                for step in remediation:
                                    st.markdown(f"• {step}")
                else:
                    st.info("No MITRE ATT&CK techniques retrieved — traffic classified as benign.")

            # ── Tab 3: Log Analysis ───────────────────────────────────────────
            with tab3:
                flags = parsed.get("flags", [])
                if flags:
                    st.markdown("**⚠️ Anomaly Indicators Detected**")
                    for f in flags:
                        st.markdown(f'<div class="anomaly-flag">⚠ {f}</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Parsed Flow Log**")
                st.markdown(f'<div class="log-box">{parsed.get("text","No log data.")}</div>', unsafe_allow_html=True)
                stats = parsed.get("stats", {})
                if stats:
                    st.markdown("<br>**Feature Values**", unsafe_allow_html=True)
                    stats_df = pd.DataFrame([
                        {"Feature": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                        for k, v in stats.items() if v != 0
                    ])
                    st.dataframe(stats_df, use_container_width=True, height=250)

            # ── Tab 4: Raw JSON ───────────────────────────────────────────────
            with tab4:
                st.json({
                    "classification": clf,
                    "mitre_techniques": [{"id": t["id"], "name": t["name"], "tactic": t["tactic"]} for t in mitre],
                    "report_sections": {"summary": report.get("summary","")[:200], "behavior": report.get("behavior","")[:200]},
                    "anomaly_flags": parsed.get("flags", []),
                })


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif "📊" in page:
    st.markdown("### Model Evaluation")
    tab_rf, tab_agent, tab_compare = st.tabs(["🌲 Random Forest Baseline", "🤖 NetGuardAgent Eval", "📊 Comparison"])

    with tab_rf:
        st.markdown("#### Random Forest Classifier — Baseline Evaluation")
        col1, col2 = st.columns([1, 2])
        with col1:
            rf_source = st.radio("Dataset source", ["Use synthetic demo data", "Upload CICIDS-2017 CSV"], key="rf_source")
            if "Upload" in rf_source:
                rf_csv = st.file_uploader("Upload CSV", type=["csv"], key="rf_csv")
            n_samples = st.slider("Sample size", 500, 10000, 2000, 500)
            run_rf = st.button("🌲 Train Random Forest")
        with col2:
            if run_rf:
                from evaluation.evaluate import load_and_sample, create_synthetic_dataset, train_random_forest
                with st.spinner("Training Random Forest..."):
                    if "Upload" in rf_source and st.session_state.get("rf_csv"):
                        df = load_and_sample(st.session_state.rf_csv, n_samples)
                    else:
                        df = create_synthetic_dataset(n=n_samples)
                    rf, le, X_test, y_test, y_pred, report_dict, accuracy = train_random_forest(df)
                    st.session_state["rf_results"] = (rf, le, X_test, y_test, y_pred, report_dict, accuracy, df)
                st.success(f"Training complete! Accuracy: **{accuracy:.4%}**")

            if "rf_results" in st.session_state:
                rf, le, X_test, y_test, y_pred, report_dict, accuracy, df = st.session_state["rf_results"]
                from evaluation.evaluate import plot_confusion_matrix, plot_class_distribution
                m1, m2, m3 = st.columns(3)
                m1.metric("Overall Accuracy", f"{accuracy:.4%}")
                m2.metric("Macro F1", f"{report_dict.get('macro avg',{}).get('f1-score',0):.4f}")
                m3.metric("Test Samples", len(y_test))
                st.markdown("**Per-Class Results**")
                rows = []
                for cls in le.classes_:
                    if cls in report_dict:
                        r = report_dict[cls]
                        rows.append({"Class": cls, "Precision": f"{r['precision']:.3f}", "Recall": f"{r['recall']:.3f}", "F1-Score": f"{r['f1-score']:.3f}", "Support": int(r["support"])})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                st.markdown("**Confusion Matrix**")
                st.image(plot_confusion_matrix(y_test, y_pred, le.classes_.tolist()), use_column_width=True)
                st.markdown("**Class Distribution**")
                st.image(plot_class_distribution(df), use_column_width=True)

    with tab_agent:
        st.markdown("#### NetGuardAgent — LLM Classification Evaluation")
        if not api_ok:
            st.warning("Add your Groq API key in the sidebar to run agent evaluation.")
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                eval_n = st.slider("Number of samples to evaluate", 5, 50, 15)
                delay  = st.slider("Delay between API calls (sec)", 1, 5, 2)
                run_eval = st.button("🤖 Run Agent Evaluation")
            with col2:
                if run_eval:
                    from agent.graph import run_agent
                    from evaluation.evaluate import create_synthetic_dataset, score_report
                    demo_df = create_synthetic_dataset(n=200)
                    eval_sample = demo_df.sample(min(eval_n, len(demo_df)), random_state=77)
                    results_log = []
                    progress = st.progress(0)
                    status   = st.empty()
                    table_ph = st.empty()
                    for i, (idx, row) in enumerate(eval_sample.iterrows()):
                        true_label = row["Label"]
                        status.markdown(f"**Analyzing sample {i+1}/{eval_n}** — `{true_label}`")
                        progress.progress((i+1)/eval_n)
                        try:
                            res  = run_agent(row.drop("Label", errors="ignore"))
                            pred = res["classification"].get("label", "BENIGN")
                            conf = res["classification"].get("confidence", "Low")
                            quality = score_report(res["report"].get("full_report",""), true_label, pred, res.get("mitre_techniques",[]))
                        except Exception:
                            pred = "ERROR"; conf = "Low"
                            quality = {"total": 4}
                        results_log.append({"True": true_label, "Predicted": pred, "Match": "✅" if pred==true_label else "❌", "Confidence": conf, "Quality Score": f"{quality['total']}/12"})
                        table_ph.dataframe(pd.DataFrame(results_log), use_container_width=True)
                        time.sleep(delay)
                    status.empty()
                    df_r = pd.DataFrame(results_log)
                    acc = (df_r["Match"]=="✅").sum()/len(df_r)
                    st.success(f"Evaluation complete! Accuracy: **{acc:.2%}** on {eval_n} samples")

    with tab_compare:
        st.markdown("#### System Comparison")
        st.dataframe(pd.DataFrame({
            "Method": ["Random Forest (Baseline)", "NetGuardAgent (Ours)"],
            "Architecture": ["Statistical ML", "LLM Agent (LangGraph)"],
            "Accuracy": ["99.93%", "88-93% (LLM)"],
            "Macro F1": ["~0.99", "Evaluated above"],
            "NL Reports": ["❌ No", "✅ Yes"],
            "MITRE Context": ["❌ No", "✅ Yes"],
            "Explainability": ["Feature importance only", "Full NL explanation"],
        }), use_container_width=True)
        st.markdown("> **Key insight:** Random Forest achieves near-perfect classification accuracy but produces no human-readable output. NetGuardAgent uniquely generates structured incident reports grounded in MITRE ATT&CK threat intelligence.")
        st.markdown("#### Random Forest — Reported Results (Wednesday file)")
        st.dataframe(pd.DataFrame([
            {"Class": "BENIGN",           "Precision": 1.00, "Recall": 1.00, "F1": 1.00, "Support": 87883},
            {"Class": "DoS GoldenEye",    "Precision": 1.00, "Recall": 1.00, "F1": 1.00, "Support": 2107},
            {"Class": "DoS Hulk",         "Precision": 1.00, "Recall": 1.00, "F1": 1.00, "Support": 46042},
            {"Class": "DoS Slowhttptest", "Precision": 1.00, "Recall": 0.99, "F1": 0.99, "Support": 1096},
            {"Class": "DoS slowloris",    "Precision": 0.99, "Recall": 1.00, "F1": 1.00, "Support": 1152},
            {"Class": "Heartbleed",       "Precision": 1.00, "Recall": 1.00, "F1": 1.00, "Support": 2},
            {"Class": "Overall",          "Precision": None, "Recall": None, "F1": 0.9993, "Support": 138282},
        ]), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif "ℹ️" in page:
    st.markdown("### About NetGuardAgent")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
#### System Overview
NetGuardAgent is a four-tool autonomous agent built on **LangGraph**
that transforms raw network traffic into actionable security intelligence.

#### Pipeline
1. **Log Analyzer** — Parses CICIDS-2017 flow features into natural language
2. **Threat Classifier** — Llama 3.3 70B via Groq classifies the attack type
3. **MITRE ATT&CK RAG** — FAISS retrieves relevant threat intelligence
4. **Report Generator** — Llama 3.3 writes a structured incident report

#### Tech Stack
| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph |
| LLM | Llama 3.3 70B (Groq) |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Store | FAISS (CPU) |
| Dataset | CICIDS-2017 |
| Baseline | Random Forest |
        """)
    with col2:
        st.markdown("""
#### References
[1] Cheng et al., "KAIROS," IEEE S&P 2024.

[2] Deng et al., "PentestGPT," USENIX Security 2024.

[3] Ayzenshteyn et al., "Cloak, Honey, Trap," USENIX Security 2025.

[4] Wu et al., "AutoGen," arXiv 2023.

#### Getting Started
1. Get a **free Groq API key** at [console.groq.com](https://console.groq.com)
2. Add it to `.env` as `GROQ_API_KEY=gsk_...`
3. `pip install -r requirements.txt`
4. `streamlit run app.py`

#### Author
**Krishna Tejaswini Paleti**
University of Texas at Dallas — CS 6349.501
        """)
    st.divider()
    st.markdown("<div style='text-align:center;color:#8b949e;font-size:0.8rem;'>NetGuardAgent · CS 6349.501 Network Security · University of Texas at Dallas</div>", unsafe_allow_html=True)