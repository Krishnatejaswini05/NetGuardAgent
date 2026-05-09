# 🛡️ NetGuardAgent

**Agentic GenAI Framework for Automated Network Intrusion Detection and Incident Report Generation**

> CS 6349.501 Network Security · University of Texas at Dallas · Krishna Tejaswini Paleti

---

## What It Does

NetGuardAgent is a four-tool autonomous LangGraph agent that:
1. **Parses** raw CICIDS-2017 network flow records into natural language
2. **Classifies** the traffic as an attack type using Llama 3 (Groq API)
3. **Retrieves** relevant MITRE ATT&CK threat intelligence via FAISS RAG
4. **Generates** a structured incident report with remediation steps

---

## Quick Setup (15 minutes)

### Step 1 — Prerequisites
- Python 3.10 or higher
- VS Code (recommended) or any terminal
- Internet connection

### Step 2 — Clone / open the project
Open this folder in VS Code.

### Step 3 — Create a virtual environment
```bash
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Get your FREE Groq API key
1. Go to https://console.groq.com
2. Sign up (free, no credit card needed)
3. Click "Create API Key"
4. Copy the key (starts with `gsk_`)

### Step 6 — Add the key to .env
Open `.env` and replace `your_groq_api_key_here`:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### Step 7 — Run the app
```bash
streamlit run app.py
```

Your browser opens at **http://localhost:8501** 🎉

---

## Getting the CICIDS-2017 Dataset

### Option A — Direct Download (Recommended)
1. Go to: https://www.unb.ca/cic/datasets/ids-2017.html
2. Download **Wednesday-workingHours.pcap_ISCX.csv** (~450 MB)
3. Place it in the `data/` folder

### Option B — Kaggle
```bash
pip install kaggle
# Upload kaggle.json to project root first
kaggle datasets download -d cicdataset/cicids2017 --unzip -p data/
```

### Option C — Demo without dataset
The app includes a synthetic dataset generator — you can test the full pipeline
without downloading anything. Just click **"Random Sample (demo)"** in the UI.

---

## Project Structure

```
NetGuardAgent/
├── app.py                      ← Streamlit dashboard (main UI)
├── agent/
│   ├── __init__.py
│   ├── graph.py                ← LangGraph agent (4 nodes)
│   ├── tools.py                ← Tool 1-4 implementations
│   └── mitre_rag.py            ← FAISS + MITRE ATT&CK knowledge base
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py             ← Random Forest baseline + metrics
├── data/
│   └── (place CICIDS CSV here)
├── models/
│   └── mitre_faiss.pkl         ← Auto-generated on first run
├── .env                        ← Your GROQ_API_KEY (never commit this)
├── requirements.txt
└── README.md
```

---

## Using the App

### Analyze Traffic (main page)
- **Upload CSV**: Upload your CICIDS-2017 CSV, pick a row, click Run
- **Random Sample**: Auto-generates a synthetic flow for quick testing
- **Manual Entry**: Enter flow feature values by hand

Click **"Run NetGuardAgent Pipeline"** to see:
- Attack classification + severity
- Parsed log with anomaly flags
- MITRE ATT&CK techniques + remediation steps
- Full structured incident report

### Evaluation Page
- **Random Forest tab**: Train and evaluate the baseline (no API key needed)
- **NetGuardAgent tab**: Run LLM evaluation on N samples (uses API key)
- **Comparison tab**: Side-by-side comparison table

---

## Groq API Rate Limits (Free Tier)

The free Groq tier allows ~30 requests/minute for Llama 3 8B.

When running agent evaluation on multiple samples, use the delay slider
(recommend 2 seconds between calls) to avoid hitting rate limits.

For higher throughput, upgrade to Groq's paid tier or use `llama3-70b-8192`
(same rate limits, better classification quality).

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `GROQ_API_KEY not set` | Add key to `.env` or sidebar input |
| `Cannot find module` | Run `pip install -r requirements.txt` |
| `Rate limit exceeded` | Increase delay between API calls |
| CSV columns not found | Make sure you strip whitespace; the app does this automatically |
| `faiss-cpu` install error | Try `pip install faiss-cpu --no-cache-dir` |

---

## Key Technical Choices

| Choice | Reason |
|--------|--------|
| LangGraph over LangChain | Stateful graph enables context accumulation across all 4 tool calls |
| Groq API | Free tier, ~10x faster than OpenAI for Llama 3, no local GPU needed |
| FAISS CPU | No GPU required, fast enough for top-3 retrieval on ~12 techniques |
| all-MiniLM-L6-v2 | Lightweight (80MB), excellent semantic similarity for security text |
| Random Forest baseline | Strong, well-established NIDS benchmark in literature |

---

## Citation

If you use this code, please cite:

```
Paleti, K.T. (2025). An Agentic GenAI Framework for Automated Network Intrusion 
Detection and Incident Report Generation Using LangGraph and MITRE ATT&CK. 
CS 6349.501 Network Security, University of Texas at Dallas.
```
