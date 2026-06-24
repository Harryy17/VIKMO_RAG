# 🔧 VIKMO Dealer Assistant & Demand Forecasting

> **AI / ML Intern Take-Home Assignment**
> A production-grade conversational AI assistant for auto-parts dealers, powered by dual-backend LLM (Groq / Gemini), semantic RAG retrieval, structured tool calling, and Holt-Winters demand forecasting.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama--4--Scout-F55036?style=for-the-badge&logo=meta&logoColor=white)
![Gemini](https://img.shields.io/badge/Google-Gemini%202.5-4285F4?style=for-the-badge&logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-00A98F?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Running the Assistant](#-running-the-assistant)
- [Evaluation](#-evaluation)
- [Demand Forecasting](#-demand-forecasting)
- [Example Interactions](#-example-interactions)
- [Configuration](#-configuration)
- [Assumptions & Limitations](#-assumptions--limitations)

---

## 🎯 Overview

This project implements two end-to-end ML/AI systems over VIKMO's auto-parts catalogue:

| Part | System | Description | Status |
|------|--------|-------------|--------|
| **A** | Conversational Dealer Assistant | RAG + LLM + Structured Tool Calling | ✅ Core |
| **B** | Demand Forecasting | Holt-Winters ES + Baseline Models | ✅ Bonus |
| **+** | Streamlit Chat UI | Real-time conversational interface | ✅ Bonus |
| **+** | Guardrails | Off-topic detection (keyword + prompt-level) | ✅ Bonus |
| **+** | Dual LLM Backend | Groq (Llama-4) **or** Google Gemini | ✅ Bonus |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     VIKMO Dealer Assistant                      │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────┐     ┌────────────────────────────────┐   │
│  │  Off-Topic Guard │────▶│  RAG Retrieval (FAISS + SBERT) │   │
│  │  (keyword regex) │ no  │  top-5 catalogue entries       │   │
│  └──────────────────┘     └────────────┬───────────────────┘   │
│      │ yes                             │                        │
│      ▼                                 ▼                        │
│  Hard Refuse            ┌─────────────────────────────────┐    │
│                         │  Augmented Prompt                │    │
│                         │  [system prompt + RAG context    │    │
│                         │   + conversation history +       │    │
│                         │   user query]                    │    │
│                         └────────────┬────────────────────┘    │
│                                      │                          │
│                         ┌────────────▼────────────────────┐    │
│                         │   LLM Backend (auto-select)      │    │
│                         │   ┌──────────┐  ┌────────────┐  │    │
│                         │   │  Groq    │  │  Gemini    │  │    │
│                         │   │ Llama-4  │  │ 2.5 Flash  │  │    │
│                         │   └──────────┘  └────────────┘  │    │
│                         └────────────┬────────────────────┘    │
│                                      │                          │
│                         ┌────────────▼────────────────────┐    │
│                         │     Tool Dispatcher              │    │
│                         │  ┌───────────┐ ┌─────────────┐  │    │
│                         │  │check_stock│ │create_order │  │    │
│                         │  └───────────┘ └─────────────┘  │    │
│                         │  ┌──────────────────────────┐   │    │
│                         │  │ find_parts_by_vehicle     │   │    │
│                         │  └──────────────────────────┘   │    │
│                         └────────────┬────────────────────┘    │
│                                      │                          │
│                         ┌────────────▼────────────────────┐    │
│                         │      Final Reply + History       │    │
│                         └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Key flows:**
- **RAG**: Sentence-BERT embeds query → FAISS retrieves top-5 → injected as grounding context
- **Tool Calling**: LLM emits structured function calls → dispatcher validates & executes → result fed back → final answer
- **Multi-turn**: Full conversation history maintained per session; original (non-augmented) messages stored

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| **LLM (primary)** | Groq — `meta-llama/llama-4-scout-17b-16e-instruct` | Free, sub-second latency, OpenAI-compatible API |
| **LLM (fallback)** | Google Gemini 2.5 Flash | Free tier, native function calling support |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, fast on CPU, strong semantic similarity |
| **Vector Search** | FAISS `IndexFlatIP` (exact cosine) | Optimal for 600 vectors; no ANN quantisation noise |
| **Validation** | Pydantic v2 | Schema enforcement on LLM-generated tool arguments |
| **Forecasting** | Statsmodels Holt-Winters ES | Handles trend + seasonality; interpretable; fast |
| **UI** | Streamlit | Rapid prototyping, session-state management |
| **CLI** | Rich | Formatted terminal output with tables & panels |

---

## 📁 Project Structure

```
VIKMO_AI_ML_Intern_Assignment/
│
├── README.md                   ← You are here
├── DESIGN.md                   ← Architecture decisions & evaluation methodology
├── DATA_README.md              ← Dataset description
├── requirements.txt
├── .env.example                ← Template for environment variables
│
├── assistant/                  ← Core AI agent
│   ├── __init__.py
│   ├── retrieval.py            # FAISS RAG pipeline (embed, index, search)
│   ├── tools.py                # check_stock, create_order, find_parts_by_vehicle
│   ├── agent.py                # Dual-backend agentic loop (Groq / Gemini)
│   └── main.py                 # CLI entry point (Rich-formatted REPL)
│
├── eval/
│   ├── eval_set.json           # 15 test cases across 5 categories
│   └── run_eval.py             # Automated evaluation with monkey-patched tool tracking
│
├── forecasting/
│   ├── forecast.py             # Holt-Winters + baselines + optional SARIMA
│   ├── results.csv             # Per-SKU 4-week predictions (generated)
│   └── summary.csv             # Model comparison table (generated)
│
├── ui/
│   └── app.py                  # Streamlit chat interface
│
├── .cache/                     # Auto-generated FAISS index cache
│   ├── faiss.index
│   └── metadata.pkl
│
├── catalogue.csv               # 600 auto-parts SKUs (primary data source)
├── sales_history.csv           # 78 weeks of weekly sales per SKU
└── orders_db.json              # Live order records (generated at runtime)
```

---

## ⚡ Quick Start

### 1. Clone & install dependencies

```bash
cd VIKMO_AI_ML_Intern_Assignment
pip install -r requirements.txt
```

### 2. Configure API key

Copy the example file and fill in your key:

```bash
copy .env.example .env
```

Edit `.env`:

```env
# Option A — Groq (recommended: free, fastest)
GROQ_API_KEY=your_groq_key_here      # Get at: https://console.groq.com/

# Option B — Google Gemini (fallback)
GEMINI_API_KEY=your_gemini_key_here  # Get at: https://aistudio.google.com/

# Optional overrides
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GEMINI_MODEL=gemini-2.5-flash
```

> **Backend selection logic:** If `GROQ_API_KEY` is set → Groq is used. Otherwise → Gemini. The FAISS index is built automatically on first run and cached to `.cache/` for sub-second subsequent startups.

---

## 🤖 Running the Assistant

### Option A — Streamlit UI (recommended)

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501` with a full chat interface, session history, and sidebar controls.

### Option B — CLI (great for testing)

```bash
python -m assistant.main
# or
python assistant/main.py
```

Rich-formatted REPL with coloured output. Type `quit` or `exit` to stop.

---

## 🧪 Evaluation

### Run the automated test suite

```bash
python eval/run_eval.py
```

**What it tests (15 cases across 5 categories):**

| Category | Count | Description |
|----------|-------|-------------|
| `happy_path` | 6 | Normal queries: product search, stock check, order placement |
| `clarification` | 2 | Ambiguous queries requiring vehicle make/model follow-up |
| `multi_turn` | 2 | Context retention across turns (clarify → answer; check → order) |
| `out_of_scope` | 2 | Guardrail effectiveness (weather, cricket) |
| `tricky` | 3 | Edge cases: invalid SKU, price from data, out-of-stock check |

**Scoring rules (a test PASSES if all hold):**
1. `must_contain` — all phrases present in response (case-insensitive)
2. `must_contain_any` — at least one phrase present (if field is set)
3. `must_not_contain` — none of the phrases appear
4. `tool_called` — expected tool was actually invoked (tracked via monkey-patching)

**Output:**
- `eval/results.json` — full per-test JSON with response text, checks, and tool traces
- Rich table to stdout — pass/fail per category + overall rate + tool accuracy

### Evaluation Techniques Used

| Technique | Implementation | Purpose |
|-----------|---------------|---------|
| **String-match scoring** | `must_contain` / `must_not_contain` | Fast, deterministic pass/fail |
| **Any-match scoring** | `must_contain_any` | Handles paraphrase variation |
| **Tool invocation tracking** | Monkey-patching tool functions | Validates correct tool selection |
| **Category-level analysis** | Grouped pass rates | Identifies weak capability areas |
| **Failure mode reporting** | Per-fail detailed output | Actionable debugging |
| **Multi-turn simulation** | Sequential turn replay per test | Tests context retention |

> See [DESIGN.md](DESIGN.md) for a full evaluation methodology discussion including known failure modes and recommended improvements (LLM-as-judge, RAGAS metrics, semantic similarity scoring).

---

## 📈 Demand Forecasting

### Run forecasting

```bash
# Holt-Winters (fast, recommended)
python forecasting/forecast.py

# With optional SARIMA (slower, more accurate for trending SKUs)
python forecasting/forecast.py --sarima
```

**Models evaluated:**

| Model | Type | Notes |
|-------|------|-------|
| Naive Last Value | Baseline | Lower bound comparison |
| Seasonal Naive (4-week) | Baseline | Captures weekly pattern |
| Rolling 4-week MA | Baseline | Standard retail smoothing |
| **Holt-Winters ES** | **Primary** | Level + trend + seasonality |
| SARIMA(1,1,1)(1,0,1,4) | Optional | Autocorrelation modelling |

**Validation protocol:**
- **Hold-out**: last 4 weeks (2026-05-19 → 2026-06-08) — ~5% of 78-week series
- **Training**: all prior weeks (~74 observations per SKU)
- **Metrics**: MAE (primary), MAPE (secondary; zero-actual weeks excluded)
- **Leakage prevention**: date-based split, no test-window hyperparameter tuning

**Output files:**
- `forecasting/results.csv` — per-SKU MAE/MAPE for all models
- `forecasting/summary.csv` — aggregate model comparison

---

## 💬 Example Interactions

```
┌─────────────────────────────────────────────────────────┐
│  Product Discovery                                       │
│                                                         │
│  You: Do you have brake pads for a Bajaj Pulsar 150?    │
│  VIKMO: Yes! Here are brake pads for the Pulsar 150:    │
│    • SKU BRK-1042 | Brake Pad Set | ₹450 | 23 units    │
│    • SKU BRK-1043 | Ceramic Brake Pad | ₹620 | 8 units  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Stock Check (Tool Call)                                 │
│                                                         │
│  You: Check stock for BRK-1042                          │
│  VIKMO: [calls check_stock("BRK-1042")]                 │
│    BRK-1042 — Brake Pad Set — Bajaj Pulsar 150          │
│    Stock: 23 units ✅ In Stock | Price: ₹450            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Order Placement (Tool Call)                             │
│                                                         │
│  You: Place an order for 5 units for Sharma Auto Parts. │
│  VIKMO: [calls create_order]                            │
│    ✅ Order Confirmed!                                   │
│    Order ID: ORD-A3F2B1C9                               │
│    Dealer: Sharma Auto Parts                            │
│    Items: 5x BRK-1042 @ ₹450 = ₹2,250                 │
│    Total: ₹2,250                                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Guardrail — Off-Topic Rejection                         │
│                                                         │
│  You: What's the weather today?                         │
│  VIKMO: I'm VIKMO Dealer Assistant and I specialise in  │
│    auto parts, inventory, and ordering. I'm not able    │
│    to help with that topic.                             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Multi-Turn Context Retention                            │
│                                                         │
│  You: I need an air filter.                             │
│  VIKMO: Could you tell me the vehicle make and model?   │
│  You: It's for a Honda Activa 6G.                       │
│  VIKMO: Here are air filters for the Honda Activa 6G:  │
│    • SKU FIL-2201 | Air Filter | ₹180 | 45 units       │
└─────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (takes priority if set) |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GROQ_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq model to use |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model to use |

---

## ⚠️ Assumptions & Limitations

| Assumption | Detail |
|-----------|--------|
| **Live stock mutation** | Stock is decremented when orders are placed; persisted to `orders_db.json` |
| **Index invalidation** | Delete `.cache/` to force FAISS index rebuild when `catalogue.csv` changes |
| **MAPE denominator** | Weeks with zero actual sales are excluded from MAPE (undefined percentage) |
| **Context window** | Long conversations may drift; no summarisation/truncation implemented |
| **Rate limiting** | Free-tier LLMs may throttle; 0.5s sleep between eval turns to mitigate |
| **Multi-user** | No session isolation in current Streamlit deployment; suitable for demo only |

---

## 📖 Further Reading

- [DESIGN.md](DESIGN.md) — Full architectural reasoning, prompt design, guardrails, failure analysis, and evaluation methodology
- [DATA_README.md](DATA_README.md) — Dataset schema and statistics

---

<div align="center">
  <sub>Built for VIKMO AI/ML Intern Assignment · June 2026</sub>
</div>
