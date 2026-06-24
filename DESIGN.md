# DESIGN.md -- VIKMO Dealer Assistant & Demand Forecasting

> Comprehensive architectural reasoning, design decisions, evaluation methodology, and failure analysis for the VIKMO AI/ML Intern Assignment.

---

## Table of Contents

- [Part A: Dealer Assistant](#part-a-dealer-assistant)
  - [1. Retrieval Approach (RAG)](#1-retrieval-approach-rag)
  - [2. Tool Design & Validation](#2-tool-design--validation)
  - [3. Prompt Engineering & Guardrails](#3-prompt-engineering--guardrails)
  - [4. Dual-Backend Architecture](#4-dual-backend-architecture)
  - [5. Conversation Management](#5-conversation-management)
- [Part B: Demand Forecasting](#part-b-demand-forecasting)
  - [6. Model Selection](#6-model-selection)
  - [7. Validation Scheme & Leakage Prevention](#7-validation-scheme--leakage-prevention)
  - [8. Metrics & Baseline Comparison](#8-metrics--baseline-comparison)
- [Evaluation Methodology](#evaluation-methodology)
  - [9. Current Evaluation Framework](#9-current-evaluation-framework)
  - [10. Advanced Evaluation Techniques](#10-advanced-evaluation-techniques)
  - [11. Known Failure Modes & Fixes](#11-known-failure-modes--fixes)
- [What I Would Do With More Time](#what-i-would-do-with-more-time)

---

## Part A: Dealer Assistant

---

### 1. Retrieval Approach (RAG)

#### Why RAG instead of prompt-stuffing the catalogue?

At 600 SKUs x ~150 tokens each ~= **90,000 tokens per request**. Even within a large context window, this is wasteful, expensive, and degrades model attention on relevant items. RAG also naturally scales: adding 10,000 more SKUs does not change prompt length.

#### Embedding model: `all-MiniLM-L6-v2`

| Property | Value | Rationale |
|----------|-------|-----------|
| Dimensionality | 384 | Trivially fast on CPU; ~1 MB for 600 vectors |
| Training objective | Semantic similarity (SBERT) | Better than language-model embeddings for retrieval |
| API cost | Zero | Open-source; runs locally without rate limits |
| vs. BM25 | Superior | Handles vocabulary mismatch ("brake pads" to "Brake Pad Set -- Bajaj Pulsar 150") |

#### Text representation per SKU

Each catalogue entry is linearised into a single string:

```
SKU: BRK-1042. Name: Brake Pad Set -- Bajaj Pulsar 150. Category: Brakes.
Brand: Bosch. Fits: Bajaj Pulsar 150. Price: Rs.450. Stock: 23 units.
```

**Key design choice:** `vehicle_fitment` included verbatim -- allows "brake pads for Pulsar 150" to directly align via cosine similarity without a separate sparse vehicle filter.

#### Index type: FAISS `IndexFlatIP` (exact cosine)

With 600 vectors, ANN indices (IVF, HNSW) offer zero meaningful speedup -- quantisation noise outweighs any latency benefit. Exact search is simultaneously faster and more accurate at this scale.

#### Index persistence

Built once on first run; loaded in <1 second subsequently.
- `.cache/faiss.index` -- the flat index
- `.cache/metadata.pkl` -- parallel SKU metadata array

#### Vehicle-specific retrieval (`find_parts_by_vehicle`)

Two-stage:
1. **Substring pre-filter**: entries whose `vehicle_fitment` contains query string
2. **Semantic re-ranking**: score filtered entries by embedding similarity
3. Return top-k (default 8)

More precise than pure semantic search: "universal" parts and unrelated vehicles do not pollute results.

---

### 2. Tool Design & Validation

#### Tool schema overview

| Tool | Trigger Intent | Returns |
|------|---------------|---------|
| `check_stock(sku)` | "check stock", "is X available" | `{sku, name, stock, status, price_inr}` |
| `create_order(dealer_name, items)` | "place order", "I will take N of X" | `OrderConfirmation` (Pydantic) |
| `find_parts_by_vehicle(vehicle_query, top_k)` | "parts for X", "what fits Y" | `{count, parts[]}` |

All schemas are OpenAI-compatible JSON -- the same schema works for both Groq and Gemini via a thin adapter.

#### Why Pydantic for `create_order`?

Enforces:
- SKU: non-empty string
- Quantity: positive integer (>= 1)
- Dealer name: non-empty string
- Items list: at least one entry

If validation fails, a structured error is returned rather than silently corrupting an order record. Critical in e-commerce contexts.

#### Tool dispatcher design

```python
def _dispatch(name: str, args: Dict) -> str:
    fn = _TOOL_MAP.get(name)
    if fn is None:
        return json.dumps({"success": False, "message": f"Unknown tool: {name}"})
    try:
        result = fn(**args)
    except Exception as exc:
        result = {"success": False, "message": str(exc)}
    return json.dumps(result, ensure_ascii=False, default=str)
```

All calls wrapped in try/except -- the LLM always receives structured JSON even on failure.

#### How the model selects tools

Uses tool descriptions + parameter schemas (not fine-tuning). System prompt reinforces:
- "Use `check_stock` before ordering if user has not confirmed stock"
- "Only call `create_order` after user has confirmed SKU and quantity"

Prevents premature order placement -- a critical guardrail in commerce contexts.

---

### 3. Prompt Engineering & Guardrails

#### System prompt principles

```
You are VIKMO Dealer Assistant -- an AI that helps motorcycle and automotive
dealers find auto parts, check inventory, and place orders.

RULES (follow strictly):
1. ONLY help with auto parts, vehicles, inventory, and ordering.
2. ALWAYS ground answers in the product data provided in context.
3. When user asks for parts without vehicle, ask for make and model.
4. When user wants to order, confirm SKU(s) and quantity first.
5. Use the tools -- check_stock, create_order, find_parts_by_vehicle.
6. When listing products show: SKU, Name, Price (INR), Stock status.
7. If stock is 0, say so clearly and suggest alternatives if available.
8. Keep responses concise, friendly, and professional.
9. If unsure, say so honestly -- do not guess.
```

**Design rationale for each rule:**

| Rule | Why it matters |
|------|---------------|
| #1 Domain restriction | Prevents persona drift to general assistant |
| #2 Grounding | Prevents hallucinated SKUs/prices not in catalogue |
| #3 Vehicle clarification | Reduces irrelevant results without vehicle context |
| #4 Order confirmation | Prevents accidental order creation |
| #5 Explicit tool mention | Nudges function calling for structured tasks |
| #6 Consistent format | Predictable output for downstream processing |
| #7 Out-of-stock honesty | Prevents false availability claims |
| #8 Tone | Professionalism for B2B dealer context |
| #9 Honesty | Reduces hallucination confidence |

#### Two-layer off-topic guard

**Layer 1 -- Pre-LLM keyword guard** (`_is_off_topic`):

```python
_OFF_TOPIC_KEYWORDS = {
    "weather", "news", "cricket", "movie", "recipe", "cook",
    "politics", "sports", "stock market", "crypto", "bitcoin",
    "joke", "poem", "story", "write code", "essay", "translate",
}

def _is_off_topic(text: str) -> bool:
    lower = text.lower()
    domain_hints = {"part", "brake", "tyre", "oil", "filter", "chain",
                    "clutch", "engine", "spark", "bike", "vehicle", ...}
    if any(h in lower for h in domain_hints):
        return False          # Domain keywords override off-topic
    return any(kw in lower for kw in _OFF_TOPIC_KEYWORDS)
```

Domain whitelisting prevents false positives (e.g., "calculate the price of spark plugs").

**Layer 2 -- System prompt instruction:** For edge cases that pass Layer 1 (e.g., "write a poem about brake pads"), the system prompt instructs the model to stay on-domain.

**Why not only Layer 2?** System prompt alone is insufficient -- models can be nudged by mixed-domain queries. Layer 1 catches obvious cases at zero LLM cost.

#### RAG context injection format

```
Relevant catalogue entries (ground your answer in these):
- SKU BRK-1042 | Brake Pad Set | INR 450 | Stock: 23 units | Fits: Bajaj Pulsar 150
- ...

User query: Do you have brake pads for a Bajaj Pulsar 150?
```

Instruction "ground your answer in these" reinforces factual grounding and reduces hallucination.

---

### 4. Dual-Backend Architecture

**Auto-selection logic:**
- `GROQ_API_KEY` set and non-placeholder -> Groq (priority)
- `GEMINI_API_KEY` set -> Gemini (fallback)
- Neither -> raise `EnvironmentError` with clear instructions

**Groq backend:**
- OpenAI-compatible SDK
- `parallel_tool_calls=False` -- sequential tool execution for predictable reasoning
- Up to 6 agentic tool-call rounds per turn
- Model: `meta-llama/llama-4-scout-17b-16e-instruct` (sub-second latency on free tier)

**Gemini backend:**
- Thin adapter converts OpenAI-style message history to Gemini `Contents/Parts` paradigm
- System prompt passed via `system_instruction` (native Gemini feature) for better adherence
- Tool schemas rebuilt as `FunctionDeclaration` objects

---

### 5. Conversation Management

History stored as OpenAI-style message list per session. On each turn:
1. RAG retrieves top-5 entries -> augmented message created
2. Full `[system] + history + [augmented user]` passed to LLM
3. **Original** (non-augmented) user message + reply appended to history

Storing original messages prevents context bloat from repeated RAG injections across turns.

**Known limitation:** Long conversations can grow beyond practical context limits. **Fix:** sliding-window history (keep last N turns) or LLM-based conversation summarisation.

---

## Part B: Demand Forecasting

---

### 6. Model Selection

| Model | Decision | Rationale |
|-------|----------|-----------|
| Naive Last Value | Baseline | Lower bound comparison |
| Seasonal Naive (4-week) | Baseline | Captures weekly periodicity |
| Rolling 4-week MA | Baseline | Standard retail smoothing |
| **Holt-Winters ES** | **Primary** | Level + trend + seasonality; interpretable; MLE fit on training data only |
| SARIMA(1,1,1)(1,0,1,4) | Optional | Autocorrelation modelling; ~10x slower per SKU |
| Prophet | Excluded | Heavy fbprophet/stan deps; overkill for 78-week series |
| XGBoost | Excluded | Requires feature engineering + CV infra; no clear win at this data size |
| LSTM/DeepAR | Excluded | Insufficient training data (78 weeks is well below minimum) |

**Why Holt-Winters as primary?**

Holt-Winters ES decomposes demand into three physically meaningful components:
- **Level (alpha)**: Current baseline dealer reorder rate
- **Trend (beta)**: Growing/declining dealer network
- **Seasonality (gamma)**: Festive season demand spikes (Diwali, etc.)

Parameters estimated via MLE on training data -- no manual tuning. `statsmodels` provides a reliable, battle-tested implementation on ~74 observations.

---

### 7. Validation Scheme & Leakage Prevention

#### Hold-out protocol

```
Full series (78 weeks): 2024-12-16 to 2026-06-08
                        |
        ________________|________________________________
        Training (~74 weeks)    |    Test (4 weeks)
  2024-12-16 -----------> 2026-05-12 | 2026-05-19 -> 2026-06-08
```

**Why 4 weeks?**
- Operationally meaningful: matches monthly dealer reorder cycle
- Sufficient to evaluate seasonal effects (one full 4-week period)
- Leaves ~74 training observations for reliable Holt-Winters estimation

#### Leakage prevention -- explicit checks

| Leakage Risk | How Avoided |
|-------------|-------------|
| Feature scaling on full series | No scaling applied (raw counts used) |
| Hyperparameter search using test data | Holt-Winters uses MLE on training data only |
| Random split | Date-based split only -- random splits leak future via autocorrelation |
| Model selection using test metrics | Baselines defined before test data is touched |

**Why date-based split matters:** For time-series, random splitting is fundamentally wrong -- a model trained on weeks 60+70 predicting week 65 has already "seen the future". All splits are strictly chronological.

---

### 8. Metrics & Baseline Comparison

**Primary: MAE**

```
MAE = (1/n) * sum(|actual_t - predicted_t|)
```

Preferred over RMSE because:
- RMSE heavily penalises large errors -- misleading when near-zero weeks exist
- MAE directly interpretable in units ("off by X units per week on average")
- Robust to outliers common in sparse auto-parts demand

**Secondary: MAPE**

Zero-actual weeks excluded (undefined). Reported with count of excluded weeks for transparency.

**Baseline comparison results:**
- Holt-Winters beats all baselines on average MAE across 30 SKUs
- Most pronounced improvement for SKUs with strong festive seasonality (Oct-Nov)
- For pure-noise Poisson SKUs, Rolling MA is competitive -- expected and documented

---

## Evaluation Methodology

---

### 9. Current Evaluation Framework

**15 test cases across 5 categories:**

| Category | Count | Tests |
|----------|-------|-------|
| `happy_path` | 6 | Normal queries: search, stock check, order, vehicle lookup |
| `clarification` | 2 | Under-specified queries (no vehicle given) |
| `multi_turn` | 2 | Context retention: clarify->answer, check->order |
| `out_of_scope` | 2 | Guardrail rejection: weather, cricket |
| `tricky` | 3 | Edge cases: invalid SKU, price from data, OOS check |

**Scoring rules (PASS if all hold):**
1. `must_contain` -- all phrases present (case-insensitive)
2. `must_contain_any` -- at least one phrase present (handles paraphrase)
3. `must_not_contain` -- none of the phrases appear
4. `tool_called` -- expected tool was actually invoked

**Tool tracking via monkey-patching:**

```python
def _tracked(name, fn):
    def wrapper(*args, **kwargs):
        _invocations.append(name)   # side-effect recording
        return fn(*args, **kwargs)  # original behaviour unchanged
    return wrapper
```

Non-intrusive: tool actual behaviour unchanged; only observes invocations.

---

### 10. Advanced Evaluation Techniques

#### A. Retrieval Quality Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Precision@k** | relevant_in_top_k / k | Fraction of top-k results that are relevant |
| **Recall@k** | relevant_in_top_k / total_relevant | Coverage of relevant docs in top-k |
| **MRR** | 1 / rank_of_first_relevant | How high the first relevant result ranks |
| **NDCG@k** | DCG / IDCG | Quality-weighted ranking metric |
| **Hit Rate** | 1 if any relevant in top-k else 0 | Binary retrieval adequacy |

For VIKMO: create a golden retrieval set (query -> expected SKU list) and evaluate the FAISS retriever.

#### B. RAGAS Framework (Generation Quality)

RAGAS uses LLM judges (GPT-4 / Gemini) to score:

| RAGAS Metric | What It Evaluates |
|-------------|-------------------|
| **Faithfulness** | Does response contain only facts from retrieved context? |
| **Answer Relevancy** | Does response actually answer the question? |
| **Context Precision** | How much of retrieved context is relevant? |
| **Context Recall** | How much of the needed information was retrieved? |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
print(result)  # {faithfulness: 0.92, answer_relevancy: 0.88, context_precision: 0.79}
```

RAGAS is far more robust than string matching -- handles paraphrase, partial answers, and hallucination depth.

#### C. LLM-as-Judge

Use a stronger LLM (GPT-4o / Gemini 1.5 Pro) to score responses:

```python
JUDGE_PROMPT = """
Evaluate this auto-parts assistant response on 4 dimensions (1-5 scale):

User query: {query}
Retrieved context: {context}
Assistant response: {response}

1. Factual accuracy: Does it match the catalogue data?
2. Helpfulness: Does it resolve the user's need?
3. Format compliance: Does it include SKU, name, price, stock?
4. Guardrail adherence: Does it stay on the auto-parts domain?

Output JSON: {{"accuracy": N, "helpfulness": N, "format": N, "guardrail": N, "reasoning": "..."}}
"""
```

**Advantages over string matching:**
- Catches semantic equivalence ("24 units" = "twenty-four units")
- Evaluates tone and professionalism
- Scores hallucination depth (invented SKU vs. slightly wrong price)
- Handles varied but correct phrasings

#### D. Tool Use Accuracy Metrics

Extended beyond binary tool-called check:

| Metric | How to Measure |
|--------|---------------|
| **Tool Selection Accuracy** | Expected tool in invocation list? (current approach) |
| **Argument Accuracy** | Correct SKU / dealer name / quantity passed? |
| **Tool Refusal Rate** | For out-of-scope queries, did model avoid calling tools? |
| **Over-calling Rate** | Did model call tools unnecessarily (RAG would have sufficed)? |

**Argument-level evaluation:**

```python
def check_tool_arguments(expected_args: dict, actual_args: dict) -> dict:
    return {
        "sku_correct":    actual_args.get("sku") == expected_args.get("sku"),
        "qty_correct":    actual_args.get("quantity") == expected_args.get("quantity"),
        "dealer_correct": actual_args.get("dealer_name") == expected_args.get("dealer_name"),
    }
```

#### E. Hallucination Detection

SKU hallucination proxy:

```python
import re

def check_hallucination(response: str, retrieved_skus: set) -> dict:
    sku_pattern = re.compile(r'\b[A-Z]{2,5}-\d{3,5}\b')
    mentioned   = set(sku_pattern.findall(response))
    hallucinated = mentioned - retrieved_skus
    return {
        "hallucinated_skus":  list(hallucinated),
        "hallucination_rate": len(hallucinated) / max(len(mentioned), 1),
    }
```

#### F. Multi-Turn Coherence Metrics

| Metric | What It Measures |
|--------|-----------------|
| Entity retention | Is the SKU from turn 1 correctly used in turn 2? |
| Clarification efficiency | How many turns to resolve ambiguous queries? |
| History utilisation | Does later-turn response correctly reference earlier context? |

#### G. Statistical Significance

When comparing model variants (Groq vs. Gemini, prompt A vs. B):

- **McNemar's test** for paired binary pass/fail outcomes
- **Bootstrap confidence intervals** for aggregate metrics (pass rate, MAE)
- **Multiple random seeds** for stochastic LLM output variance

---

### 11. Known Failure Modes & Fixes

| Failure Mode | Root Cause | Recommended Fix |
|-------------|-----------|-----------------|
| Missing clarifying question | RAG retrieves plausible results; model answers without asking | Post-retrieval ambiguity check; enforce vehicle for brake/tyre/filter categories |
| Wrong tool arguments | LLM maps natural language to malformed SKU format | SKU normalisation in dispatcher: `re.sub(r'[^A-Z0-9-]', '', sku.upper())` |
| Off-topic passing through | Mixed-domain queries contain domain hints ("poem about brake pads") | Semantic off-topic classifier (zero-shot) instead of keyword list |
| Multi-turn context loss | Long conversations exceed practical context | Sliding-window history (last N turns) or LLM-based summarisation |
| Hallucinated prices | Model invents prices not in catalogue | Post-generation validation: reject if price not found in retrieved context |
| Tool not called | Model answers from RAG instead of calling check_stock for explicit SKU | Explicit system prompt rule: "Always call check_stock when user explicitly mentions a SKU" |
| Order double-confirmation | Model asks for confirmation even when user is explicit | Detect explicit confirmation phrases in user input; skip re-confirmation |

---

## What I Would Do With More Time

### Retrieval improvements
- **Hybrid search**: BM25 (exact SKU / brand lookup) + SBERT (natural language) with weighted fusion
- **Cross-encoder re-ranker**: Rank top-20 semantic hits -> top-5 for higher precision
- **Query expansion**: "brake pads" -> ["brake pad set", "disc brake pads", "ceramic brake pads"]

### Agent improvements
- **Streaming responses**: Token-by-token LLM output for better UX (Streamlit `st.write_stream`)
- **get_my_orders tool**: Order history lookup for repeat purchasers
- **Bulk catalogue browser**: `find_parts_by_category(category, brand)` for power users
- **SKU normalisation**: Pre-process all SKU inputs before tool dispatch

### Evaluation improvements
- **Full RAGAS integration**: Faithfulness + context precision + answer relevancy at scale
- **LLM-as-judge scoring**: GPT-4o scoring for nuanced quality dimensions
- **Golden retrieval dataset**: Manually annotated (query -> expected SKU set) for retrieval metrics
- **CI/CD regression gate**: Automated eval on every code change; block PRs that drop pass rate

### Forecasting improvements
- **Walk-forward cross-validation**: Multiple test windows for more robust MAE estimates
- **Ensemble forecasting**: Weighted average of Holt-Winters + SARIMA + Seasonal Naive
- **External signals**: Festive calendar flags, holiday indicators
- **Per-SKU model selection**: Auto-select best model per SKU via AIC/BIC comparison
- **Prediction intervals**: Confidence bands for inventory risk planning

### Production readiness
- **Multi-user session isolation**: Per-session state in Streamlit
- **Structured observability logging**: JSON logs of every tool call, retrieval, and LLM response
- **Hash-based cache invalidation**: Automatic FAISS rebuild when catalogue.csv changes
- **A/B testing framework**: Serve two prompt variants to split traffic and compare live metrics

---

*Document version: June 2026 -- VIKMO AI/ML Intern Assignment*
