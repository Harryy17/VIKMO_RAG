"""
assistant/agent.py
──────────────────
Agentic loop supporting two LLM backends:
  * Google Gemini  (via google-genai SDK) -- set GEMINI_API_KEY
  * Groq           (via groq SDK)         -- set GROQ_API_KEY

Backend auto-selection:
  1. GROQ_API_KEY set  -> Groq  (model: meta-llama/llama-4-scout-17b-16e-instruct)
  2. GEMINI_API_KEY    -> Gemini (model: gemini-2.5-flash)

Architecture:
  Each user turn:
    a. RAG retrieval injects top-5 catalogue entries into context.
    b. LLM generates a response (may include tool calls).
    c. Tool calls execute and results feed back to LLM.
    d. Final text reply returned.

Guardrails:
  * System prompt restricts to auto-parts domain.
  * Off-topic keyword guard refuses unrelated queries.
  * LLM grounded on retrieved catalogue -- no invented data.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

from assistant.retrieval import init_retrieval, search
from assistant.tools import check_stock, create_order, find_parts_by_vehicle

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are VIKMO Dealer Assistant -- an AI that helps motorcycle and "
    "automotive dealers find auto parts, check inventory, and place orders.\n\n"
    "RULES (follow strictly):\n"
    "1. You ONLY help with auto parts, vehicles, inventory, and ordering. Politely refuse anything else.\n"
    "2. ALWAYS ground your answers in the product data provided in the context. Never invent SKUs, prices, or stock levels.\n"
    "3. When a user asks for parts but has not specified any vehicle make/model, ask for the make and model before answering. If they specified a vehicle make/model but it is not in the catalogue, do not ask again; instead, suggest similar vehicles/alternatives from the retrieved context.\n"
    "4. When a user wants to order, confirm the SKU(s) and quantity before calling create_order. When an order is successfully placed, you MUST always output the Order ID (which starts with 'ORD-') and the Dealer Name in your response, and use the word 'confirmed' or 'order confirmed'.\n"
    "5. Use the tools -- check_stock, create_order, find_parts_by_vehicle.\n"
    "6. When listing products, always show: SKU, Name, Price (always use the ₹ symbol, e.g. ₹365), Stock status.\n"
    "7. If stock is 0, say so clearly and suggest alternatives if available.\n"
    "8. Keep responses concise, friendly, and professional.\n"
    "9. If you are unsure or the exact product/vehicle is missing, do not say 'I don't know the price' or 'I am not sure'; instead, clearly state that the exact model is not available and suggest the closest compatible matches from the retrieved context using their real prices with the ₹ symbol.\n"
    "10. CRITICAL: If the user explicitly requests to place an order (e.g., 'Order 5 of FAKE-9999 for XYZ Dealers'), you MUST call the create_order tool immediately. Do not pre-validate or refuse the order; simply execute the create_order tool with the provided SKU. The tool itself will return the appropriate validation/invalid error which you should report.\n"
)

# ---------------------------------------------------------------------------
# Off-topic guard
# ---------------------------------------------------------------------------
_OFF_TOPIC_KEYWORDS = {
    "weather", "news", "cricket", "movie", "recipe", "cook",
    "politics", "sports", "stock market", "share price", "crypto",
    "bitcoin", "joke", "poem", "story", "write code", "essay",
    "translate", "math", "calculate",
}

def _is_off_topic(text: str) -> bool:
    lower = text.lower()
    domain_hints = {
        "part", "brake", "tyre", "oil", "filter", "chain", "clutch",
        "engine", "spark", "bike", "vehicle", "motor", "order", "stock",
        "price", "sku", "catalogue", "bajaj", "honda", "yamaha", "ktm",
        "suzuki", "tvs", "hero", "royal", "pulsar", "duke", "splendor",
    }
    if any(h in lower for h in domain_hints):
        return False
    return any(kw in lower for kw in _OFF_TOPIC_KEYWORDS)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
_TOOL_MAP: Dict[str, Any] = {
    "check_stock":           check_stock,
    "create_order":          create_order,
    "find_parts_by_vehicle": find_parts_by_vehicle,
}

# OpenAI-compatible tool schema (used by both Groq and Gemini wrappers)
_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Check the current stock level and price for a specific product SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "Product SKU, e.g. BRK-1042"}
                },
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": (
                "Place a confirmed order for a dealer. "
                "Only call this after the user has confirmed SKU(s) and quantity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dealer_name": {"type": "string", "description": "Name of the dealer"},
                    "items": {
                        "type": "array",
                        "description": "Line items to order",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku":      {"type": "string",  "description": "Product SKU"},
                                "quantity": {"type": "integer", "description": "Units to order"},
                            },
                            "required": ["sku", "quantity"],
                        },
                    },
                },
                "required": ["dealer_name", "items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_parts_by_vehicle",
            "description": "Find auto parts compatible with a given vehicle make/model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_query": {
                        "type": "string",
                        "description": "Vehicle make and model, e.g. 'Bajaj Pulsar 150'",
                    },
                },
                "required": ["vehicle_query"],
            },
        },
    },
]


def _dispatch(name: str, args: Dict) -> str:
    fn = _TOOL_MAP.get(name)
    if fn is None:
        return json.dumps({"success": False, "message": f"Unknown tool: {name}"})
    try:
        result = fn(**args)
    except Exception as exc:
        result = {"success": False, "message": str(exc)}
    return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Groq backend
# ---------------------------------------------------------------------------
class _GroqBackend:
    def __init__(self, api_key: str, model: str) -> None:
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._model  = model
        print(f"[Agent] Groq backend -> {model}")

    def generate(self, messages: List[Dict]) -> str:
        msgs: List[Dict] = list(messages)
        MAX_ROUNDS = 6

        for _ in range(MAX_ROUNDS):
            resp = self._client.chat.completions.create(
                model               = self._model,
                messages            = msgs,
                tools               = _TOOL_SCHEMAS,
                tool_choice         = "auto",
                parallel_tool_calls = False,
            )
            choice = resp.choices[0]
            msg    = choice.message

            # --- No tool calls: return the text reply ----------------------
            if not msg.tool_calls:
                return msg.content or ""

            # --- Tool calls: execute and continue --------------------------
            # Append assistant turn (must include tool_calls in dict form)
            msgs.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": [
                    {
                        "id":   tc.id,
                        "type": "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Append tool results
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                msgs.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      _dispatch(tc.function.name, args),
                })

        return "I was unable to generate a response. Please try again."


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------
class _GeminiBackend:
    def __init__(self, api_key: str, model: str) -> None:
        from google import genai
        from google.genai import types as gt
        self._gt     = gt
        self._client = genai.Client(api_key=api_key)
        self._model  = model
        self._gtools = self._build_tools()
        print(f"[Agent] Gemini backend -> {model}")

    def _build_tools(self):
        gt = self._gt
        return [gt.Tool(function_declarations=[
            gt.FunctionDeclaration(
                name="check_stock",
                description="Check stock level and price for a product SKU.",
                parameters=gt.Schema(
                    type=gt.Type.OBJECT,
                    properties={"sku": gt.Schema(type=gt.Type.STRING)},
                    required=["sku"],
                ),
            ),
            gt.FunctionDeclaration(
                name="create_order",
                description="Place a confirmed order. Only call after user confirms.",
                parameters=gt.Schema(
                    type=gt.Type.OBJECT,
                    properties={
                        "dealer_name": gt.Schema(type=gt.Type.STRING),
                        "items": gt.Schema(
                            type=gt.Type.ARRAY,
                            items=gt.Schema(
                                type=gt.Type.OBJECT,
                                properties={
                                    "sku":      gt.Schema(type=gt.Type.STRING),
                                    "quantity": gt.Schema(type=gt.Type.INTEGER),
                                },
                                required=["sku", "quantity"],
                            ),
                        ),
                    },
                    required=["dealer_name", "items"],
                ),
            ),
            gt.FunctionDeclaration(
                name="find_parts_by_vehicle",
                description="Find parts compatible with a vehicle.",
                parameters=gt.Schema(
                    type=gt.Type.OBJECT,
                    properties={
                        "vehicle_query": gt.Schema(type=gt.Type.STRING),
                    },
                    required=["vehicle_query"],
                ),
            ),
        ])]

    def generate(self, messages: List[Dict]) -> str:
        gt = self._gt

        # Convert OpenAI-style messages -> Gemini Contents
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue  # handled via system_instruction
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(gt.Content(
                role=role,
                parts=[gt.Part(text=m.get("content") or "")],
            ))

        cfg = gt.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            tools=self._gtools,
        )

        MAX_ROUNDS = 6
        resp = self._client.models.generate_content(
            model=self._model, contents=contents, config=cfg
        )

        for _ in range(MAX_ROUNDS):
            fn_calls = []
            if resp.candidates:
                for part in resp.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fn_calls.append(part.function_call)

            if not fn_calls:
                break

            contents.append(resp.candidates[0].content)
            tool_parts = []
            for fc in fn_calls:
                out = _dispatch(fc.name, dict(fc.args))
                tool_parts.append(gt.Part.from_function_response(
                    name=fc.name, response={"result": out}
                ))
            contents.append(gt.Content(role="user", parts=tool_parts))
            resp = self._client.models.generate_content(
                model=self._model, contents=contents, config=cfg
            )

        try:
            return resp.text or ""
        except Exception:
            return "I was unable to generate a response. Please try again."


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------
def _build_backend():
    groq_key    = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_key  = os.environ.get("GEMINI_API_KEY", "").strip()
    groq_model  = os.environ.get("GROQ_MODEL",   "meta-llama/llama-4-scout-17b-16e-instruct")
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    if groq_key and groq_key != "your_groq_key_here":
        return _GroqBackend(groq_key, groq_model)
    if gemini_key:
        return _GeminiBackend(gemini_key, gemini_model)
    raise EnvironmentError(
        "No LLM API key found.\n"
        "Set GROQ_API_KEY (free at console.groq.com) or GEMINI_API_KEY in .env"
    )


# ---------------------------------------------------------------------------
# Public agent class
# ---------------------------------------------------------------------------
class DealerAssistant:
    """
    Stateful conversational Dealer Assistant.

    Usage:
        agent = DealerAssistant()
        reply = agent.chat("Brake pads for Bajaj Pulsar 150?")
    """

    def __init__(self) -> None:
        init_retrieval()
        self._backend = _build_backend()
        self._history: List[Dict] = []   # OpenAI-style message list

    def reset(self) -> None:
        self._history = []

    def chat(self, user_message: str) -> str:
        # Off-topic guard
        if _is_off_topic(user_message):
            return (
                "I'm VIKMO Dealer Assistant and I specialise in auto parts, "
                "inventory, and ordering. I can't help with that topic. "
                "Is there anything auto-parts related I can assist you with?"
            )

        # RAG: retrieve top-5 relevant catalogue entries
        hits = search(user_message, top_k=5)
        if hits:
            lines = ["Relevant catalogue entries (ground your answer in these):"]
            for r in hits:
                stock = f"{r['stock']} units" if r["stock"] > 0 else "OUT OF STOCK"
                lines.append(
                    f"- SKU {r['sku']} | {r['name']} | INR {r['price_inr']} "
                    f"| Stock: {stock} | Fits: {r['vehicle_fitment']}"
                )
            augmented = "\n".join(lines) + f"\n\nUser query: {user_message}"
        else:
            augmented = user_message

        # Build full message list
        messages = (
            [{"role": "system", "content": _SYSTEM_PROMPT}]
            + self._history
            + [{"role": "user", "content": augmented}]
        )

        # Call backend
        reply = self._backend.generate(messages).strip()
        if not reply:
            reply = "I'm sorry, I couldn't generate a response. Please try again."

        # Update history with original (not augmented) user message
        self._history.append({"role": "user",      "content": user_message})
        self._history.append({"role": "assistant",  "content": reply})

        return reply

    @property
    def history(self) -> List[Dict]:
        return self._history
