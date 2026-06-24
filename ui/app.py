"""
ui/app.py
─────────
Streamlit Chat UI for VIKMO Dealer Assistant – Light / Branded Edition.

Run:
    streamlit run ui/app.py

Requires GEMINI_API_KEY to be set in environment or .env file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VIKMO Dealer Assistant",
    page_icon="🟨",
    layout="centered",
)

# ── Custom CSS (VIKMO Brand Theme - Yellow & Black) ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

/* VIKMO Brand Colors */
:root {
    --vikmo-yellow: #FFD700; /* Matches the logo background */
    
    --vikmo-light: #FAFAFA;
}

/* White background, black text */
.stApp {
    background: #ffffff;
    color: #000000;
    min-height: 100vh;
}

/* --- VIKMO HEADER --- */
.vikmo-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem 0 1.5rem 0;
    border-bottom: 2px solid #f0f0f0;
    margin-bottom: 1rem;
}
.vikmo-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    
    margin: 0;
    line-height: 1;
}
.vikmo-title span {
    color: var(--vikmo-yellow);
    background: transparent;
    padding: 0 8px;
    border-radius: 4px;
    display: inline-block;
}
}
.vikmo-subtitle {
    color: #555555;
    font-size: 1rem;
    font-weight: 400;
    margin-top: 0.25rem;
    letter-spacing: 0.05em;
}

/* Chat messages – Clean Card Style */
.stChatMessage {
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
    background: #ffffff !important;
    border: 1px solid #e8e8e8 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    padding: 0.75rem 1rem !important;
}

/* Assistant bubble – Yellow left border + subtle yellow tint */
[data-testid="stChatMessage"][data-message-author-role="assistant"] {
    border-left: 4px solid var(--vikmo-yellow) !important;
    background: #fefcf5 !important;
}

/* User bubble – Dark left border */
[data-testid="stChatMessage"][data-message-author-role="user"] {
    border-left: 4px solid #000000 !important;
}

/* Chat input */
.stChatInputContainer {
    background: #ffffff !important;
    border: 1px solid #d0d0d0 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03) !important;
}

/* Status pill */
.status-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
    background: #000;
    color: var(--vikmo-yellow);
    border: 1px solid #000;
}

/* Metric cards – Brand matched */
.metric-card {
    background: #fdfdfd;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin: 0.5rem 0;
    transition: border 0.2s;
}
.metric-card:hover {
    border-color: var(--vikmo-yellow);
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #000;
}
.metric-label {
    font-size: 0.8rem;
    color: #555555;
    margin-top: 2px;
}

/* Sidebar – Pure white */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8e8e8 !important;
}
[data-testid="stSidebarContent"] {
    background: #ffffff !important;
}

/* Buttons – Yellow outline on white, Yellow fill on hover */
.stButton > button {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1.5px solid var(--vikmo-yellow) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--vikmo-yellow) !important;
    color: #000000 !important;
    border-color: var(--vikmo-yellow) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(255, 215, 0, 0.2);
}

/* Divider */
hr { border-color: #e0e0e0 !important; }

/* Chat history in sidebar */
.chat-history-item {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    color: #333333;
    line-height: 1.4;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
.chat-history-item .role-user {
    font-weight: 700;
    color: #000;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
}
.chat-history-item .role-assistant {
    font-weight: 700;
    color: #000;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
    border-bottom: 1px solid var(--vikmo-yellow);
    display: inline-block;
}
.chat-history-scroll {
    max-height: 320px;
    overflow-y: auto;
    padding-right: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state (init early so sidebar can read it) ─────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
_LOGO_PATH = Path(__file__).parent / "vikmo_logo.png"

with st.sidebar:
    # ─── SIDEBAR LOGO ─────────────────────────────────────────────────────
    st.markdown('<div style="text-align:center; padding: 0.5rem 0 0.25rem 0;">', unsafe_allow_html=True)
    if _LOGO_PATH.exists():
        st.image(str(_LOGO_PATH), width=80)
    else:
        # Fallback: text logo when image not found
        st.markdown(
            '<div style="font-size:2rem; font-weight:900; background:#000; color:#FFD700; '
            'display:inline-block; padding:4px 14px; border-radius:6px; margin-bottom:4px;">VIKMO</div>',
            unsafe_allow_html=True
        )
    st.markdown(
        '<div style="font-size:1.2rem; font-weight:800; color:#000; margin-top:6px;">VIKMO</div>'
        '<div style="color:#555; font-size:0.8rem;">Dealer Assistant</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("**🛠 Quick Tools**")

    example_queries = [
        "🔍 Brake pads for Bajaj Pulsar 150",
        "📦 Check stock for BRK-1042",
        "🚗 Parts for KTM Duke 390",
        "💰 Cheapest chain lube?",
        "📋 I need tyres",
    ]

    for q in example_queries:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            st.session_state["quick_query"] = q[2:].strip()
            st.rerun()

    st.divider()
    st.markdown("**⚙️ Settings**")
    top_k = st.slider("RAG results (top-k)", 3, 10, 5)
    st.session_state["top_k"] = top_k

    if st.button("🗑 Clear Conversation", use_container_width=True):
        st.session_state["messages"] = []
        if "agent" in st.session_state:
            st.session_state["agent"].reset()
        st.rerun()

    # ── Chat History ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**💬 Chat History**")

    if st.session_state["messages"]:
        history_html = '<div class="chat-history-scroll">'
        for msg in st.session_state["messages"]:
            role_class = "role-user" if msg["role"] == "user" else "role-assistant"
            role_label = "You" if msg["role"] == "user" else "VIKMO"
            # Truncate long messages for the sidebar preview
            preview = msg["content"].replace("\n", " ").strip()
            if len(preview) > 80:
                preview = preview[:80] + "…"
            history_html += f"""
            <div class="chat-history-item">
                <div class="{role_class}">{role_label}</div>
                <div>{preview}</div>
            </div>
            """
        history_html += "</div>"
        st.markdown(history_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="color:#aaaaaa; font-size:0.8rem; text-align:center; padding: 0.5rem 0;">No messages yet</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("""
    <div style="color:#555555; font-size:0.75rem; line-height: 1.5;">
    Powered by Groq / Gemini<br>
    RAG: FAISS + MiniLM-L6<br>
    600 SKUs · Real-time stock
    </div>
    """, unsafe_allow_html=True)

# ── Agent init ────────────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    groq_key   = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not groq_key and not gemini_key:
        st.error(
            "⚠️ **No API key set.**\n\n"
            "Add either `GROQ_API_KEY` (free at console.groq.com) or "
            "`GEMINI_API_KEY` to your `.env` file, then restart."
        )
        st.stop()

    with st.spinner("🔧 Loading catalogue index …"):
        try:
            from assistant.agent import DealerAssistant
            st.session_state["agent"] = DealerAssistant()
        except Exception as e:
            st.error(f"Failed to initialise agent: {e}")
            st.stop()

    # Welcome message
    st.session_state["messages"].append({
        "role": "assistant",
        "content": (
            "👋 **Welcome to VIKMO Dealer Assistant!**\n\n"
            "I can help you:\n"
            "- 🔍 **Find parts** for any vehicle\n"
            "- 📦 **Check stock** for specific SKUs\n"
            "- 🛒 **Place orders** with structured confirmations\n\n"
            "What can I help you with today?"
        )
    })

# ── Display Brand Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="vikmo-header">
    <h1 class="vikmo-title"><span>VIKMO</span></h1>
    <div class="vikmo-subtitle">Dealer Assistant</div>
    <div style="margin-top: 8px;"><span class="status-pill">🟢 Online</span></div>
</div>
""", unsafe_allow_html=True)

# ── Display chat history ───────────────────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar="🏍️" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# ── Handle quick query from sidebar ───────────────────────────────────────────
quick_query = st.session_state.pop("quick_query", None)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about parts, stock, or place an order …") or quick_query

if user_input:
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Get assistant response
    with st.chat_message("assistant", avatar="🏍️"):
        with st.spinner("Thinking …"):
            try:
                agent = st.session_state["agent"]
                reply = agent.chat(user_input)
            except Exception as exc:
                reply = f"⚠️ Error: {exc}"
        st.markdown(reply)

    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.rerun()