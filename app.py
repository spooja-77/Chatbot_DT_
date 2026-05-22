import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime
from groq import Groq

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DownBot — Factory Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #161920;
    --border: #252a35;
    --accent: #00e5a0;
    --accent2: #ff6b35;
    --text: #e8eaf0;
    --muted: #7a8099;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: var(--bg) !important;
    color: var(--text);
}

.stApp { background: var(--bg) !important; }

/* Header */
.main-header {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 0 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.main-header .logo { font-size: 2.2rem; }
.main-header h1 {
    margin: 0; font-size: 1.7rem; font-weight: 800;
    background: linear-gradient(90deg, var(--accent), #00b8ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.main-header p { margin: 0; font-size: 0.8rem; color: var(--muted); font-family: 'JetBrains Mono', monospace; }

/* Chat bubbles */
.msg-user {
    background: linear-gradient(135deg, #1a2a3a, #1e2d42);
    border: 1px solid #2a3f5a;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px; margin: 8px 0 8px 60px;
    color: var(--text); font-size: 0.95rem; line-height: 1.6;
    box-shadow: 0 2px 12px rgba(0,229,160,0.08);
}
.msg-bot {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px; margin: 8px 60px 8px 0;
    color: var(--text); font-size: 0.95rem; line-height: 1.7;
}
.msg-bot code {
    background: #1a1f2e; border: 1px solid var(--border);
    border-radius: 4px; padding: 1px 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--accent);
}
.msg-bot pre {
    background: #0a0d14; border: 1px solid var(--border);
    border-radius: 8px; padding: 12px; overflow-x: auto;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
}
.msg-meta {
    font-size: 0.72rem; color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 10px; margin-bottom: 16px; }
.kpi-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px;
    border-top: 2px solid var(--accent);
}
.kpi-label { font-size: 0.7rem; color: var(--muted); font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 1px; }
.kpi-value { font-size: 1.4rem; font-weight: 800; color: var(--accent); margin-top: 2px; }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Style buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 6px !important;
    padding: 4px 12px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
}

/* Input */
.stTextInput > div > div > input, .stChatInput textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
}

/* Personality chip */
.personality-active {
    display: inline-block; padding: 3px 10px;
    background: rgba(0,229,160,0.12); border: 1px solid var(--accent);
    border-radius: 20px; font-size: 0.72rem; color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
}

/* Suggestion chips */
.chip {
    display: inline-block; margin: 3px;
    padding: 5px 12px;
    background: #1a1f2e; border: 1px solid var(--border);
    border-radius: 20px; font-size: 0.78rem; color: var(--muted);
    cursor: pointer; transition: all 0.2s;
}

/* Status dot */
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); display: inline-block; margin-right: 6px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Data Layer ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "downtime_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Start_time"] = pd.to_datetime(df["Start_time"], errors="coerce")
    df["Stop_time"]  = pd.to_datetime(df["Stop_time"],  errors="coerce")
    df["down_time"]  = pd.to_numeric(df["down_time"], errors="coerce")
    return df


def build_data_context(df: pd.DataFrame) -> str:
    """Compress dataframe into a rich text context for the LLM."""
    total_dt   = df["down_time"].sum()
    avg_dt     = df["down_time"].mean()
    top_reason = df.groupby("Breakdown_Reason")["down_time"].sum().idxmax()
    top_global = df.groupby("Global_reason")["down_time"].sum().idxmax()
    by_shift   = df.groupby("Shift")["down_time"].sum().to_dict()
    by_reason  = df.groupby("Breakdown_Reason")["down_time"].sum().nlargest(5).to_dict()
    by_global  = df.groupby("Global_reason")["down_time"].sum().to_dict()
    type_map   = {3: "Unplanned", 4: "Planned"}
    by_type    = df["type"].map(type_map).value_counts().to_dict()
    machines   = df["Machine_ID"].unique().tolist()
    shifts     = df["Shift"].unique().tolist()

    sample_rows = df.head(5).to_string(index=False)

    ctx = f"""
=== FACTORY DOWNTIME DATABASE CONTEXT ===
Total records: {len(df)}
Date range: {df['Start_time'].min()} → {df['Start_time'].max()}
Machines monitored: {machines}
Shifts: {shifts}

--- AGGREGATE STATS ---
Total downtime (seconds): {total_dt:,.0f}  ({total_dt/3600:.1f} hours)
Average downtime per event (seconds): {avg_dt:.1f}
Top breakdown reason: {top_reason}
Top global reason category: {top_global}

--- DOWNTIME BY SHIFT ---
{json.dumps(by_shift, indent=2)}

--- TOP 5 BREAKDOWN REASONS (total seconds) ---
{json.dumps(by_reason, indent=2)}

--- DOWNTIME BY GLOBAL REASON CATEGORY ---
{json.dumps(by_global, indent=2)}

--- PLANNED vs UNPLANNED ---
{json.dumps(by_type, indent=2)}

--- SAMPLE RECORDS (first 5 rows) ---
{sample_rows}

--- COLUMN DESCRIPTIONS ---
ID: unique event id | Business: business unit | Plant_ID/Line_ID/Machine_ID: location hierarchy
Shift: Shift1/Shift2/Shift3 | Hour: time slot | type: 3=Unplanned, 4=Planned
down_time: duration in seconds | Reason: numeric reason code
Breakdown_Reason: human-readable reason | Global_reason: category (Man/Method/Material/Machine)
=== END CONTEXT ===
"""
    return ctx


def compute_kpis(df: pd.DataFrame):
    total_sec   = df["down_time"].sum()
    events      = len(df)
    unplanned   = len(df[df["type"] == 3])
    top_reason  = df.groupby("Breakdown_Reason")["down_time"].sum().idxmax()
    return {
        "total_hours": f"{total_sec/3600:.1f}h",
        "events":      str(events),
        "unplanned":   f"{unplanned}/{events}",
        "top_cause":   top_reason.replace(" ", "\u00a0")[:18],
    }


# ── Personality System ────────────────────────────────────────────────────────
PERSONALITIES = {
    "🔬 Analyst":    ("analyst",    "You are a precise manufacturing analyst. Respond with bullet-point structure, exact numbers, percentages, and data-driven insights. Use engineering vocabulary. Lead every answer with the key metric."),
    "🤖 Engineer":   ("engineer",   "You are a seasoned factory floor engineer. Give practical, hands-on advice. Use technical jargon naturally. Focus on root causes and corrective actions. Be direct and solution-oriented."),
    "📊 Executive":  ("executive",  "You are a C-suite operations advisor. Translate data into business impact: costs, productivity, competitive risk. Speak in terms of KPIs, OEE, and ROI. Keep answers concise and impactful."),
    "🧑‍🏫 Teacher":  ("teacher",    "You are a patient manufacturing coach. Explain concepts clearly with analogies. Break complex patterns into simple steps. Use examples from the data to illustrate every point. Encourage curiosity."),
    "🕵️ Detective":  ("detective",  "You are a downtime detective. Build hypotheses from data clues. Think out loud. Connect patterns across shifts, machines, and categories. Highlight anomalies. Ask probing follow-up questions."),
}

SYSTEM_TEMPLATE = """You are DownBot, an expert industrial AI assistant embedded in a smart factory analytics platform.
You have deep expertise in manufacturing KPIs, OEE (Overall Equipment Effectiveness), Six Sigma, TPM (Total Productive Maintenance), and lean manufacturing.

PERSONALITY MODE: {personality_desc}

LIVE DATA CONTEXT:
{data_context}

RULES:
1. Always ground answers in the actual data provided above.
2. When quoting numbers, be precise (use exact seconds, convert to minutes/hours when helpful).
3. If asked something outside the data, say so clearly then offer what you CAN answer.
4. Format responses with markdown: use **bold** for key terms, `code` for IDs/codes, tables when comparing multiple items.
5. Proactively surface related insights the user didn't ask for but would find valuable.
6. End complex analytical answers with a "💡 Recommendation:" section.
7. Never make up data that isn't in the context.
"""

SUGGESTED_QUESTIONS = [
    "Which shift had the most downtime?",
    "What's the top unplanned breakdown reason?",
    "Compare Man vs Method vs Material losses",
    "Which machine had the longest single downtime event?",
    "Show me a summary of Shift1 performance",
    "What % of downtime is planned vs unplanned?",
    "How can we reduce Cleaning/POC/DRM downtime?",
    "Any anomalies I should investigate?",
]


# ── Groq Client ──────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def chat(client, messages, system_prompt):
    all_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        messages=all_messages,
    )
    return response.choices[0].message.content


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "personality" not in st.session_state:
    st.session_state.personality = "🔬 Analyst"
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()
data_ctx = build_data_context(df)
kpis = compute_kpis(df)
client = get_client()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:1.3rem;font-weight:800;margin-bottom:4px;">⚙️ DownBot</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.72rem;color:#7a8099;font-family:JetBrains Mono,monospace;margin-bottom:18px;"><span class="status-dot"></span>LIVE · {len(df)} events loaded</div>', unsafe_allow_html=True)

    st.markdown("**Personality Mode**")
    for label in PERSONALITIES:
        if st.button(label, key=f"pers_{label}", use_container_width=True):
            st.session_state.personality = label
            st.rerun()
    pid, pdesc = PERSONALITIES[st.session_state.personality]
    st.markdown(f'<div class="personality-active">{st.session_state.personality} active</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**📊 KPIs**")
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">Total Downtime</div><div class="kpi-value">{kpis['total_hours']}</div></div>
        <div class="kpi-card"><div class="kpi-label">Events</div><div class="kpi-value">{kpis['events']}</div></div>
        <div class="kpi-card"><div class="kpi-label">Unplanned</div><div class="kpi-value">{kpis['unplanned']}</div></div>
        <div class="kpi-card"><div class="kpi-label">Top Cause</div><div class="kpi-value" style="font-size:0.85rem;">{kpis['top_cause']}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("**📂 Upload Your Data**")
    uploaded = st.file_uploader("Drop CSV here", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df = pd.read_csv(uploaded)
        df["Start_time"] = pd.to_datetime(df["Start_time"], errors="coerce")
        df["Stop_time"]  = pd.to_datetime(df["Stop_time"], errors="coerce")
        df["down_time"]  = pd.to_numeric(df["down_time"], errors="coerce")
        data_ctx = build_data_context(df)
        kpis = compute_kpis(df)
        st.success(f"✅ Loaded {len(df)} records")


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div class="logo">⚙️</div>
  <div>
    <h1>DownBot — Factory Intelligence</h1>
    <p>RAG-powered downtime analysis &nbsp;·&nbsp; Ask anything about your manufacturing data</p>
  </div>
</div>
""", unsafe_allow_html=True)

# API key check
if not client:
    st.error("⚠️ ANTHROPIC_API_KEY not found. Add it to your `.streamlit/secrets.toml` or environment variables.")
    st.code('# .streamlit/secrets.toml\nANTHROPIC_API_KEY = "sk-ant-..."', language="toml")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div class="msg-bot">
        👋 <strong>Welcome to DownBot!</strong><br><br>
        I have your factory downtime data loaded and ready for analysis. I can help you:<br>
        • Identify top downtime causes and patterns<br>
        • Compare shift performance<br>
        • Distinguish planned vs unplanned events<br>
        • Surface root-cause insights and recommendations<br><br>
        Try one of the suggested questions below, or ask me anything! 🔍
        </div>
        """, unsafe_allow_html=True)

        # Suggestion chips via buttons
        st.markdown("**💬 Suggested Questions**")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"sugg_{i}"):
                st.session_state.pending_question = q
                st.rerun()
    else:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            ts = msg.get("ts", "")
            if role == "user":
                st.markdown(f'<div class="msg-meta" style="text-align:right;">You · {ts}</div><div class="msg-user">{content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="msg-meta">⚙️ DownBot · {ts}</div><div class="msg-bot">{content}</div>', unsafe_allow_html=True)


# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about your downtime data…")

# Handle suggestion chip OR typed input
question = st.session_state.pending_question or user_input
if st.session_state.pending_question:
    st.session_state.pending_question = None

if question:
    ts = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": question, "ts": ts})

    _, pdesc = PERSONALITIES[st.session_state.personality]
    system_prompt = SYSTEM_TEMPLATE.format(
        personality_desc=pdesc,
        data_context=data_ctx,
    )

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.spinner("Analysing…"):
        try:
            answer = chat(client, history, system_prompt)
        except Exception as e:
            answer = f"⚠️ API error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer, "ts": ts})
    st.rerun()
