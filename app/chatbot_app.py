"""
Medical Symptom Triage Chatbot — Streamlit Web Interface
A premium, bilingual (FR/EN) conversational UI for AI-powered medical triage.
"""
import sys
import os
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# ── Page configuration (must be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="MedTriage AI — Chatbot de Triage Médical",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import pipeline modules ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load all heavy models once and cache them across sessions."""
    from src.safety.safety_filter import SafetyFilter
    from src.ner.ner_extractor import MedicalNERExtractor
    from src.rag.rag_pipeline import MedicalRAGPipeline
    from src.triage.triage_classifier import TriageClassifier
    from src.llm.llm_handler import MedicalLLMHandler
    return (
        SafetyFilter(),
        MedicalNERExtractor(),
        MedicalRAGPipeline(),
        TriageClassifier(),
        MedicalLLMHandler(),
    )


# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global Reset ── */
html, body, [data-testid="stApp"] {
    font-family: 'Inter', sans-serif;
    background: #0a0f1e;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1224 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #cbd5e1; }

/* ── Main area ── */
.main .block-container {
    padding-top: 1.5rem;
    max-width: 900px;
}

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a2a5e 50%, #0f2044 100%);
    border: 1px solid #2d4a7a;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.header-title {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem 0;
}
.header-subtitle {
    color: #94a3b8;
    font-size: 0.9rem;
    font-weight: 400;
    margin: 0;
}

/* ── Chat messages ── */
.chat-message {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    animation: slideIn 0.3s ease-out;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.chat-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 4px;
}
.avatar-user { background: linear-gradient(135deg, #3b82f6, #8b5cf6); }
.avatar-bot  { background: linear-gradient(135deg, #10b981, #0ea5e9); }
.chat-bubble {
    max-width: 85%;
    padding: 0.85rem 1.1rem;
    border-radius: 16px;
    font-size: 0.92rem;
    line-height: 1.6;
}
.bubble-user {
    background: linear-gradient(135deg, #1e3a5f, #1a2a5e);
    border: 1px solid #2d4a7a;
    margin-left: auto;
    border-top-right-radius: 4px;
}
.bubble-bot {
    background: #111827;
    border: 1px solid #1e293b;
    border-top-left-radius: 4px;
}

/* ── Triage badges ── */
.triage-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    margin-right: 0.4rem;
}
.badge-p1 { background: rgba(239,68,68,0.2);   color: #f87171; border: 1px solid #dc2626; }
.badge-p2 { background: rgba(249,115,22,0.2);  color: #fb923c; border: 1px solid #ea580c; }
.badge-p3 { background: rgba(234,179,8,0.2);   color: #facc15; border: 1px solid #ca8a04; }
.badge-p4 { background: rgba(59,130,246,0.2);  color: #60a5fa; border: 1px solid #2563eb; }
.badge-p5 { background: rgba(16,185,129,0.2);  color: #34d399; border: 1px solid #059669; }

/* ── Emergency alert ── */
.emergency-alert {
    background: linear-gradient(135deg, rgba(220,38,38,0.2), rgba(239,68,68,0.1));
    border: 2px solid #dc2626;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.4); }
    50%       { box-shadow: 0 0 0 8px rgba(220,38,38,0); }
}

/* ── Entity chips ── */
.entity-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.5rem;
}
.chip {
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}
.chip-symptom  { background: rgba(139,92,246,0.2); color: #a78bfa; border: 1px solid #7c3aed; }
.chip-duration { background: rgba(14,165,233,0.2); color: #38bdf8; border: 1px solid #0284c7; }
.chip-intensity{ background: rgba(249,115,22,0.2); color: #fb923c; border: 1px solid #ea580c; }
.chip-location { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid #059669; }

/* ── RAG disease card ── */
.disease-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
}
.disease-card:hover {
    border-color: #3b82f6;
    transition: border-color 0.2s;
}
.disease-name { font-weight: 600; color: #60a5fa; font-size: 0.9rem; }
.disease-icd  { color: #64748b; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; }
.confidence-bar {
    height: 4px;
    background: #1e293b;
    border-radius: 2px;
    margin-top: 0.4rem;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
}

/* ── Input area ── */
.stTextInput > div > div > input {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Spinner / loading ── */
.typing-indicator {
    display: flex;
    gap: 5px;
    align-items: center;
    padding: 0.75rem 1rem;
}
.typing-dot {
    width: 8px; height: 8px;
    background: #3b82f6;
    border-radius: 50%;
    animation: bounce 1.2s ease-in-out infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40%           { transform: scale(1.0); opacity: 1; }
}

/* ── Disclaimer box ── */
.disclaimer-box {
    background: rgba(234,179,8,0.07);
    border: 1px solid rgba(234,179,8,0.3);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.78rem;
    color: #a3a3a3;
    margin-top: 0.75rem;
}

/* ── Divider ── */
hr { border-color: #1e293b; margin: 1rem 0; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Helper: render triage badge ────────────────────────────────────────────
def triage_badge_html(level: str) -> str:
    labels = {
        "P1": ("🚨 P1 — Urgence Vitale", "badge-p1"),
        "P2": ("🔴 P2 — Urgence", "badge-p2"),
        "P3": ("🟡 P3 — Semi-urgence", "badge-p3"),
        "P4": ("🔵 P4 — Non urgent", "badge-p4"),
        "P5": ("🟢 P5 — Conseil", "badge-p5"),
    }
    label, cls = labels.get(level, (level, "badge-p5"))
    return f'<span class="triage-badge {cls}">{label}</span>'


# ── Helper: render entity chips ────────────────────────────────────────────
def render_entity_chips(entities: dict) -> str:
    chips = []
    for sym in entities.get("symptoms", []):
        chips.append(f'<span class="chip chip-symptom">💊 {sym.replace("_", " ")}</span>')
    if entities.get("duration"):
        chips.append(f'<span class="chip chip-duration">⏱ {entities["duration"]}</span>')
    if entities.get("intensity"):
        chips.append(f'<span class="chip chip-intensity">⚡ {entities["intensity"]}</span>')
    if entities.get("location"):
        chips.append(f'<span class="chip chip-location">📍 {entities["location"]}</span>')
    if not chips:
        return ""
    return '<div class="entity-chips">' + "".join(chips) + "</div>"


# ── Helper: build full formatted bot reply ─────────────────────────────────
def build_structured_reply(
    query: str,
    entities: dict,
    triage: dict,
    diseases: list,
    llm_text: str,
    disclaimer: str,
    lang: str,
    is_emergency: bool = False,
) -> dict:
    return {
        "query": query,
        "entities": entities,
        "triage": triage,
        "diseases": diseases,
        "llm_text": llm_text,
        "disclaimer": disclaimer,
        "lang": lang,
        "is_emergency": is_emergency,
    }


# ── Render a full bot message ───────────────────────────────────────────────
def render_bot_message(msg: dict):
    triage = msg["triage"]
    level = triage.get("triage_level", "P5")
    lang = msg["lang"]
    is_emergency = msg.get("is_emergency", False)

    # Avatar + bubble wrapper
    st.markdown(
        '<div class="chat-message">'
        '<div class="chat-avatar avatar-bot">🏥</div>'
        '<div class="chat-bubble bubble-bot" style="width:100%">',
        unsafe_allow_html=True,
    )

    if is_emergency:
        action = triage.get("action_fr") if lang == "fr" else triage.get("action_en")
        st.markdown(
            f'<div class="emergency-alert">'
            f'<strong style="color:#f87171;font-size:1rem;">🚨 ALERTE URGENCE VITALE</strong><br>'
            f'<span style="color:#fca5a5">{action}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        # Entity chips
        chips_html = render_entity_chips(msg["entities"])
        if chips_html:
            st.markdown(
                f"<div style='margin-bottom:0.6rem'><span style='color:#64748b;font-size:0.78rem'>Entités détectées :</span>"
                + chips_html
                + "</div>",
                unsafe_allow_html=True,
            )

        # Triage badge
        st.markdown(triage_badge_html(level), unsafe_allow_html=True)

        # RAG disease cards (collapsible)
        diseases = msg.get("diseases", [])
        if diseases:
            with st.expander("🔬 Pathologies candidates (RAG)", expanded=(level in ["P1", "P2"])):
                for d in diseases[:3]:
                    conf = int(min(max(d["score"] * 100, 15), 98))
                    st.markdown(
                        f'<div class="disease-card">'
                        f'<span class="disease-name">{d["disease_name"]}</span> '
                        f'<span class="disease-icd">{d["icd10"]}</span>'
                        f'<div class="confidence-bar"><div class="confidence-fill" style="width:{conf}%"></div></div>'
                        f'<div style="color:#94a3b8;font-size:0.78rem;margin-top:0.3rem">{d.get("description","")[:120]}…</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # LLM text (markdown)
        st.markdown(msg["llm_text"])

    # Disclaimer
    disc_text = (
        "⚠️ Ce système est un outil d'aide préliminaire et ne remplace pas un diagnostic médical. "
        "En urgence : appelez le **15** ou le **112**."
        if lang == "fr"
        else
        "⚠️ This tool is for preliminary guidance only and does not replace professional medical advice. "
        "In an emergency, call **112**, **911**, or **15** immediately."
    )
    st.markdown(
        f'<div class="disclaimer-box">{disc_text}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div></div>", unsafe_allow_html=True)


# ── Render a user message ──────────────────────────────────────────────────
def render_user_message(text: str):
    st.markdown(
        f'<div class="chat-message" style="flex-direction:row-reverse">'
        f'<div class="chat-avatar avatar-user">👤</div>'
        f'<div class="chat-bubble bubble-user">{text}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Main pipeline call ─────────────────────────────────────────────────────
def run_triage(query: str, safety, ner, rag, triage_clf, llm) -> dict:
    lang = safety.detect_language(query)

    # 1. Emergency check
    if safety.detect_immediate_emergency(query):
        emergency_resp = safety.get_emergency_response(lang)
        text = emergency_resp["action_fr"] if lang == "fr" else emergency_resp["action_en"]
        disclaimer = safety.inject_disclaimer("", lang)
        return build_structured_reply(
            query=query,
            entities={"symptoms": [], "duration": None, "intensity": None, "location": None},
            triage=emergency_resp,
            diseases=[],
            llm_text=text,
            disclaimer=disclaimer,
            lang=lang,
            is_emergency=True,
        )

    # 2. NER
    entities = ner.extract_symptoms(query)

    # 3. RAG
    diseases = rag.search_conditions(query, entities["symptoms"], k=3)

    # 4. Triage
    triage_result = triage_clf.classify(entities, diseases)

    # 5. LLM
    llm_text = llm.generate_response(query, entities, triage_result, diseases, lang)

    # 6. Disclaimer
    disclaimer = safety.inject_disclaimer("", lang)

    return build_structured_reply(
        query=query,
        entities=entities,
        triage=triage_result,
        diseases=diseases,
        llm_text=llm_text,
        disclaimer=disclaimer,
        lang=lang,
        is_emergency=False,
    )


# ══════════════════════════════════════════════════════════════════════════
#  PAGE LAYOUT
# ══════════════════════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:1rem 0 0.5rem'>"
        "<span style='font-size:3rem'>🏥</span>"
        "<h2 style='color:#60a5fa;margin:0.4rem 0 0.1rem;font-size:1.1rem'>MedTriage AI</h2>"
        "<p style='color:#64748b;font-size:0.75rem;margin:0'>Système de Triage Médical Intelligent</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("#### 📋 Niveaux de Triage")
    for lvl, info in {
        "🚨 P1": ("Urgence vitale", "#f87171", "Appel immédiat 15/112"),
        "🔴 P2": ("Urgence", "#fb923c", "Urgences dans l'heure"),
        "🟡 P3": ("Semi-urgence", "#facc15", "Médecin sous 24h"),
        "🔵 P4": ("Non urgent", "#60a5fa", "RDV sous 48-72h"),
        "🟢 P5": ("Conseil", "#34d399", "Pharmacie / Télémédecine"),
    }.items():
        label, color, action = info
        st.markdown(
            f"<div style='margin-bottom:0.5rem'>"
            f"<strong style='color:{color}'>{lvl}</strong> "
            f"<span style='color:#cbd5e1;font-size:0.82rem'>— {label}</span><br>"
            f"<span style='color:#64748b;font-size:0.75rem;padding-left:1.2rem'>{action}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### ⚙️ Paramètres")
    show_entities = st.toggle("Afficher les entités NER", value=True)
    show_rag_details = st.toggle("Afficher les pathologies RAG", value=True)

    st.divider()
    st.markdown("#### 💬 Exemples de requêtes")
    example_queries = {
        "🚨 Urgence P1": "J'ai une douleur intense à la poitrine depuis 10 minutes, je pense que c'est une crise cardiaque",
        "🟡 Semi-urgent P3": "I have a high fever of 39°C and chills for 3 days with joint pain",
        "🟢 Banal P5": "J'ai le nez qui coule, je tousse légèrement et j'ai mal à la gorge depuis hier",
        "🔵 Chronique P4": "I have acid reflux and vomiting mildly for 2 weeks",
    }
    for label, example in example_queries.items():
        if st.button(label, key=f"ex_{label}", use_container_width=True):
            st.session_state.pending_query = example

    st.divider()
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<div style='color:#334155;font-size:0.7rem;text-align:center;margin-top:1rem'>"
        "v1.0 · Python · LangChain · RAG · NER<br>"
        "Usage non médical — outil d'aide uniquement"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="header-banner">'
    '<h1 class="header-title">🏥 MedTriage AI</h1>'
    '<p class="header-subtitle">'
    "Chatbot de Triage Médical Intelligent · Analyse de symptômes en langage naturel (FR / EN) · "
    "Pipeline NER → RAG → Triage P1-P5 → LLM"
    "</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Session state init ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

# ── Load pipeline ───────────────────────────────────────────────────────────
if not st.session_state.pipeline_ready:
    with st.spinner("⚙️ Chargement des modèles IA... (première exécution uniquement)"):
        try:
            safety, ner, rag, triage_clf, llm = load_pipeline()
            st.session_state.pipeline_ready = True
            st.session_state.safety = safety
            st.session_state.ner = ner
            st.session_state.rag = rag
            st.session_state.triage_clf = triage_clf
            st.session_state.llm = llm
        except Exception as e:
            st.error(f"Erreur de chargement du pipeline : {e}")
            st.stop()
else:
    safety = st.session_state.safety
    ner = st.session_state.ner
    rag = st.session_state.rag
    triage_clf = st.session_state.triage_clf
    llm = st.session_state.llm

# ── Welcome message ─────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        '<div class="chat-message">'
        '<div class="chat-avatar avatar-bot">🏥</div>'
        '<div class="chat-bubble bubble-bot">'
        "<strong>Bonjour ! Je suis MedTriage AI</strong> 👋<br><br>"
        "Je suis un assistant d'aide à l'orientation médicale. Décrivez-moi vos symptômes en <strong>français</strong> ou en <strong>anglais</strong> "
        "et je vous indiquerai le niveau d'urgence approprié (P1 à P5) avec des recommandations adaptées.<br><br>"
        "<span style='color:#f87171;font-weight:600'>⚠️ En cas d'urgence vitale, appelez le 15 ou le 112 immédiatement.</span><br><br>"
        "<span style='color:#64748b;font-size:0.82rem'>Exemple : <em>\"J'ai de la fièvre et des frissons depuis 3 jours\"</em></span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Render conversation history ─────────────────────────────────────────────
for entry in st.session_state.messages:
    render_user_message(entry["query"])
    render_bot_message(entry)

# ── Handle pending example query from sidebar ────────────────────────────────
if "pending_query" in st.session_state:
    pending = st.session_state.pop("pending_query")
    st.session_state._inject_query = pending

# ── Input form ─────────────────────────────────────────────────────────────
with st.container():
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Décrivez vos symptômes…",
            key="chat_input",
            placeholder="Ex: J'ai mal à la tête depuis 2 jours avec de la fièvre · I have chest pain and shortness of breath",
            label_visibility="collapsed",
            value=st.session_state.get("_inject_query", ""),
        )
        if "_inject_query" in st.session_state:
            del st.session_state["_inject_query"]

    with col_btn:
        send_clicked = st.button("Envoyer →", use_container_width=True)

# ── Process input ───────────────────────────────────────────────────────────
query_to_process = user_input.strip() if (send_clicked and user_input.strip()) else None

if query_to_process:
    # Show user message immediately
    render_user_message(query_to_process)

    # Typing indicator
    with st.spinner("🔍 Analyse en cours…"):
        try:
            result = run_triage(
                query_to_process,
                safety,
                ner,
                rag,
                triage_clf,
                llm,
            )
        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {e}")
            st.stop()

    # Append to history and render
    st.session_state.messages.append(result)
    render_bot_message(result)
    st.rerun()
