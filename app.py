import os, json, time, re
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import CrossEncoder, SentenceTransformer
from nli_contradiction import check_contradiction
import pypdf

load_dotenv()

st.set_page_config(page_title="ContraRAG", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if st.session_state.theme == "dark":
    theme_css = """:root {
        --bg: linear-gradient(-45deg, #050511, #0f172a, #1e1b4b, #020617);
        --card: rgba(15,23,42,0.55); --border: rgba(255,255,255,0.08);
        --text: #F8FAFC; --sub: #94A3B8; --accent: #38BDF8;
        --shadow: 0 8px 32px rgba(0,0,0,0.5);}"""
else:
    theme_css = """:root {
        --bg: linear-gradient(-45deg, #F8FAFC, #F1F5F9, #E2E8F0, #F8FAFC);
        --card: rgba(255,255,255,0.85); --border: rgba(15,23,42,0.15);
        --text: #020617; --sub: #334155; --accent: #0284C7;
        --shadow: 0 8px 32px rgba(15,23,42,0.08);}"""

st.markdown(f"""<style>
{theme_css}
.stApp {{background:var(--bg);background-size:400% 400%;animation:gradientBG 25s ease infinite;color:var(--text) !important;}}
@keyframes gradientBG {{0%{{background-position:0% 50%}}50%{{background-position:100% 50%}}100%{{background-position:0% 50%}}}}
.main .block-container {{padding-top:1.5rem;max-width:1400px;}}
.title {{font-size:2.4rem;font-weight:800;color:var(--text);letter-spacing:-0.03em;margin-bottom:2px;}}
.subtitle {{color:var(--sub);font-size:1.05rem;margin-bottom:2rem;}}
div[data-testid="stVerticalBlockBorderWrapper"] {{background:var(--card) !important;backdrop-filter:blur(16px) !important;border:1px solid var(--border) !important;border-radius:16px !important;padding:1.5rem !important;box-shadow:var(--shadow) !important;margin-bottom:1.25rem;}}
.bubble {{padding:1.25rem;border-radius:12px;margin-bottom:1rem;border-left:4px solid #64748B;}}
.eu {{border-left-color:#3B82F6 !important;background:rgba(59,130,246,0.08);}}
.us {{border-left-color:#EF4444 !important;background:rgba(239,68,68,0.08);}}
.agent {{font-size:0.85rem;font-weight:800;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;}}
p,li,span,label,div {{color:var(--text) !important;}}
.caption {{font-size:0.8rem;font-weight:800;text-transform:uppercase;color:var(--accent) !important;letter-spacing:0.08em;margin-bottom:0.75rem;}}
div[data-testid="stMetricValue"] {{font-size:2.5rem !important;font-weight:800 !important;color:var(--text) !important;}}
.status {{background:rgba(0,0,0,0.05);border-radius:10px;padding:12px;margin-top:1rem;border:1px solid var(--border);}}
.decomp-tag {{display:inline-block;background:rgba(56,189,248,0.15);border:1px solid rgba(56,189,248,0.3);border-radius:6px;padding:2px 8px;font-size:0.78rem;margin:2px;color:var(--accent) !important;}}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return embed, reranker

@st.cache_resource
def get_qdrant():
    return QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), timeout=30)

embed_model, reranker = load_models()
qdrant = get_qdrant()

client_eu = Groq(api_key=os.getenv("GROQ_API_KEY_EU"))
client_us = Groq(api_key=os.getenv("GROQ_API_KEY_US"))
client_judge = Groq(api_key=os.getenv("GROQ_API_KEY_JUDGE"))

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_context" not in st.session_state:
    st.session_state.last_context = ""


def decompose_query(query):
    """
    Use LLM to detect if query is complex (multi-hop) and split into sub-questions.
    Returns list of sub-questions, or [original_query] if simple.
    """
    prompt = f"""You are a query analyzer for an AI regulation question-answering system.

Analyze this question: "{query}"

If this is a SIMPLE question (asks about one thing from one jurisdiction), return:
{{"is_complex": false, "sub_questions": ["{query}"]}}

If this is a COMPLEX question (compares multiple jurisdictions, asks about differences/similarities, or needs multiple lookups), break it into 2-4 simple sub-questions and return:
{{"is_complex": true, "sub_questions": ["sub-question 1", "sub-question 2", ...]}}

Return ONLY valid JSON, nothing else."""

    resp = client_judge.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.1,
        response_format={"type": "json_object"}
    ).choices[0].message.content

    try:
        result = json.loads(resp)
        return result.get("sub_questions", [query]), result.get("is_complex", False)
    except:
        return [query], False


def search_qdrant_single(query, jurisdictions, limit=5):
    """Search Qdrant for a single query across given jurisdictions."""
    vector = embed_model.encode(query).tolist()
    raw = []
    for region in jurisdictions:
        results = qdrant.query_points(
            collection_name="ai_regulations",
            query=vector,
            limit=limit,
            query_filter=Filter(must=[FieldCondition(key="jurisdiction", match=MatchValue(value=region))])
        ).points
        for r in results:
            raw.append({
                "text": r.payload["text"],
                "source": r.payload.get("source_file", "unknown"),
                "jurisdiction": r.payload.get("jurisdiction", "unknown"),
                "score": r.score,
                "sub_query": query
            })
    return raw


def search_qdrant(query, jurisdictions):
    """
    Full retrieval pipeline with query decomposition:
    1. Decompose complex queries into sub-questions
    2. Retrieve for each sub-question separately
    3. Rerank combined results
    4. Return top chunks
    """
    sub_questions, is_complex = decompose_query(query)

    all_raw = []
    for sub_q in sub_questions:
        results = search_qdrant_single(sub_q, jurisdictions, limit=4)
        all_raw.extend(results)

    if not all_raw:
        return "", [], sub_questions, is_complex

    # Deduplicate by text
    seen = set()
    unique = []
    for r in all_raw:
        key = r["text"][:100]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Rerank against original query
    scores = reranker.predict([[query, c["text"]] for c in unique])
    for i, c in enumerate(unique):
        c["rerank_score"] = float(scores[i])
    unique.sort(key=lambda x: x["rerank_score"], reverse=True)

    top = unique[:6]
    context = "\n\n---\n\n".join(
        [f"Doc: {c['source']} ({c['jurisdiction']})\n{c['text']}" for c in top]
    )
    return context, top, sub_questions, is_complex


def stream_bubble(container, name, color, text, css):
    words = text.split(" ")
    displayed = ""
    for word in words:
        displayed += word + " "
        container.markdown(
            f'<div class="bubble {css}"><div class="agent" style="color:{color};">{name}</div>'
            f'<div style="font-size:0.95rem;line-height:1.6;">{displayed}</div></div>',
            unsafe_allow_html=True
        )
        time.sleep(0.02)


# ── Header ──────────────────────────────────────────────────────────────
st.markdown('<div class="title">ContraRAG ⚖️</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Contradiction-aware AI regulation question answering across EU, US, and UK</div>', unsafe_allow_html=True)

with st.sidebar:
    icon = "☀️ Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
    if st.button(icon, use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='caption'>Settings</div>", unsafe_allow_html=True)
    regions = st.multiselect("Jurisdictions", ["EU", "US", "UK"], default=["EU", "US", "UK"])
    query = st.text_area("Your Question", value="What are the transparency requirements for AI systems?", height=100)

    st.markdown("<hr>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a document (optional)", type=["pdf", "txt"])
    run = st.button("🔍 Ask ContraRAG", use_container_width=True, type="primary")

    st.markdown("""<div class="status">
        <div style="color:var(--sub)"><span style="color:#10B981;">●</span> Qdrant Cloud (3982 chunks)</div>
        <div style="color:var(--sub)"><span style="color:#10B981;">●</span> Query Decomposition Active</div>
        <div style="color:var(--sub)"><span style="color:#10B981;">●</span> Llama 3.1 8B Ready</div>
    </div>""", unsafe_allow_html=True)

# ── Main ─────────────────────────────────────────────────────────────────
if run:
    if not regions:
        st.warning("Select at least one jurisdiction.")
        st.stop()

    extra = ""
    if uploaded:
        try:
            if uploaded.name.endswith(".pdf"):
                extra = "\n".join([p.extract_text() for p in pypdf.PdfReader(uploaded).pages if p.extract_text()])
            else:
                extra = uploaded.getvalue().decode("utf-8")
            extra = f"\nUploaded document:\n{extra[:3000]}"
        except:
            pass

    with st.spinner("Analysing and decomposing query..."):
        context, trail, sub_questions, is_complex = search_qdrant(query, regions)
        st.session_state.last_context = context

    if not context:
        st.error("No relevant documents found.")
        st.stop()

    # Show decomposition info
    if is_complex:
        st.info(f"**Multi-hop query detected** — split into {len(sub_questions)} sub-questions:")
        tags = " ".join([f'<span class="decomp-tag">🔍 {q}</span>' for q in sub_questions])
        st.markdown(tags, unsafe_allow_html=True)
    else:
        st.success("**Single-hop query** — direct retrieval")

    # EU Agent
    st.markdown("<div class='caption'>⚔️ Agent Responses</div>", unsafe_allow_html=True)
    eu_box = st.empty()
    us_box = st.empty()

    eu_prompt = f"""You are an EU AI regulation expert. Answer based ONLY on the context below.
Highlight EU-specific requirements clearly in 3-4 sentences.
Context:\n{context}{extra}\nQuestion: {query}"""

    eu_resp = client_eu.chat.completions.create(
        messages=[{"role": "user", "content": eu_prompt}],
        model="llama-3.1-8b-instant", temperature=0.3
    ).choices[0].message.content
    stream_bubble(eu_box, "🇪🇺 EU Perspective", "#3B82F6", eu_resp, "eu")

    # US Agent
    us_prompt = f"""You are a US AI regulation expert. Answer based ONLY on the context below.
Highlight US-specific requirements and how they differ from the EU in 3-4 sentences.
EU view:\n{eu_resp}\nContext:\n{context}{extra}\nQuestion: {query}"""

    us_resp = client_us.chat.completions.create(
        messages=[{"role": "user", "content": us_prompt}],
        model="llama-3.1-8b-instant", temperature=0.3
    ).choices[0].message.content
    stream_bubble(us_box, "🇺🇸 US Perspective", "#EF4444", us_resp, "us")

    # Step 1: NLI contradiction detection on raw chunks (reproducible score)
    with st.spinner("Running NLI contradiction detection..."):
        chunk_texts = [c["text"] for c in trail]
        nli_score, nli_max_pair, nli_details = check_contradiction(chunk_texts)
        nli_contradiction_found = nli_score > 0.5
        # Confidence = inverse of contradiction score
        nli_confidence = round(1.0 - nli_score, 2)
        if nli_confidence > 0.75:
            nli_decision = "ANSWER"
        elif nli_confidence > 0.4:
            nli_decision = "WARN"
        else:
            nli_decision = "REFUSE"

    # Step 2: Judge LLM for summaries and explanation (uses NLI score as ground truth)
    with st.spinner("Generating jurisdiction summaries..."):
        judge_prompt = f"""You are ContraRAG's judge. The NLI model has already determined:
- Contradiction score: {nli_score:.2f} (0=no contradiction, 1=strong contradiction)
- Contradiction found: {nli_contradiction_found}
- Decision: {nli_decision}

Your job is ONLY to provide summaries and explanation. Output ONLY valid JSON:
{{
  "regional_summaries": {{"EU": "2-3 sentence summary", "US": "2-3 sentence summary", "UK": "2-3 sentence summary"}},
  "contradiction_explanation": "explain what specifically contradicts based on the context, or say No contradiction found",
  "decision_reason": "explain why this decision was made given the evidence"
}}
EU Agent: {eu_resp[:500]}
US Agent: {us_resp[:500]}
Context: {context[:1500]}"""

        judge_raw = client_judge.chat.completions.create(
            messages=[{"role": "user", "content": judge_prompt}],
            model="llama-3.1-8b-instant", temperature=0.1,
            response_format={"type": "json_object"}
        ).choices[0].message.content
        judge_data = json.loads(judge_raw)

    # Merge NLI scores + LLM summaries into final verdict
    verdict = {
        "regional_summaries": judge_data.get("regional_summaries", {}),
        "contradiction_found": nli_contradiction_found,
        "contradiction_explanation": judge_data.get("contradiction_explanation", ""),
        "confidence_score": nli_confidence,
        "decision": nli_decision,
        "decision_reason": judge_data.get("decision_reason", ""),
        "friction_severity_score": round(nli_score * 10),
        "nli_score": nli_score,
        "nli_details": nli_details
    }

    # Results
    col_left, col_right = st.columns([6, 4], gap="medium")

    with col_left:
        decision = verdict.get("decision", "ANSWER")
        confidence = verdict.get("confidence_score", 0.8)
        if decision == "ANSWER":
            st.success(f"✅ ANSWER — Confidence: {confidence:.0%}")
        elif decision == "WARN":
            st.warning(f"⚠️ WARNING — Confidence: {confidence:.0%} — Sources partially conflict")
        else:
            st.error(f"🚫 REFUSED — Confidence: {confidence:.0%} — Sources strongly conflict")

        st.markdown(f"**Reason:** {verdict.get('decision_reason','')}")

        with st.container(border=True):
            st.markdown("<div class='caption'>🌐 Per-Jurisdiction Summary</div>", unsafe_allow_html=True)
            summaries = verdict.get("regional_summaries", {})
            tabs = st.tabs(["🇪🇺 EU", "🇺🇸 US", "🇬🇧 UK"])
            for i, region in enumerate(["EU", "US", "UK"]):
                with tabs[i]:
                    st.markdown(summaries.get(region, "Not covered in retrieved documents."))

        with st.container(border=True):
            st.markdown("<div class='caption'>⚡ Contradiction Analysis</div>", unsafe_allow_html=True)
            found = verdict.get("contradiction_found", False)
            st.markdown(f"**Contradiction detected:** {'Yes 🔴' if found else 'No 🟢'}")
            st.markdown(verdict.get("contradiction_explanation", ""))

    with col_right:
        with st.container(border=True):
            st.markdown("<div class='caption'>📊 Conflict Severity</div>", unsafe_allow_html=True)
            score = verdict.get("friction_severity_score", 0)
            nli_raw = verdict.get("nli_score", 0.0)
            st.metric("Policy Discord Score", f"{score} / 10")
            st.progress(score / 10.0)
            st.markdown(f"**NLI Contradiction Score:** `{nli_raw:.3f}`")
            st.caption("Scored by DeBERTa NLI model — not LLM opinion")

        with st.container(border=True):
            st.markdown("<div class='caption'>🔍 Sources Used</div>", unsafe_allow_html=True)
            for item in trail:
                sub_q = item.get("sub_query", "")
                label = f"{item['jurisdiction']} | {item['source']}"
                with st.expander(label):
                    if sub_q and is_complex:
                        st.markdown(f"*Retrieved for:* `{sub_q}`")
                    st.markdown(f"**Relevance:** {item['rerank_score']:.4f}")
                    st.markdown(f"<div style='font-size:0.85rem;color:var(--sub);'>{item['text'][:300]}...</div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div class='caption'>🔗 Query Breakdown</div>", unsafe_allow_html=True)
            for i, sq in enumerate(sub_questions):
                st.markdown(f"**{i+1}.** {sq}")

# ── Follow-up ────────────────────────────────────────────────────────────
if st.session_state.last_context:
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<div class='caption'>💬 Follow-up Questions</div>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if followup := st.chat_input("Ask a follow-up..."):
            with st.chat_message("user"):
                st.markdown(followup)
            st.session_state.chat_history.append({"role": "user", "content": followup})

            msgs = [{"role": "system", "content": f"Answer using only this context:\n{st.session_state.last_context}"}]
            for m in st.session_state.chat_history[-6:]:
                msgs.append({"role": m["role"], "content": m["content"]})

            resp = client_judge.chat.completions.create(
                messages=msgs, model="llama-3.1-8b-instant", temperature=0.2
            ).choices[0].message.content

            with st.chat_message("assistant"):
                st.markdown(resp)
            st.session_state.chat_history.append({"role": "assistant", "content": resp})
            st.rerun()