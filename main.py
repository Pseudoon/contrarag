"""
ContraRAG — FastAPI Backend (Option 1: Ultra-Low Memory for Render 512MB Free Tier)
"""

import os
import gc
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from nli_contradiction import check_contradiction

load_dotenv()

app = FastAPI(title="ContraRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading lightweight ONNX models...")
embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", threads=1)
reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2", threads=1)
gc.collect()

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30
)

client_eu = Groq(api_key=os.getenv("GROQ_API_KEY_EU") or os.getenv("GROQ_API_KEY"))
client_us = Groq(api_key=os.getenv("GROQ_API_KEY_US") or os.getenv("GROQ_API_KEY"))
client_uk = Groq(api_key=os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_JUDGE"))
client_judge = Groq(api_key=os.getenv("GROQ_API_KEY_JUDGE") or os.getenv("GROQ_API_KEY"))
print("All models loaded")

MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


class QueryRequest(BaseModel):
    question: str
    jurisdictions: list[str] = ["EU", "US", "UK"]


class ChatRequest(BaseModel):
    message: str
    context: str
    history: list[dict] = []


def strip_think(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def call_groq_with_retry(client: Groq, messages: list[dict], temperature: float = 0.2, max_tokens: int = 300) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                messages=messages,
                model=MODEL,
                temperature=temperature,
                max_tokens=max_tokens
            ).choices[0].message.content
            return strip_think(resp)
        except RateLimitError:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return ""


def decompose_query(query: str):
    prompt = f"""Analyze this question: "{query}"
If SIMPLE (one jurisdiction, one concept): {{"is_complex": false, "sub_questions": ["{query}"]}}
If COMPLEX (comparison, multiple jurisdictions, multiple concepts): {{"is_complex": true, "sub_questions": ["sub-q 1", "sub-q 2"]}}
Return ONLY valid JSON, no extra text."""
    try:
        resp = call_groq_with_retry(
            client_judge,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        start = resp.find('{')
        end = resp.rfind('}') + 1
        r = json.loads(resp[start:end])
        return r.get("sub_questions", [query]), r.get("is_complex", False)
    except Exception:
        return [query], False


def search_qdrant(query: str, jurisdictions: list[str], sub_questions: list[str]):
    all_raw = []
    for sub_q in sub_questions:
        vector = list(embed_model.embed([sub_q]))[0].tolist()
        for region in jurisdictions:
            results = qdrant.query_points(
                collection_name="ai_regulations",
                query=vector,
                limit=3,
                query_filter=Filter(
                    must=[FieldCondition(key="jurisdiction", match=MatchValue(value=region))]
                )
            ).points
            for r in results:
                all_raw.append({
                    "text": r.payload["text"],
                    "source": r.payload.get("source_file", r.payload.get("source", "unknown")),
                    "jurisdiction": r.payload.get("jurisdiction", "unknown"),
                    "score": r.score,
                    "sub_query": sub_q
                })

    seen, unique = set(), []
    for r in all_raw:
        key = r["text"][:100]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    if not unique:
        return "", []

    docs = [c["text"] for c in unique]
    scores = list(reranker.rerank(query, docs))
    for i, c in enumerate(unique):
        c["rerank_score"] = float(scores[i])
    unique.sort(key=lambda x: x["rerank_score"], reverse=True)

    TOP_N = 5
    per_region_guarantee = 1 if len(jurisdictions) >= 3 else 2

    top, used_keys = [], set()
    for region in jurisdictions:
        region_chunks = [c for c in unique if c["jurisdiction"] == region]
        for c in region_chunks[:per_region_guarantee]:
            key = c["text"][:100]
            if key not in used_keys:
                top.append(c)
                used_keys.add(key)

    for c in unique:
        if len(top) >= TOP_N:
            break
        key = c["text"][:100]
        if key not in used_keys:
            top.append(c)
            used_keys.add(key)

    top.sort(key=lambda x: x["rerank_score"], reverse=True)

    context = "\n\n---\n\n".join(
        [f"[{c['jurisdiction']}] {c['source']}\n{c['text'][:600]}" for c in top]
    )
    gc.collect()
    return context, top


def run_agents(query: str, context: str, jurisdictions: list[str]):
    def region_context(region: str) -> str:
        lines = [l for l in context.split("\n\n---\n\n") if l.startswith(f"[{region}]")]
        return "\n\n---\n\n".join(lines) if lines else context

    def run_one(region: str, client: Groq) -> str:
        ctx = region_context(region)
        prompt = f"""You are a {region} AI regulation expert. Answer using ONLY the {region} context below, in 3-4 sentences.
If the context contains no {region}-specific material, say so explicitly instead of describing another jurisdiction's rules.

Context:
{ctx}

Question: {query}"""
        try:
            return call_groq_with_retry(
                client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
        except RateLimitError:
            return f"[{region} agent unavailable: Groq rate limit reached. Try again in a moment.]"

    clients = {"EU": client_eu, "US": client_us, "UK": client_uk}
    active = [(r, clients[r]) for r in jurisdictions if r in clients]

    def run_staggered(region: str, client: Groq, delay: float) -> str:
        time.sleep(delay)
        return run_one(region, client)

    responses = {}
    with ThreadPoolExecutor(max_workers=len(active) or 1) as pool:
        futures = {
            pool.submit(run_staggered, region, client, i * 0.8): region
            for i, (region, client) in enumerate(active)
        }
        for future in as_completed(futures):
            region = futures[future]
            responses[region] = future.result()

    return responses


def run_judge(query: str, context: str, responses: dict, nli_score: float, nli_decision: str, nli_contradiction_found: bool):
    judge_prompt = f"""You are ContraRAG's arbitrator. The NLI model has already determined:
- Contradiction score: {nli_score:.2f} (0=no contradiction, 1=strong contradiction)
- Contradiction found: {nli_contradiction_found}
- Decision: {nli_decision}

Based on the agent responses below, output ONLY valid JSON with no extra text:
{{
  "regional_summaries": {{
    "EU": "2-3 sentence summary of EU position",
    "US": "2-3 sentence summary of US position",
    "UK": "2-3 sentence summary of UK position"
  }},
  "contradiction_explanation": "Specific explanation of what contradicts, or No contradiction found",
  "decision_reason": "Why this decision was made"
}}

Use "Not queried for this request." for any jurisdiction below that has no agent response.

{chr(10).join(f"{region} Agent: {resp[:600]}" for region, resp in responses.items())}"""

    try:
        raw = call_groq_with_retry(
            client_judge,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1,
            max_tokens=500
        )
        cleaned = re.sub(r'```(?:json)?', '', raw).strip()
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        return json.loads(cleaned[start:end])
    except Exception:
        return {
            "regional_summaries": {
                region: resp[:300] for region, resp in responses.items()
            },
            "contradiction_explanation": "Could not parse judge response",
            "decision_reason": nli_decision
        }


@app.get("/")
def root():
    return {"service": "ContraRAG API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok", "chunks": 3982, "model": MODEL}


@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")

    sub_questions, is_complex = decompose_query(req.question)
    context, trail = search_qdrant(req.question, req.jurisdictions, sub_questions)

    if not context:
        raise HTTPException(404, "No relevant documents found")

    responses = run_agents(req.question, context, req.jurisdictions)

    chunk_texts = [c["text"] for c in trail]
    nli_score, nli_max_pair, nli_details = check_contradiction(chunk_texts)
    nli_confidence = round(1.0 - nli_score, 2)

    if nli_confidence > 0.75:
        nli_decision = "ANSWER"
    elif nli_confidence > 0.4:
        nli_decision = "WARN"
    else:
        nli_decision = "REFUSE"

    # --- RELEVANCE / WEAK JURISDICTION GAP CHECK ---
    # Downgrade absolute ANSWER to WARN if any queried jurisdiction returned zero relevant/positive chunks (rerank_score > 0)
    weak = [
        j for j in req.jurisdictions 
        if not any(c["jurisdiction"] == j and c.get("rerank_score", 0) > 0 for c in trail)
    ]
    if weak and nli_decision == "ANSWER":
        nli_decision = "WARN"
        nli_confidence = 0.85

    nli_contradiction_found = nli_decision != "ANSWER"

    judge_data = run_judge(
        req.question, context, responses,
        nli_score, nli_decision, nli_contradiction_found
    )

    gc.collect()
    return {
        "sub_questions": sub_questions,
        "is_complex": is_complex,
        "eu_response": responses.get("EU", "Not queried for this request."),
        "us_response": responses.get("US", "Not queried for this request."),
        "uk_response": responses.get("UK", "Not queried for this request."),
        "sources": trail,
        "contradiction": {
            "found": nli_contradiction_found,
            "nli_score": nli_score,
            "explanation": judge_data.get("contradiction_explanation", ""),
            "max_pair": nli_max_pair
        },
        "verdict": {
            "decision": nli_decision,
            "confidence": nli_confidence,
            "friction_score": round(nli_score * 10),
            "decision_reason": judge_data.get("decision_reason", ""),
            "regional_summaries": judge_data.get("regional_summaries", {}),
        },
        "context": context
    }


@app.post("/chat")
def chat(req: ChatRequest):
    msgs = [{"role": "system", "content": f"Answer using only this context:\n{req.context[:3000]}"}]
    for m in req.history[-6:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": req.message})

    try:
        resp = call_groq_with_retry(
            client_judge,
            messages=msgs,
            temperature=0.2,
            max_tokens=500
        )
        return {"response": resp}
    except RateLimitError:
        raise HTTPException(429, "Groq rate limit reached. Please try again in a moment.")