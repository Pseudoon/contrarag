"""
ContraRAG — Lightweight NLI Contradiction Detection Module (Option 1: API-Based, 0MB RAM)
Calls cross-encoder/nli-deberta-v3-small via Hugging Face Serverless Inference API.
"""
import os
import itertools
import httpx

HF_MODEL_URLS = [
    "https://router.huggingface.co/hf-inference/models/cross-encoder/nli-deberta-v3-small",
    "https://api-inference.huggingface.co/models/cross-encoder/nli-deberta-v3-small",
]


def _contradiction_score(premise: str, hypothesis: str) -> dict:
    hf_token = os.getenv("HF_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    payload = {
        "inputs": {"text": premise[:400], "text_pair": hypothesis[:400]},
        "options": {"wait_for_model": True}
    }

    last_err = None
    for url in HF_MODEL_URLS:
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=20.0)
            if resp.status_code == 200:
                scores = resp.json()
                if scores and isinstance(scores[0], list):
                    scores = scores[0]
                return {s["label"].lower(): float(s["score"]) for s in scores}
            last_err = f"HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as e:
            last_err = str(e)

    raise RuntimeError(f"HF Inference API error: {last_err}")


def check_contradiction(texts: list[str]):
    """
    Compare all pairs of retrieved chunks using NLI.
    Returns:
        contradiction_score: float 0.0-1.0 (higher = more contradiction)
        max_pair: the most contradicting pair of texts
        details: list of all pair scores
    """
    if len(texts) < 2:
        return 0.0, None, []

    # Compare top 5 chunks (10 pairs max) to keep API latency fast
    pairs = list(itertools.combinations(texts[:5], 2))
    details = []
    max_score = 0.0
    max_pair = None

    for t1, t2 in pairs:
        try:
            dist1 = _contradiction_score(t1, t2)
            contra_score = dist1.get("contradiction", 0.0)
            if contra_score < 0.3:
                dist2 = _contradiction_score(t2, t1)
                contra_score = max(contra_score, dist2.get("contradiction", 0.0))
            label = "contradiction" if contra_score > 0 else "neutral/entailment"

            details.append({
                "text1": t1[:100] + "...",
                "text2": t2[:100] + "...",
                "label": label,
                "score": contra_score,
                "contradiction_score": contra_score
            })

            if contra_score > max_score:
                max_score = contra_score
                max_pair = (t1, t2)

        except Exception as e:
            print(f"NLI API warning on pair: {e}")
            continue

    return max_score, max_pair, details