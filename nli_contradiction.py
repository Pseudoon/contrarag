"""
ContraRAG — Lightweight NLI Contradiction Detection Module (API-Based)
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
            print(f"HF API Status: {resp.status_code}, Response: {resp.text[:200]}") # Debug log
            
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    scores = data[0] if isinstance(data[0], list) else data
                    return {str(s.get("label", "")).lower(): float(s.get("score", 0.0)) for s in scores}
            
            last_err = f"HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as e:
            last_err = str(e)

    print(f"NLI API Error across all URLs: {last_err}")
    return {"contradiction": 0.0}


def check_contradiction(texts: list[str]):
    if len(texts) < 2:
        return 0.0, None, []

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
            
            label = "contradiction" if contra_score > 0.1 else "neutral/entailment"

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
            print(f"NLI pair error: {e}")
            continue

    return max_score, max_pair, details