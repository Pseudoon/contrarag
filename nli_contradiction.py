"""
ContraRAG — NLI Contradiction Detection Module
Drop this file into your project folder.
It is imported by app.py.
"""
from transformers import pipeline
import itertools

# Load once — this downloads ~90MB model first time
print("Loading NLI model...")
nli = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=-1,  # CPU
    top_k=None  # return scores for ALL labels (contradiction/entailment/neutral),
                # not just the argmax — needed below.
)

def _contradiction_score(premise, hypothesis):
    """Run NLI one direction and pull out the contradiction label's score
    directly from the full distribution, regardless of which label wins.
    Using nli(text, text_pair=...) — the documented call signature —
    instead of a raw dict, since dict-style calls are version-fragile and
    were failing silently here."""
    scores = nli(premise[:400], text_pair=hypothesis[:400], truncation=True, max_length=512)
    # top_k=None gives a list of {"label":..., "score":...} for one input
    if scores and isinstance(scores[0], list):
        scores = scores[0]
    return {s["label"].lower(): s["score"] for s in scores}


def check_contradiction(texts):
    """
    Compare all pairs of retrieved chunks using NLI.
    Returns:
        contradiction_score: float 0.0-1.0 (higher = more contradiction)
        max_pair: the most contradicting pair of texts
        details: list of all pair scores
    """
    if len(texts) < 2:
        return 0.0, None, []

    pairs = list(itertools.combinations(texts[:6], 2))  # max 15 pairs
    details = []
    max_score = 0.0
    max_pair = None

    for t1, t2 in pairs:
        try:
            # NLI is directional; only pay for the second (reverse) pass
            # when the first direction found nothing, instead of always
            # running both — this was doubling NLI latency for no benefit
            # in the common case.
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
            # Surfaced loudly on purpose — a silently-swallowed exception
            # here is exactly what made every score read as 0.000 before.
            import traceback
            print(f"NLI ERROR on pair: {e}")
            traceback.print_exc()
            continue

    return max_score, max_pair, details