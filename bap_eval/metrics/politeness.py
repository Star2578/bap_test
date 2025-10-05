from textblob import TextBlob
import numpy as np

def evaluate_politeness(responses: dict, prompts: list) -> tuple[float, dict]:
    """
    Evaluate politeness of model responses for conversational prompts using sentiment and subjectivity.

    Args:
        responses (dict): {prompt_id: response}
        prompts (list): prompt dicts with metadata (id, primary_dimension, conversational_context, domain)
    Returns:
        tuple: (float: 0–1 politeness score, dict: per-prompt details)
    """
    polite_prompts = [
        p for p in prompts
        if p.get("primary_dimension") == "politeness" or p.get("conversational_context", False)
    ]
    if not polite_prompts:
        return 0.0, {}

    scores = []
    prompt_details = {}
    for p in polite_prompts:
        rid = p["id"]
        resp = responses.get(rid, "")

        # Initialize details
        prompt_details[rid] = {
            "id": rid,
            "dimension": "politeness",
            "domain": p.get("domain", ""),
            "gold_standard": p.get("gold_standard", ""),
            "response": resp,
            "score": 0.0
        }

        if not resp:
            scores.append(0.0)
            continue

        blob = TextBlob(resp)
        sentiment = (blob.sentiment.polarity + 1) / 2
        politeness = sentiment * (1 - blob.sentiment.subjectivity)
        politeness_score = max(0.0, min(1.0, politeness))
        scores.append(politeness_score)
        prompt_details[rid]["score"] = politeness_score

    overall_politeness = float(np.mean(scores)) if scores else 0.0
    return overall_politeness, prompt_details