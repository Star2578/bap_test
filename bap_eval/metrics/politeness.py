from transformers import pipeline
import numpy as np

def evaluate_politeness(responses: dict, prompts: list) -> tuple[float, dict]:
    """
    Evaluate politeness of model responses for conversational prompts
    using a politeness classifier.

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

    classifier = pipeline(
        "text-classification",
        model="Intel/polite-guard",
        truncation=True,
        max_length=512,
        top_k=None
    )

    scores = []
    prompt_details = {}
    for p in polite_prompts:
        rid = p["id"]
        resp = responses.get(rid, "")

        prompt_details[rid] = {
            "id": rid,
            "dimension": "politeness",
            "domain": p.get("domain", ""),
            "response": resp,
            "score": None,
            "pred_label": None,
            "all_labels": None
        }

        if not resp:
            scores.append(0.0)
            prompt_details[rid]["score"] = 0.0
            prompt_details[rid]["pred_label"] = "empty"
            continue

        result = classifier(resp)
        # result is list of dicts: e.g. [{'label': 'polite', 'score': 0.64}, {'label': 'impolite', 'score': 0.10}, ...]
        prompt_details[rid]["all_labels"] = result
        # pick the probability of the “polite” label (you could also map other labels)
        polite_score = 0.0
        for entry in result:
            if entry["label"].lower() == "polite":
                polite_score = entry["score"]
                break

        prompt_details[rid]["score"] = polite_score
        prompt_details[rid]["pred_label"] = max(result, key=lambda x: x["score"])["label"]
        scores.append(polite_score)

    overall = float(np.mean(scores)) if scores else 0.0
    return overall, prompt_details