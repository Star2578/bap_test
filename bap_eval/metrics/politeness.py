from transformers import pipeline
import numpy as np

def evaluate_politeness(responses: dict, prompts: list) -> tuple[float, dict]:
    """
    Evaluate politeness of model responses for conversational prompts
    using a politeness classifier (RoBERTa fine-tuned).

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

    # Load politeness classifier once
    classifier = pipeline(
        "text-classification",
        model="salesken/politeness-classifier-roberta",
        truncation=True,
        max_length=512,
        top_k=None
    )

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

        # Classify politeness
        result = classifier(resp)[0]
        # result looks like: [{'label': 'Polite', 'score': 0.87}, {'label': 'Impolite', 'score': 0.13}]
        polite_score = 0.0
        for r in result:
            if r["label"].lower() == "polite":
                polite_score = r["score"]
                break

        scores.append(polite_score)
        prompt_details[rid]["score"] = polite_score

    overall_politeness = float(np.mean(scores)) if scores else 0.0
    return overall_politeness, prompt_details
