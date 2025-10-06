from sentence_transformers import SentenceTransformer, util
import numpy as np

def evaluate_accuracy(responses: dict, prompts: list) -> tuple[float, dict]:
    """
    Hybrid accuracy evaluation:
    - Semantic similarity between response and gold standard
    - Fact coverage: proportion of gold facts mentioned in response
    Final = average of both

    Args:
        responses (dict): {prompt_id: response}
        prompts (list): prompt dicts with metadata (id, primary_dimension, gold_standard, domain)
    Returns:
        tuple: (float: 0–1 accuracy score, dict: per-prompt details)
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    acc_prompts = [p for p in prompts if p.get("primary_dimension") == "accuracy" and p.get("gold_standard")]
    if not acc_prompts:
        return 0.0, {}

    scores = []
    prompt_details = {}
    for p in acc_prompts:
        rid = p["id"]
        resp = responses.get(rid, "")
        gold = p["gold_standard"]

        # Initialize details
        prompt_details[rid] = {
            "id": rid,
            "dimension": "accuracy",
            "domain": p.get("domain", ""),
            "gold_standard": gold,
            "question": p.get("text", ""),
            "response": resp,
            "score": 0.0
        }

        if not resp or not gold:
            scores.append(0.0)
            continue

        # Semantic similarity
        embeddings = model.encode([resp, gold], convert_to_tensor=True)
        sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        sim = max(0.0, min(1.0, sim))

        # Fact coverage
        gold_facts = [fact.strip().lower() for fact in gold.split(",") if fact.strip()]
        if not gold_facts:
            coverage = sim  # Fallback if gold not list-like
        else:
            hits = sum(1 for fact in gold_facts if fact in resp.lower())
            coverage = hits / len(gold_facts)

        # Hybrid score
        final_score = (sim + coverage) / 2
        scores.append(final_score)
        prompt_details[rid]["score"] = final_score

    overall_accuracy = float(np.mean(scores)) if scores else 0.0
    return overall_accuracy, prompt_details