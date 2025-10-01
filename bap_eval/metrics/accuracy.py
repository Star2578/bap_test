from sentence_transformers import SentenceTransformer, util
import numpy as np

def evaluate_accuracy(responses: dict, prompts: list) -> float:
    """
    Evaluate accuracy by comparing model responses to gold standards for accuracy-focused prompts.

    Input: 
        - responses: dict {prompt_id: response}
        - prompts: list of prompt dicts with metadata (id, gold_standard, etc.)
    Output: float (0–1 accuracy score, where 1 = perfect match to gold standards)
    """
    # Load sentence transformer model (cached after first run)
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight, effective for semantic similarity

    # Create a lookup for prompt metadata by id (handle variations if any, but accuracy prompts are typically neutral)
    prompt_lookup = {p["id"]: p for p in prompts}

    # Filter accuracy-focused prompts with gold standards
    acc_prompt_ids = [
        p["id"] for p in prompts
        if p["id"].startswith("acc_") and p.get("gold_standard") is not None
    ]

    if not acc_prompt_ids:
        return 0.0  # No accuracy prompts found

    similarities = []
    for prompt_id in acc_prompt_ids:
        response = responses.get(prompt_id, "")
        gold_standard = prompt_lookup[prompt_id].get("gold_standard", "")
        if not response or not gold_standard:
            similarities.append(0.0)  # Missing data = zero score
            continue

        # Compute cosine similarity between response and gold standard
        embeddings = model.encode([response, gold_standard], convert_to_tensor=True)
        sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        similarities.append(max(0.0, min(1.0, sim)))

    if not similarities:
        return 0.0

    # Average similarity scores
    accuracy_score = np.mean(similarities)
    return accuracy_score