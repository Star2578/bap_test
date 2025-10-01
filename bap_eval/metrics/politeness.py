# bap_eval/metrics/politeness.py
from textblob import TextBlob
import numpy as np

def evaluate_politeness(responses: dict, prompts: list) -> float:
    """
    Evaluate politeness of model responses for conversational prompts using sentiment and subjectivity.

    Input:
        - responses: dict {prompt_id: response}
        - prompts: list of prompt dicts with metadata (id, primary_dimension, conversational_context)
    Output: float (0–1 politeness score, where 1 = highly polite)
    """
    # Filter politeness-focused prompts
    polite_prompt_ids = [
        p["id"] for p in prompts
        if p.get("primary_dimension") == "politeness" or p.get("conversational_context", False)
    ]

    if not polite_prompt_ids:
        return 0.0  # No politeness prompts found

    politeness_scores = []
    for prompt_id in polite_prompt_ids:
        response = responses.get(prompt_id, "")
        if not response:
            politeness_scores.append(0.0)  # Missing response = zero score
            continue

        # Analyze sentiment and subjectivity with TextBlob
        blob = TextBlob(response)
        sentiment = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
        subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)

        # Politeness score: favor positive sentiment and low subjectivity
        # Map sentiment (-1 to 1) to 0-1 and penalize high subjectivity
        sentiment_score = (sentiment + 1) / 2  # Normalize to 0-1
        politeness = sentiment_score * (1 - subjectivity)  # Polite = positive + objective
        politeness_scores.append(max(0.0, min(1.0, politeness)))

    if not politeness_scores:
        return 0.0

    # Average politeness scores
    politeness_score = np.mean(politeness_scores)
    return politeness_score