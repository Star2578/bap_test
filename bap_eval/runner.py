from .datasets.default_prompts import DEFAULT_PROMPTS
from .metrics.bias import evaluate_bias
from .metrics.accuracy import evaluate_accuracy
from .metrics.politeness import evaluate_politeness
from .scoring import compute_pei

def run_bap_test(wrapper):
    """
    Run Bias, Accuracy, and Politeness evaluation on a given model wrapper.

    Args:
        wrapper (BaseModelWrapper): A model wrapper instance (HF, OpenAI, Ollama).

    Returns:
        dict: A dictionary with keys:
            - "bias": float, 0–1 score
            - "accuracy": float, 0–1 score
            - "politeness": float, 0–1 score
            - "PEI": float, composite score
    """
    # Step 1: generate responses
    responses = {p["id"]: wrapper.generate(p["text"]) for p in DEFAULT_PROMPTS}

    # Step 2: evaluate (placeholders for now)
    bias_score = evaluate_bias(responses)
    accuracy_score = evaluate_accuracy(responses)
    politeness_score = evaluate_politeness(responses)

    # Step 3: compute PEI
    pei = compute_pei(bias_score, accuracy_score, politeness_score)

    return {
        "bias": bias_score,
        "accuracy": accuracy_score,
        "politeness": politeness_score,
        "PEI": pei
    }
