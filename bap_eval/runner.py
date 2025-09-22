from tqdm import tqdm
import logging
from .datasets.default_prompts import DEFAULT_PROMPTS
from .metrics.bias import evaluate_bias
from .metrics.accuracy import evaluate_accuracy
from .metrics.politeness import evaluate_politeness
from .scoring import compute_pei

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_bap_test(wrapper, verbose: bool = False) -> dict:
    """
    Run Bias, Accuracy, and Politeness evaluation on a given model wrapper with progress feedback.

    Args:
        wrapper (BaseModelWrapper): A model wrapper instance (HF, OpenAI, Ollama).
        verbose (bool, optional): If True, log detailed information about each prompt being processed.

    Returns:
        dict: A dictionary with keys:
            - "bias": float, 0–1 score
            - "accuracy": float, 0–1 score
            - "politeness": float, 0–1 score
            - "PEI": float, composite score
    """
    # Step 1: Generate responses with progress bar
    responses = {}
    for p in tqdm(DEFAULT_PROMPTS, desc="Generating responses", unit="prompt"):
        prompt_id = p["id"]
        prompt_text = p["text"]
        if verbose:
            logger.info(f"Processing prompt ID {prompt_id}: {prompt_text[:50]}...")
        try:
            response = wrapper.generate(prompt_text)
            responses[prompt_id] = response
        except Exception as e:
            logger.error(f"Error processing prompt ID {prompt_id}: {str(e)}")
            responses[prompt_id] = ""  # Store empty response on failure to continue evaluation

    # Step 2: Evaluate metrics
    logger.info("Evaluating bias...")
    bias_score = evaluate_bias(responses)
    logger.info("Evaluating accuracy...")
    accuracy_score = evaluate_accuracy(responses)
    logger.info("Evaluating politeness...")
    politeness_score = evaluate_politeness(responses)

    # Step 3: Compute PEI
    logger.info("Computing PEI score...")
    pei = compute_pei(bias_score, accuracy_score, politeness_score)

    # Step 4: Return results
    logger.info("Evaluation complete.")
    return {
        "bias": bias_score,
        "accuracy": accuracy_score,
        "politeness": politeness_score,
        "PEI": pei
    }