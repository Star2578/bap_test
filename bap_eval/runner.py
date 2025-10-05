from tqdm import tqdm
import logging
import csv
import os
from .datasets.prompt_dataset import generate_full_prompt_set
from .metrics.bias import evaluate_bias
from .metrics.accuracy import evaluate_accuracy
from .metrics.politeness import evaluate_politeness
from .scoring import compute_pei
from .report import generate_report

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_bap_test(wrapper=None, verbose: bool = False, export_csv: str = None, import_csv: str = None, report_output: str = "bap_report.json") -> dict:
    """
    Run Bias, Accuracy, and Politeness evaluation on a given model wrapper with progress feedback.

    Args:
        wrapper (BaseModelWrapper, optional): A model wrapper instance (HF, OpenAI, Ollama). Required if import_csv is not provided.
        verbose (bool, optional): If True, log detailed information about each prompt being processed.
        export_csv (str, optional): Path to save generated responses as CSV (e.g., 'responses.csv').
        import_csv (str, optional): Path to load responses from CSV instead of generating them.
        report_output (str, optional): Path to save the detailed report as JSON (e.g., 'bap_report.json').

    Returns:
        dict: A dictionary with keys:
            - "bias": float, 0–1 score
            - "accuracy": float, 0–1 score
            - "politeness": float, 0–1 score
            - "PEI": float, composite score
    """
    # Validate arguments
    if import_csv and wrapper:
        logger.warning("Both import_csv and wrapper provided. Ignoring wrapper and loading from CSV.")
        wrapper = None
    elif not import_csv and not wrapper:
        raise ValueError("Must provide either a wrapper or import_csv path.")

    # Generate full prompt set (with variations for bias)
    prompts = generate_full_prompt_set(include_variations=True)

    # Step 1: Load or generate responses
    responses = {}
    if import_csv:
        # Import responses from CSV
        logger.info(f"Loading responses from {import_csv}...")
        try:
            with open(import_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    responses[row['prompt_id']] = row['response']
            logger.info(f"Loaded {len(responses)} responses from {import_csv}.")
        except FileNotFoundError:
            logger.error(f"CSV file {import_csv} not found.")
            raise
        except Exception as e:
            logger.error(f"Error reading CSV {import_csv}: {str(e)}")
            raise
    else:
        # Generate responses with progress bar
        for p in tqdm(prompts, desc="Generating responses", unit="prompt"):
            prompt_id = p["id"]
            if "variation_key" in p:
                prompt_id += f"_{p['variation_key']}"  # Unique ID for variations
            prompt_text = p["text"]
            if verbose:
                logger.info(f"Processing prompt ID {prompt_id}: {prompt_text[:50]}...")
            try:
                response = wrapper.generate(prompt_text)
                responses[prompt_id] = response
            except Exception as e:
                logger.error(f"Error processing prompt ID {prompt_id}: {str(e)}")
                responses[prompt_id] = ""  # Store empty response on failure

        # Export responses to CSV if specified
        if export_csv:
            logger.info(f"Exporting responses to {export_csv}...")
            try:
                with open(export_csv, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            'prompt_id', 'prompt_text', 'response', 'gold_standard',
                            'primary_dimension', 'variation_key', 'domain'
                        ]
                    )
                    writer.writeheader()
                    for p in prompts:
                        prompt_id = p["id"]
                        if "variation_key" in p:
                            prompt_id += f"_{p['variation_key']}"
                        writer.writerow({
                            'prompt_id': prompt_id,
                            'prompt_text': p['text'],
                            'response': responses.get(prompt_id, ""),
                            'gold_standard': p.get('gold_standard', ''),
                            'primary_dimension': p.get('primary_dimension', ''),
                            'variation_key': p.get('variation_key', ''),
                            'domain': p.get('domain', '')
                        })
                logger.info(f"Exported {len(responses)} responses to {export_csv}.")
            except Exception as e:
                logger.error(f"Error exporting to CSV {export_csv}: {str(e)}")
                raise

    # Step 2: Evaluate metrics (pass prompts for metadata access)
    logger.info("Evaluating bias...")
    bias_score, bias_details = evaluate_bias(responses, prompts)
    logger.info(f"Bias score: {bias_score:.2f}")

    logger.info("Evaluating accuracy...")
    accuracy_score, acc_details = evaluate_accuracy(responses, prompts)
    logger.info(f"Accuracy score: {accuracy_score:.2f}")

    logger.info("Evaluating politeness...")
    politeness_score, pol_details = evaluate_politeness(responses, prompts)
    logger.info(f"Politeness score: {politeness_score:.2f}")

    # Step 3: Compute PEI
    logger.info("Computing PEI score...")
    pei = compute_pei(bias_score, accuracy_score, politeness_score)

    logger.info("Evaluation complete.")
    score = {
        "bias": bias_score,
        "accuracy": accuracy_score,
        "politeness": politeness_score,
        "PEI": pei
    }
    
    detailed_results = {**bias_details, **acc_details, **pol_details}
    report = generate_report(responses, prompts, score, detailed_results, output_path=report_output)
    logger.info(f"Report generated and saved to {report_output}.")
    return score