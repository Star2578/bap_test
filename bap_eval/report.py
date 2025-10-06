import numpy as np

def generate_report(responses: dict, prompts: list, scores: dict, detailed_results: dict, output_path: str = None):
    """
    Generate a detailed evaluation report.

    responses: {prompt_id: response}
    prompts: list of prompt dicts
    scores: overall scores (bias, accuracy, politeness, PEI)
    detailed_results: per-prompt results from each evaluator
    output_path: if set, saves report as JSON
    """
    import json
    from collections import defaultdict

    # Organize by domain
    domain_scores = defaultdict(lambda: {"bias": [], "accuracy": [], "politeness": []})
    for pid, detail in detailed_results.items():
        domain = detail["domain"]
        dim = detail.get("dimension")
        score = detail.get("score")
        if dim in domain_scores[domain] and score is not None:
            domain_scores[domain][dim].append(score)

    # Aggregate domain-level averages
    domain_summary = {
        d: {dim: (float(np.mean(vals)) if vals else None)
            for dim, vals in dims.items()}
        for d, dims in domain_scores.items()
    }

    report = {
        "overall_scores": scores,
        "domain_summary": domain_summary,
        "prompt_level": detailed_results
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report
