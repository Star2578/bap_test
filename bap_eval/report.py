import os
import csv
import numpy as np

def generate_report(
    responses: dict,
    prompts: list,
    scores: dict,
    detailed_results: dict,
    output_path: str | None = None
):
    """
    Write two CSVs:

    1) <prefix>_prompt_level.csv
       Columns: prompt_id, prompt_text, response, gold_standard,
                primary_dimension, variation_key, domain, score, score_color

    2) <prefix>_domain_summary.csv
       Columns: domain, bias, accuracy, politeness
       + a final OVERALL row with overall scores.

    'output_path' is used as a *prefix*: we strip its extension and append suffixes.
    If not provided, prefix defaults to 'bap_report'.
    """
    # ---------- derive output prefix ----------
    if output_path:
        base, _ext = os.path.splitext(output_path)
        out_prefix = base
    else:
        out_prefix = "bap_report"

    prompt_level_csv   = f"{out_prefix}_prompt_level.csv"
    domain_summary_csv = f"{out_prefix}_domain_summary.csv"

    # ---------- build quick lookups ----------
    # detailed_results is expected as: {prompt_id: {"dimension": "...", "domain": "...", "score": float, ...}, ...}
    details_by_pid = detailed_results

    # Compute domain buckets from details (not from prompts) so they reflect only evaluated items
    domain_scores = {}
    for pid, d in details_by_pid.items():
        dom = d.get("domain", "")
        dim = d.get("dimension")
        sc  = d.get("score")
        if sc is None or dim not in ("bias", "accuracy", "politeness"):
            continue
        domain_scores.setdefault(dom, {"bias": [], "accuracy": [], "politeness": []})
        domain_scores[dom][dim].append(sc)

    # Aggregate domain-level averages
    domain_summary = {
        d: {
            k: (float(np.mean(v)) if v else None)
            for k, v in buckets.items()
        }
        for d, buckets in domain_scores.items()
    }

    with open(prompt_level_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_id",
                "prompt_text",
                "response",
                "gold_standard",
                "primary_dimension",
                "variation_key",
                "domain",
                "score",
            ],
        )
        writer.writeheader()

        # We iterate over 'prompts' to preserve original ordering and metadata
        for p in prompts:
            # Robust prompt_id reconstruction (mirrors runner’s export) :contentReference[oaicite:0]{index=0}
            pid = p.get("id") or p.get("qid") or p.get("prompt_id")
            if p.get("variation_key"):
                pid = f"{pid}_{p['variation_key']}"

            pid = str(pid)
            detail = details_by_pid.get(pid, {})
            score = detail.get("score")

            writer.writerow({
                "prompt_id": pid,
                "prompt_text": p.get("text") or p.get("question") or p.get("prompt") or "",
                "response": responses.get(pid, ""),
                "gold_standard": p.get("gold_standard", ""),
                "primary_dimension": p.get("primary_dimension", ""),
                "variation_key": p.get("variation_key", ""),
                "domain": p.get("domain", ""),
                "score": (None if score is None else f"{float(score):.6f}"),
            })

    # ---------- write domain-summary CSV (plus OVERALL row) ----------
    with open(domain_summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["domain", "bias", "accuracy", "politeness"]
        )
        writer.writeheader()

        # Domain rows
        for dom, agg in sorted(domain_summary.items()):
            writer.writerow({
                "domain": dom,
                "bias":        ("" if agg.get("bias")        is None else f"{agg['bias']:.6f}"),
                "accuracy":    ("" if agg.get("accuracy")    is None else f"{agg['accuracy']:.6f}"),
                "politeness":  ("" if agg.get("politeness")  is None else f"{agg['politeness']:.6f}"),
            })

        # Overall row from supplied 'scores'
        writer.writerow({
            "domain":     "OVERALL",
            "bias":       ("" if scores.get("bias")       is None else f"{scores['bias']:.6f}"),
            "accuracy":   ("" if scores.get("accuracy")   is None else f"{scores['accuracy']:.6f}"),
            "politeness": ("" if scores.get("politeness") is None else f"{scores['politeness']:.6f}"),
        })

    # Return a small dict for programmatic checks (optional)
    return {
        "prompt_level_path": prompt_level_csv,
        "domain_summary_path": domain_summary_csv,
        "overall_scores": scores,
    }
