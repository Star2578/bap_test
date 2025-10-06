#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, sys, math
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Intel/polite-guard"

def infer_label2ord(id2label: dict):
    # สร้าง mapping ordinal 0..3 จากชื่อ label ของโมเดลแบบ robust
    lmap = {}
    for k, v in id2label.items():
        name = str(v).lower().replace("-", " ").strip()
        if "impolite" in name:
            lmap[name] = 0
        elif "neutral" in name:
            lmap[name] = 1
        elif "somewhat" in name or "moderately" in name:
            lmap[name] = 2
        elif "polite" in name:
            lmap[name] = 3
        else:
            # เผื่อกรณีชื่อไม่ตรง ให้ใช้ลำดับ index
            lmap[name] = int(k)
    # กลับเป็น id->ord
    id2ord = {}
    for k, v in id2label.items():
        name = str(v).lower().replace("-", " ").strip()
        id2ord[int(k)] = lmap[name]
    return id2ord

def map_to_5level(expected_01: float, top_label: str, top_prob: float):
    """
    expected_01: ค่าเชิงคาดหวัง 0..1 จาก ordinal (0..3)/3
    กติกาหลัก: แบ่งช่วงคงที่ 0.2
    กติกาเสริม: ถ้า top_prob >= 0.85 และ top_label เป็นปลายสุด ให้ดันเป็น very_*
    """
    # base by fixed bins
    if expected_01 < 0.2:
        five = "very_impolite"
    elif expected_01 < 0.4:
        five = "impolite"
    elif expected_01 < 0.6:
        five = "neutral"
    elif expected_01 < 0.8:
        five = "polite"
    else:
        five = "very_polite"

    # edge push by confidence
    tl = top_label.lower().replace("-", " ").strip()
    if top_prob >= 0.85:
        if "impolite" in tl and five != "very_polite":
            five = "very_impolite"
        if tl == "polite" and five != "very_impolite":
            five = "very_polite"
    return five

def batched(iterable, n=32):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to input CSV")
    ap.add_argument("--output", default="scored_with_politeguard_5level.csv", help="path to output CSV")
    ap.add_argument("--text-col", default="response", help="column name containing text")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    text_col = args.text_col
    if text_col not in df.columns:
        for cand in ["response_text", "text"]:
            if cand in df.columns:
                text_col = cand
                print(f"[warn] '{args.text-col}' not found; falling back to '{cand}'", file=sys.stderr)
                break
        else:
            raise SystemExit(f"[error] text column '{args.text_col}' not found. Available: {list(df.columns)}")

    texts = df[text_col].fillna("").astype(str).tolist()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    id2label = model.config.id2label
    label_names = [id2label[i] for i in range(len(id2label))]
    id2ord = infer_label2ord(id2label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_top_label, all_top_prob = [], []
    all_expected01, all_probs = [], []

    for batch in tqdm(list(batched(texts, n=args.batch_size)), desc="Scoring"):
        enc = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # top-1
        top_idx = probs.argmax(axis=1)
        top_p = probs.max(axis=1)

        # expected ordinal in 0..1
        ord_weights = np.array([id2ord[i] for i in range(len(label_names))], dtype=float)  # 0..3
        exp_ord = (probs @ ord_weights) / 3.0  # 0..1

        all_top_label += [label_names[i] for i in top_idx]
        all_top_prob += top_p.tolist()
        all_expected01 += exp_ord.tolist()
        all_probs += probs.tolist()

    # map to 5 levels
    five_levels = [map_to_5level(e, l, p) for e, l, p in zip(all_expected01, all_top_label, all_top_prob)]
    five_ids = {"very_impolite":0, "impolite":1, "neutral":2, "polite":3, "very_polite":4}
    five_id_vals = [five_ids[x] for x in five_levels]

    # ใส่ผลกลับลง df
    df["politeguard_top_label"] = all_top_label
    df["politeguard_top_prob"]  = np.round(all_top_prob, 6)
    df["politeguard_expected01"]= np.round(all_expected01, 6)
    # เก็บ probs เป็น list/JSON string เผื่อวิเคราะห์ภายหลัง
    df["politeguard_probs_json"]= [str(p) for p in all_probs]

    df["politeness_5level"]     = five_levels
    df["politeness_5level_id"]  = five_id_vals  # 0..4

    df.to_csv(args.output, index=False)
    print(f"[done] wrote: {args.output}")
    print("[info] labels in model:", label_names)

if __name__ == "__main__":
    main()