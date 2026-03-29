import os
import json
import re
import time
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from transformers import AutoTokenizer

from model import TinyBertMultiTaskSpan, CLS_TASKS

# -------------------------
# Paths / Config
# -------------------------
DATA_PATH = "data/processed/data.csv"
MAP_JSON  = "data/processed/label_maps.json"

BEST_DIR  = "outputs/runs/run_001/best_model"
OUT_DIR   = "outputs/runs/run_001"

# FP16 deployment export folder (this is the real deploy size)
FP16_DIR = "outputs/exports/fp16_span_model"

MODEL_NAME_FALLBACK = "prajjwal1/bert-tiny"
MAX_LEN = 64
NONE = "NONE"

# -------------------------
# Formatting helpers
# -------------------------
def pct(x: float, digits: int = 1) -> str:
    """0.942 -> '94.2%'"""
    return f"{x * 100:.{digits}f}%"

def mb(x: float, digits: int = 2) -> str:
    """34.5 -> '34.50 MB'"""
    return f"{x:.{digits}f} MB"

# -------------------------
# Utility helpers
# -------------------------
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "classification_reports"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "confusion_matrices"), exist_ok=True)

def get_folder_size_mb(path: str) -> float:
    total = 0
    if not os.path.exists(path):
        return 0.0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)

def norm_ref_for_match(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"^(the|a|an)\s+", "", x)
    return x

def find_char_span(command: str, ref: str):
    """Find (start_char, end_char) of ref inside command (case-insensitive)."""
    if ref == NONE:
        return None
    cmd = command.lower()
    rr = ref.lower()
    idx = cmd.find(rr)
    if idx == -1:
        return None
    return idx, idx + len(rr)

def charspan_to_tokens(offsets, char_start, char_end):
    """
    offsets: list of (s,e) for each token
    returns (tok_start, tok_end) or None
    """
    tok_start = None
    tok_end = None

    for i, (s, e) in enumerate(offsets):
        if s == e == 0:
            continue
        if tok_start is None and s <= char_start < e:
            tok_start = i
        if tok_start is not None and s < char_end <= e:
            tok_end = i
            break

    if tok_start is not None and tok_end is None:
        for i, (s, e) in enumerate(offsets):
            if s == e == 0:
                continue
            if e >= char_end:
                tok_end = i
                break

    if tok_start is None or tok_end is None or tok_end < tok_start:
        return None
    return tok_start, tok_end

def load_maps():
    with open(MAP_JSON, "r", encoding="utf-8") as f:
        maps = json.load(f)
    id2label = {t: {int(k): v for k, v in maps[t]["id2label"].items()} for t in CLS_TASKS}
    return maps, id2label

def rebuild_test_split(df):
    """
    Must match train.py split logic:
    train=80%, val=10%, test=10% stratified by intent_id
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["intent_id"]
    )
    _, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["intent_id"]
    )
    return test_df.reset_index(drop=True)

def build_dataset(df, tokenizer):
    base_cols = ["command", "spatial_reference"] + [t + "_id" for t in CLS_TASKS]
    ds = Dataset.from_pandas(df[base_cols].copy())
    ds = ds.rename_columns({t + "_id": t for t in CLS_TASKS})

    def tok_and_span(batch):
        enc = tokenizer(
            batch["command"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_offsets_mapping=True
        )

        ref_starts = []
        ref_ends = []

        for cmd, ref, offsets in zip(batch["command"], batch["spatial_reference"], enc["offset_mapping"]):
            ref_norm = norm_ref_for_match(ref)
            if ref_norm == NONE:
                ref_starts.append(-100)
                ref_ends.append(-100)
                continue

            span = find_char_span(cmd, ref_norm)
            if span is None:
                ref_starts.append(-100)
                ref_ends.append(-100)
                continue

            cs, ce = span
            tok_span = charspan_to_tokens(offsets, cs, ce)
            if tok_span is None:
                ref_starts.append(-100)
                ref_ends.append(-100)
            else:
                s, e = tok_span
                ref_starts.append(s)
                ref_ends.append(e)

        enc.pop("offset_mapping")
        enc["ref_start"] = ref_starts
        enc["ref_end"] = ref_ends
        return enc

    ds = ds.map(tok_and_span, batched=True)

    cols = ["input_ids", "attention_mask"] + CLS_TASKS + ["ref_start", "ref_end"]
    if "token_type_ids" in ds.column_names:
        cols.insert(2, "token_type_ids")

    ds.set_format(type="torch", columns=cols)
    return ds

def load_model_and_tokenizer(maps):
    base_model_path = os.path.join(BEST_DIR, "base_model.txt")
    base_model = MODEL_NAME_FALLBACK
    if os.path.exists(base_model_path):
        base_model = open(base_model_path, "r", encoding="utf-8").read().strip()

    tokenizer = AutoTokenizer.from_pretrained(BEST_DIR, use_fast=True)

    num_labels = {t: maps[t]["num_labels"] for t in CLS_TASKS}
    model = TinyBertMultiTaskSpan(base_model, num_labels=num_labels)

    state_path = os.path.join(BEST_DIR, "multitask_span_state.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing {state_path}. Train the span model first.")

    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device

@torch.no_grad()
def predict_all(model, ds, device, batch_size=64):
    cls_preds = {t: [] for t in CLS_TASKS}
    cls_gold  = {t: [] for t in CLS_TASKS}

    start_preds, end_preds = [], []
    start_gold, end_gold = [], []

    for i in range(0, len(ds), batch_size):
        batch = ds[i:i+batch_size]

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

        cls_logits, start_logits, end_logits = model(**kwargs)

        for t in CLS_TASKS:
            cls_preds[t].append(torch.argmax(cls_logits[t], dim=1).cpu().numpy())
            cls_gold[t].append(batch[t].cpu().numpy())

        start_preds.append(torch.argmax(start_logits, dim=1).cpu().numpy())
        end_preds.append(torch.argmax(end_logits, dim=1).cpu().numpy())

        start_gold.append(batch["ref_start"].cpu().numpy())
        end_gold.append(batch["ref_end"].cpu().numpy())

    cls_preds = {t: np.concatenate(cls_preds[t]) for t in CLS_TASKS}
    cls_gold  = {t: np.concatenate(cls_gold[t]) for t in CLS_TASKS}
    start_preds = np.concatenate(start_preds)
    end_preds   = np.concatenate(end_preds)
    start_gold  = np.concatenate(start_gold)
    end_gold    = np.concatenate(end_gold)

    return cls_preds, cls_gold, start_preds, end_preds, start_gold, end_gold

@torch.no_grad()
def benchmark_inference(model, tokenizer, device, n_runs=300):
    """
    Benchmarks average latency per single sample (ms/sample).
    Runs on current device (CPU/GPU).
    """
    texts = [
        "Click the red button below the cart",
        "Open the settings tab",
        "Delete the last image",
        "Tap the icon next to the menu",
        "Scroll to the last item in the list",
    ]

    def run_one(text):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        model(**enc)

    # warmup
    for _ in range(20):
        run_one(texts[0])

    start = time.perf_counter()
    for i in range(n_runs):
        run_one(texts[i % len(texts)])
    end = time.perf_counter()

    avg_ms = (end - start) * 1000.0 / n_runs
    return float(avg_ms)

# -------------------------
# Main
# -------------------------
def main():
    ensure_dirs()

    df = pd.read_csv(DATA_PATH)
    maps, id2label = load_maps()
    test_df = rebuild_test_split(df)

    model, tokenizer, device = load_model_and_tokenizer(maps)
    test_ds = build_dataset(test_df, tokenizer)

    cls_preds, cls_gold, sp_s, sp_e, gd_s, gd_e = predict_all(
        model, test_ds, device=device, batch_size=64
    )

    metrics = {}

    # ---- Classification metrics ----
    for t in CLS_TASKS:
        metrics[f"{t}_acc"] = float(accuracy_score(cls_gold[t], cls_preds[t]))
        metrics[f"{t}_f1_weighted"] = float(f1_score(cls_gold[t], cls_preds[t], average="weighted", zero_division=0))
        metrics[f"{t}_f1_macro"] = float(f1_score(cls_gold[t], cls_preds[t], average="macro", zero_division=0))

    # ---- Span EM ----
    mask = gd_s != -100
    if mask.sum() == 0:
        metrics["spatial_reference_span_em"] = 0.0
    else:
        metrics["spatial_reference_span_em"] = float(np.mean((sp_s[mask] == gd_s[mask]) & (sp_e[mask] == gd_e[mask])))

    # ---- End-to-end exact match ----
    all_ok = np.ones(len(gd_s), dtype=bool)
    for t in CLS_TASKS:
        all_ok &= (cls_preds[t] == cls_gold[t])

    span_ok = ((gd_s == -100) & (gd_e == -100)) | ((sp_s == gd_s) & (sp_e == gd_e))
    all_ok &= span_ok

    metrics["exact_match_all_fields_with_span"] = float(np.mean(all_ok))
    metrics["exact_match_intent_target"] = float(np.mean(
        (cls_preds["intent"] == cls_gold["intent"]) &
        (cls_preds["target_type"] == cls_gold["target_type"])
    ))

    # ---- Size metrics (both) ----
    metrics["train_checkpoint_size_mb"] = round(get_folder_size_mb(BEST_DIR), 2)
    metrics["fp16_deploy_size_mb"] = round(get_folder_size_mb(FP16_DIR), 2)

    # ---- Inference ----
    metrics["avg_inference_time_ms"] = round(benchmark_inference(model, tokenizer, device, n_runs=300), 2)

    # ---- Save metrics ----
    metrics_path = os.path.join(OUT_DIR, "metrics_eval.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ---- Reports + confusion matrices ----
    for t in CLS_TASKS:
        label_ids = list(range(maps[t]["num_labels"]))
        target_names = [id2label[t][i] for i in label_ids]

        report = classification_report(
            cls_gold[t],
            cls_preds[t],
            labels=label_ids,
            target_names=target_names,
            digits=4,
            zero_division=0
        )
        rep_path = os.path.join(OUT_DIR, "classification_reports", f"{t}.txt")
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(report)

        cm = confusion_matrix(cls_gold[t], cls_preds[t], labels=label_ids)
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        cm_path = os.path.join(OUT_DIR, "confusion_matrices", f"{t}.csv")
        cm_df.to_csv(cm_path)

    # ---- Pretty console summary (percentages + MB) ----
    print("✅ Evaluation complete")
    print("Saved metrics:", metrics_path)

    print(f"Train checkpoint size: {mb(metrics['train_checkpoint_size_mb'])}")
    print(f"FP16 deploy size:      {mb(metrics['fp16_deploy_size_mb'])}")
    print(f"Avg inference time:    {metrics['avg_inference_time_ms']:.2f} ms/sample")

    print(f"Intent  Acc/F1w: {pct(metrics['intent_acc'])} / {pct(metrics['intent_f1_weighted'])}")
    print(f"Target  Acc/F1w: {pct(metrics['target_type_acc'])} / {pct(metrics['target_type_f1_weighted'])}")
    print(f"Attr    Acc/F1w: {pct(metrics['attribute_acc'])} / {pct(metrics['attribute_f1_weighted'])}")
    print(f"Spatial Acc/F1w: {pct(metrics['spatial_relation_acc'])} / {pct(metrics['spatial_relation_f1_weighted'])}")
    print(f"Pos     Acc/F1w: {pct(metrics['position_acc'])} / {pct(metrics['position_f1_weighted'])}")

    print(f"Span EM:               {pct(metrics['spatial_reference_span_em'], digits=2)}")
    print(f"Exact (intent+target): {pct(metrics['exact_match_intent_target'], digits=2)}")
    print(f"Exact (all+span):      {pct(metrics['exact_match_all_fields_with_span'], digits=2)}")

if __name__ == "__main__":
    main()
