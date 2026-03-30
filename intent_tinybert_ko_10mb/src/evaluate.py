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

# FP16 deployment export folder
FP16_DIR = "outputs/exports/fp16_span_model"

MODEL_NAME_FALLBACK = "kykim/bert-kor-base"
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
    # Remove Korean particles
    particles = ["은", "는", "이", "가", "을", "를", "에", "에서", "로"]
    for particle in particles:
        if x.endswith(particle):
            x = x[:-len(particle)].strip()
            break
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

def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.array(x)

# -------------------------
# Evaluation
# -------------------------
def evaluate():
    """Evaluate the trained model on validation set"""
    ensure_dirs()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    with open(MAP_JSON, "r", encoding="utf-8") as f:
        maps = json.load(f)
    
    # Use same split as training
    np.random.seed(42)
    _, val_df = train_test_split(df, test_size=0.1, random_state=42)
    val_df = val_df.reset_index(drop=True)
    
    # Load model and tokenizer
    if os.path.exists(BEST_DIR):
        try:
            model_name = open(f"{BEST_DIR}/base_model.txt", "r", encoding="utf-8").read().strip()
        except:
            model_name = MODEL_NAME_FALLBACK
    else:
        model_name = MODEL_NAME_FALLBACK
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    num_labels = {t: maps[t]["num_labels"] for t in CLS_TASKS}
    
    model = TinyBertMultiTaskSpan(model_name, num_labels=num_labels)
    
    # Load weights if available
    if os.path.exists(f"{BEST_DIR}/pytorch_model.bin"):
        state = torch.load(f"{BEST_DIR}/pytorch_model.bin", map_location=device)
        model.load_state_dict(state, strict=False)
    
    model.eval()
    model.to(device)
    
    # Prepare predictions
    all_preds = {t: [] for t in CLS_TASKS}
    all_labels = {t: [] for t in CLS_TASKS}
    all_start_preds = []
    all_end_preds = []
    all_start_labels = []
    all_end_labels = []
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        for idx in range(len(val_df)):
            row = val_df.iloc[idx]
            cmd = str(row["command"])
            
            enc = tokenizer(
                cmd,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
            )
            
            input_ids = torch.tensor(enc["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)
            token_type_ids = torch.tensor(enc.get("token_type_ids", [0] * len(enc["input_ids"])), dtype=torch.long).unsqueeze(0).to(device)
            
            cls_logits, start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
            
            # Classification predictions
            for t in CLS_TASKS:
                pred = int(np.argmax(_to_numpy(cls_logits[t]), axis=1)[0])
                label = int(row[t + "_id"])
                all_preds[t].append(pred)
                all_labels[t].append(label)
            
            # Span predictions
            start_pred = int(np.argmax(_to_numpy(start_logits), axis=1)[0])
            end_pred = int(np.argmax(_to_numpy(end_logits), axis=1)[0])
            all_start_preds.append(start_pred)
            all_end_preds.append(end_pred)
            
            # Span labels
            ref = str(row["spatial_reference"])
            span = find_char_span(cmd, ref)
            offsets = enc["offset_mapping"]
            
            start_label = 0
            end_label = 0
            if span is not None:
                char_start, char_end = span
                tok_span = charspan_to_tokens(offsets, char_start, char_end)
                if tok_span is not None:
                    start_label, end_label = tok_span
            
            all_start_labels.append(start_label)
            all_end_labels.append(end_label)
    
    elapsed = time.time() - start_time
    
    # Compute metrics
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    
    results = {}
    for t in CLS_TASKS:
        acc = accuracy_score(all_labels[t], all_preds[t])
        f1 = f1_score(all_labels[t], all_preds[t], average='weighted', zero_division=0)
        
        results[t] = {"accuracy": acc, "f1": f1}
        print(f"\n{t.upper()}")
        print(f"  Accuracy: {pct(acc)}")
        print(f"  F1 Score: {pct(f1)}")
        
        # Save classification report
        all_classes = sorted(set(all_labels[t] + all_preds[t]))
        report = classification_report(
            all_labels[t], all_preds[t],
            labels=all_classes,
            target_names=[maps[t]["id2label"][str(i)] for i in all_classes],
            zero_division=0
        )
        with open(f"{OUT_DIR}/classification_reports/{t}.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        # Save confusion matrix with consistent label sets
        cm = confusion_matrix(all_labels[t], all_preds[t], labels=all_classes)
        cm_df = pd.DataFrame(
            cm,
            index=[maps[t]["id2label"][str(i)] for i in all_classes],
            columns=[maps[t]["id2label"][str(i)] for i in all_classes]
        )
        cm_df.to_csv(f"{OUT_DIR}/confusion_matrices/{t}.csv")
    
    # Span extraction metrics
    print(f"\nSPAN EXTRACTION (spatial_reference)")
    start_acc = accuracy_score(all_start_labels, all_start_preds)
    end_acc = accuracy_score(all_end_labels, all_end_preds)
    print(f"  Start token accuracy: {pct(start_acc)}")
    print(f"  End token accuracy: {pct(end_acc)}")
    
    # Save metrics
    metrics = {
        "classification_metrics": results,
        "span_metrics": {
            "start_accuracy": float(start_acc),
            "end_accuracy": float(end_acc)
        },
        "inference_time_seconds": elapsed,
        "samples_evaluated": len(val_df),
    }
    
    with open(f"{OUT_DIR}/metrics_eval.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # Model size
    model_size = get_folder_size_mb(BEST_DIR)
    print("\n" + "="*60)
    print(f"Model size: {mb(model_size)}")
    print(f"Evaluation time: {elapsed:.2f}s")
    print(f"Samples: {len(val_df)}")
    print("="*60)

if __name__ == "__main__":
    evaluate()
