import os, json, re
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)

from model import TinyBertMultiTaskSpan, CLS_TASKS

DATA_PATH = "data/processed/data.csv"
MAP_JSON  = "data/processed/label_maps.json"

MODEL_NAME = "prajjwal1/bert-tiny"
MAX_LEN = 64
NONE = "NONE"

def norm_ref_for_match(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"^(the|a|an)\s+", "", x)
    return x

def find_char_span(command: str, ref: str):
    """
    Find (start_char, end_char) of ref inside command (case-insensitive).
    Returns None if not found or ref is NONE.
    """
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

    # fallback: if end lands on boundary, allow last token whose e >= char_end
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
    # works for torch.Tensor or numpy arrays
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return x  # already numpy

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # preds/labels are dicts; values may be torch tensors or numpy arrays depending on HF version
    cls_preds = {t: _to_numpy(preds[t]) for t in CLS_TASKS}
    cls_gold  = {t: _to_numpy(labels[t]) for t in CLS_TASKS}

    metrics = {}
    for t in CLS_TASKS:
        metrics[f"{t}_acc"] = float(accuracy_score(cls_gold[t], cls_preds[t]))
        metrics[f"{t}_f1_weighted"] = float(f1_score(cls_gold[t], cls_preds[t], average="weighted", zero_division=0))

    # span EM (token-level exact match)
    start_pred = _to_numpy(preds["ref_start"])
    end_pred   = _to_numpy(preds["ref_end"])
    start_gold = _to_numpy(labels["ref_start"])
    end_gold   = _to_numpy(labels["ref_end"])

    mask = start_gold != -100
    if mask.sum() == 0:
        metrics["spatial_reference_span_em"] = 0.0
    else:
        em = np.mean((start_pred[mask] == start_gold[mask]) & (end_pred[mask] == end_gold[mask]))
        metrics["spatial_reference_span_em"] = float(em)

    # strict exact match across cls + span
    all_ok = np.ones(len(start_gold), dtype=bool)
    for t in CLS_TASKS:
        all_ok &= (cls_preds[t] == cls_gold[t])

    span_ok = ((start_gold == -100) & (end_gold == -100)) | ((start_pred == start_gold) & (end_pred == end_gold))
    all_ok &= span_ok
    metrics["exact_match_all_fields_with_span"] = float(np.mean(all_ok))

    metrics["exact_match_intent_target"] = float(np.mean(
        (cls_preds["intent"] == cls_gold["intent"]) &
        (cls_preds["target_type"] == cls_gold["target_type"])
    ))

    return metrics


class MultiTaskSpanTrainer(Trainer):
    def __init__(self, *args, task_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_weights = task_weights or {t: 1.0 for t in CLS_TASKS}
        self.task_weights["spatial_reference_span"] = self.task_weights.get("spatial_reference_span", 2.0)

        self.ce = torch.nn.CrossEntropyLoss()
        self.span_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)

        # classification labels
        cls_labels = {t: inputs.pop(t) for t in CLS_TASKS}

        # span labels
        ref_start = inputs.pop("ref_start")
        ref_end   = inputs.pop("ref_end")

        cls_logits, start_logits, end_logits = model(**inputs)

        loss = 0.0
        for t in CLS_TASKS:
            loss = loss + self.task_weights[t] * self.ce(cls_logits[t], cls_labels[t])

        span_loss = self.span_ce(start_logits, ref_start) + self.span_ce(end_logits, ref_end)
        loss = loss + self.task_weights["spatial_reference_span"] * span_loss

        outputs = {"cls_logits": cls_logits, "start_logits": start_logits, "end_logits": end_logits}
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # return torch tensors (accelerate-safe)
        cls_labels = {t: inputs[t].detach() for t in CLS_TASKS}
        labels = dict(cls_labels)
        labels["ref_start"] = inputs["ref_start"].detach()
        labels["ref_end"] = inputs["ref_end"].detach()

        with torch.no_grad():
            cls_logits, start_logits, end_logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        preds = {t: torch.argmax(cls_logits[t], dim=1).detach() for t in CLS_TASKS}
        preds["ref_start"] = torch.argmax(start_logits, dim=1).detach()
        preds["ref_end"]   = torch.argmax(end_logits, dim=1).detach()

        return (None, preds, labels)

def main():
    set_seed(42)
    os.makedirs("outputs/runs/run_001", exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    with open(MAP_JSON, "r", encoding="utf-8") as f:
        maps = json.load(f)

    num_labels = {t: maps[t]["num_labels"] for t in CLS_TASKS}

    # split stratified by intent
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["intent_id"])
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["intent_id"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def build_ds(dframe: pd.DataFrame):
        base_cols = ["command", "spatial_reference"] + [t + "_id" for t in CLS_TASKS]
        ds = Dataset.from_pandas(dframe[base_cols].copy())

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
                    # if not found, ignore this span in loss
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

            enc.pop("offset_mapping")  # not needed after converting to labels
            enc["ref_start"] = ref_starts
            enc["ref_end"] = ref_ends
            return enc

        ds = ds.map(tok_and_span, batched=True)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"] + CLS_TASKS + ["ref_start", "ref_end"]
        )
        return ds

    train_ds = build_ds(train_df)
    val_ds   = build_ds(val_df)
    test_ds  = build_ds(test_df)

    model = TinyBertMultiTaskSpan(MODEL_NAME, num_labels=num_labels)

    args = TrainingArguments(
        output_dir="outputs/runs/run_001/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="exact_match_intent_target",
        greater_is_better=True,

        learning_rate=2e-5,
        warmup_ratio=0.1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=8,
        weight_decay=0.01,

        report_to="none",
        logging_steps=50,
        dataloader_pin_memory=False,

        remove_unused_columns=False,
    )

    task_weights = {
        "intent": 1.0,
        "target_type": 1.0,
        "attribute": 1.1,
        "spatial_relation": 1.2,
        "position": 1.2,
        "spatial_reference_span": 2.0,
    }

    trainer = MultiTaskSpanTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        task_weights=task_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # test metrics
    test_metrics = trainer.evaluate(test_ds)
    with open("outputs/runs/run_001/metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # save
    best_dir = "outputs/runs/run_001/best_model"
    os.makedirs(best_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{best_dir}/multitask_span_state.pt")
    tokenizer.save_pretrained(best_dir)

    with open(f"{best_dir}/label_maps.json", "w", encoding="utf-8") as f:
        json.dump(maps, f, indent=2)

    with open(f"{best_dir}/base_model.txt", "w", encoding="utf-8") as f:
        f.write(MODEL_NAME)

    print("✅ Saved best model to:", best_dir)
    print("✅ Test metrics:", test_metrics)

if __name__ == "__main__":
    main()
