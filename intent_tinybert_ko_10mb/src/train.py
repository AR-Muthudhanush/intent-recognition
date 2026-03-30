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

# Korean BERT model
MODEL_NAME = "kykim/bert-kor-base"
MAX_LEN = 64
NONE = "NONE"

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
    return np.array(x)

def compute_metrics(p):
    """Compute multitask metrics"""
    # Handle dict format logits (from our custom prediction_step)
    if isinstance(p.predictions, dict):
        logits_dict = p.predictions
        labels_dict = p.label_ids
        
        metrics = {}
        for t in CLS_TASKS:
            logits_t = logits_dict[t]  # [batch, num_labels]
            labels_t = labels_dict[t]  # [batch]
            
            preds_t = np.argmax(logits_t, axis=1)
            labels_t_np = _to_numpy(labels_t)
            
            acc = accuracy_score(labels_t_np, preds_t)
            f1 = f1_score(labels_t_np, preds_t, average='weighted', zero_division=0)
            metrics[f"{t}_accuracy"] = acc
            metrics[f"{t}_f1"] = f1
        
        return metrics
    else:
        # Handle stacked tensor format (fallback)
        preds = p.predictions[0]
        labels_dict = p.label_ids
        
        metrics = {}
        for i, t in enumerate(CLS_TASKS):
            preds_t = np.argmax(preds[i], axis=1)
            labels_t = labels_dict[t]
            acc = accuracy_score(labels_t, preds_t)
            f1 = f1_score(labels_t, preds_t, average='weighted', zero_division=0)
            metrics[f"{t}_accuracy"] = acc
            metrics[f"{t}_f1"] = f1
        
        return metrics

class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        cmd = str(row["command"])
        
        enc = self.tokenizer(
            cmd,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        token_type_ids = enc.get("token_type_ids", [0] * len(input_ids))
        offsets = enc["offset_mapping"]
        
        # Classification labels
        cls_labels = {t: int(row[t + "_id"]) for t in CLS_TASKS}
        
        # Span labels
        ref = str(row["spatial_reference"])
        span = find_char_span(cmd, ref)
        
        start_label = 0
        end_label = 0
        if span is not None:
            char_start, char_end = span
            tok_span = charspan_to_tokens(offsets, char_start, char_end)
            if tok_span is not None:
                start_label, end_label = tok_span
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            **{f"{t}_label": torch.tensor(cls_labels[t], dtype=torch.long) for t in CLS_TASKS},
            "start_label": torch.tensor(start_label, dtype=torch.long),
            "end_label": torch.tensor(end_label, dtype=torch.long),
        }

def multitask_loss(logits_dict, start_logits, end_logits, labels_dict, start_label, end_label, device):
    """Compute combined multitask loss"""
    loss_fn = torch.nn.CrossEntropyLoss()
    
    total_loss = 0.0
    
    # Classification losses
    for t in CLS_TASKS:
        cls_loss = loss_fn(logits_dict[t], labels_dict[t].to(device))
        total_loss += cls_loss
    
    # Span extraction loss
    start_loss = loss_fn(start_logits, start_label.to(device))
    end_loss = loss_fn(end_logits, end_label.to(device))
    total_loss += start_loss + end_loss
    
    return total_loss

class MultiTaskDataCollator:
    """Custom collator for multitask learning that preserves label keys"""
    def __call__(self, batch):
        # Batch is a list of dicts from __getitem__
        batch_dict = {}
        
        for key in batch[0].keys():
            if key.endswith("_label") or key in ["start_label", "end_label"]:
                # Stack label tensors
                batch_dict[key] = torch.stack([item[key] for item in batch])
            elif key in ["input_ids", "attention_mask", "token_type_ids"]:
                # Stack token tensors
                batch_dict[key] = torch.stack([item[key] for item in batch])
        
        return batch_dict

class MultiTaskTrainer(Trainer):
    def get_train_dataloader(self):
        """Override to use custom data collator"""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=MultiTaskDataCollator(),
            shuffle=True,
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Override to use custom data collator for eval"""
        from torch.utils.data import DataLoader
        eval_dataset = eval_dataset or self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=MultiTaskDataCollator(),
        )
    
    def _prepare_inputs(self, inputs):
        """Override to preserve custom label keys"""
        # Don't use parent's _prepare_inputs which filters keys
        # Instead, just move all tensors to device
        device = next(self.model.parameters()).device
        for k, v in inputs.items():
            if hasattr(v, "to"):
                inputs[k] = v.to(device)
        return inputs
    
    def _save(self, output_dir=None, state_dict=None):
        """Override to make tensors contiguous before saving"""
        # Make sure all tensors are contiguous
        if state_dict is None:
            state_dict = self.model.state_dict()
        
        contiguous_state_dict = {}
        for k, v in state_dict.items():
            if hasattr(v, 'contiguous'):
                contiguous_state_dict[k] = v.contiguous()
            else:
                contiguous_state_dict[k] = v
        
        return super()._save(output_dir=output_dir, state_dict=contiguous_state_dict)
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to handle custom label keys"""
        # Separate label keys from input keys
        label_keys_dict = {}
        input_keys = ["input_ids", "attention_mask", "token_type_ids"]
        
        for key in list(inputs.keys()):
            if key not in input_keys:
                label_keys_dict[key] = inputs.pop(key)
        
        # Move input tensors to device
        device = next(model.parameters()).device
        for k in input_keys:
            if k in inputs and hasattr(inputs[k], "to"):
                inputs[k] = inputs[k].to(device)
        
        # Call model with only input keys
        with torch.no_grad():
            outputs = model(**inputs)
        
        loss = None
        if prediction_loss_only:
            labels_dict = {t: label_keys_dict[f"{t}_label"] for t in CLS_TASKS}
            start_label = label_keys_dict["start_label"]
            end_label = label_keys_dict["end_label"]
            loss = multitask_loss(outputs[0], outputs[1], outputs[2], labels_dict, start_label, end_label, device)
        
        # Prepare outputs for compute_metrics
        logits_dict, start_logits, end_logits = outputs
        
        # Return classification logits as dict (compute_metrics handles this)
        # For compatibility with compute_metrics, we return this special format
        label_ids = {t: label_keys_dict[f"{t}_label"] for t in CLS_TASKS}
        
        # Store logits_dict as a special attribute that compute_metrics can access
        # We'll return logits_dict directly as the logits
        return (loss, logits_dict, label_ids)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits_dict, start_logits, end_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
        )
        
        labels_dict = {t: inputs[f"{t}_label"] for t in CLS_TASKS}
        start_label = inputs["start_label"]
        end_label = inputs["end_label"]
        
        device = next(model.parameters()).device
        loss = multitask_loss(logits_dict, start_logits, end_logits, labels_dict, start_label, end_label, device)
        
        return (loss, (logits_dict, start_logits, end_logits)) if return_outputs else loss

def main():
    set_seed(42)
    os.makedirs("outputs/runs/run_001", exist_ok=True)
    
    # Load data
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    with open(MAP_JSON, "r", encoding="utf-8") as f:
        maps = json.load(f)
    
    # Train/val split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_labels = {t: maps[t]["num_labels"] for t in CLS_TASKS}
    model = TinyBertMultiTaskSpan(MODEL_NAME, num_labels=num_labels)
    
    # Datasets
    train_dataset = MultiTaskDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = MultiTaskDataset(val_df, tokenizer, MAX_LEN)
    
    # Training args
    training_args = TrainingArguments(
        output_dir="outputs/runs/run_001",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=250,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_intent_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=42,
    )
    
    # Trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=MultiTaskDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train
    trainer.train()
    
    # Save best model
    best_model_dir = "outputs/runs/run_001/best_model"
    os.makedirs(best_model_dir, exist_ok=True)
    trainer.save_model(best_model_dir)
    
    # Save label maps
    model_maps = {t: maps[t] for t in CLS_TASKS}
    with open(f"{best_model_dir}/label_maps.json", "w", encoding="utf-8") as f:
        json.dump(model_maps, f, ensure_ascii=False, indent=2)
    
    # Save model name
    with open(f"{best_model_dir}/base_model.txt", "w", encoding="utf-8") as f:
        f.write(MODEL_NAME)
    
    print("✅ Training complete!")
    print(f"✅ Best model saved to {best_model_dir}")

if __name__ == "__main__":
    main()
