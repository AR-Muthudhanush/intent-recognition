import json
import torch
from transformers import AutoTokenizer
from model import TinyBertMultiTaskSpan, CLS_TASKS

MODEL_DIR = "outputs/exports/fp16_span_model"
MAX_LEN = 64
NONE = "NONE"

def drop_nulls(d):
    if isinstance(d, dict):
        cleaned = {}
        for k, v in d.items():
            vv = drop_nulls(v)
            if vv is None:
                continue
            if isinstance(vv, dict) and len(vv) == 0:
                continue
            cleaned[k] = vv
        return cleaned
    return d

def load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = open(f"{MODEL_DIR}/base_model.txt", "r", encoding="utf-8").read().strip()
    with open(f"{MODEL_DIR}/label_maps.json", "r", encoding="utf-8") as f:
        maps = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    num_labels = {t: maps[t]["num_labels"] for t in CLS_TASKS}

    model = TinyBertMultiTaskSpan(base_model, num_labels=num_labels)
    state = torch.load(f"{MODEL_DIR}/multitask_span_state_fp16.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    if device.type == "cuda":
        model = model.half()
    else:
        model = model.float()

    model.to(device)

    id2label = {t: {int(k): v for k, v in maps[t]["id2label"].items()} for t in CLS_TASKS}
    return tokenizer, model, device, id2label

@torch.no_grad()
def predict(command: str, tokenizer, model, device, id2label):
    enc = tokenizer(
        command,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_offsets_mapping=True
    )

    offsets = enc.pop("offset_mapping")[0].tolist()  # [L,2] on CPU
    inputs = {k: v.to(device) for k, v in enc.items()}

    cls_logits, start_logits, end_logits = model(**inputs)

    # classification labels
    pred = {}
    for t in CLS_TASKS:
        pid = int(torch.argmax(cls_logits[t], dim=1).item())
        pred[t] = id2label[t][pid]

    # span prediction
    s = int(torch.argmax(start_logits, dim=1).item())
    e = int(torch.argmax(end_logits, dim=1).item())
    if e < s:
        s, e = e, s

    # Convert token span -> char span using offsets
    # offsets can include special/pad tokens with (0,0)
    valid_offsets = [(i, a, b) for i, (a, b) in enumerate(offsets) if not (a == 0 and b == 0)]

    ref_text = None
    if 0 <= s < len(offsets) and 0 <= e < len(offsets):
        a_start, _ = offsets[s]
        _, b_end = offsets[e]
        # if offsets are zero (pad token), treat as None
        if not (a_start == 0 and b_end == 0):
            ref_text = command[a_start:b_end].strip()
            if ref_text == "":
                ref_text = None

    pred["spatial_reference_text"] = ref_text
    return pred

def to_output_json(pred):
    intent = pred["intent"]

    ttype = None if pred["target_type"] == NONE else pred["target_type"]
    attr  = None if pred["attribute"] == NONE else pred["attribute"]
    pos   = None if pred["position"] == NONE else pred["position"]
    rel   = None if pred["spatial_relation"] == NONE else pred["spatial_relation"]
    ref   = pred["spatial_reference_text"]  # span output (already text or None)

    out = {
        "intent": intent,
        "target": {
            "type": ttype,
            "attribute": attr,
            "position": pos,
            "spatial": {
                "relation": rel,
                "reference": ref
            }
        }
    }
    return out

if __name__ == "__main__":
    tokenizer, model, device, id2label = load()

    cmd = "copy orange text below the first email"
    pred = predict(cmd, tokenizer, model, device, id2label)

    out = drop_nulls(to_output_json(pred))
    print(json.dumps(out, indent=2))
