import json
import re
import torch
from transformers import AutoTokenizer
from model import TinyBertMultiTaskSpan, CLS_TASKS

MODEL_DIR = "outputs/exports/fp16_span_model"
MAX_LEN = 64
NONE = "NONE"

def drop_nulls(d):
    """Remove None/empty values from result dictionary"""
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
    """Load FP16 model and tokenizer for inference"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = open(f"{MODEL_DIR}/base_model.txt", "r", encoding="utf-8").read().strip()
    with open(f"{MODEL_DIR}/label_maps.json", "r", encoding="utf-8") as f:
        maps = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=True)
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
    """Run inference on a Korean command"""
    
    enc = tokenizer(
        command,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )

    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(
        enc.get("token_type_ids", [0] * len(enc["input_ids"])),
        dtype=torch.long
    ).unsqueeze(0).to(device)

    cls_logits, start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

    # Classification predictions
    result = {}
    for t in CLS_TASKS:
        logits_t = cls_logits[t].squeeze(0)
        pred_id = int(logits_t.argmax(dim=-1).item())
        pred_label = id2label[t].get(pred_id, str(pred_id))

        if pred_label != NONE:
            result[t] = pred_label

    # Span extraction
    start_pred = int(start_logits.squeeze(0).argmax(dim=-1).item())
    end_pred = int(end_logits.squeeze(0).argmax(dim=-1).item())

    # Ensure valid span
    if end_pred < start_pred:
        end_pred = start_pred

    # Extract token span
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    span_tokens = tokens[start_pred:end_pred+1]
    
    # Decode tokens to text (Korean tokenizer may produce subword tokens)
    try:
        span_text = tokenizer.convert_tokens_to_string(span_tokens).replace(" ##", "").strip()
        if span_text and span_text not in ["[CLS]", "[SEP]", "[PAD]"]:
            result["spatial_reference"] = span_text
    except:
        pass

    return drop_nulls(result)

def main():
    """Example inference on Korean commands"""
    print("Loading model...")
    tokenizer, model, device, id2label = load()
    print("✅ Model loaded")

    # Example Korean commands
    test_commands = [
        "마지막 메뉴를 열기",  # "open the last menu"
        "회색 탭 3번 클릭",     # "click the 3rd gray tab"
        "텍스트 복사",          # "copy text"
        "왼쪽의 파란색 버튼 클릭",  # "click the blue button on the left"
    ]

    print("\n" + "="*70)
    print("KOREAN INTENT RECOGNITION INFERENCE")
    print("="*70)

    for cmd in test_commands:
        result = predict(cmd, tokenizer, model, device, id2label)
        print(f"\nCommand: {cmd}")
        print(f"Predicted: {json.dumps(result, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    main()
