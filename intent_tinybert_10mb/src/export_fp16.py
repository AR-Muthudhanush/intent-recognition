import os, json, torch
from transformers import AutoTokenizer
from model import TinyBertMultiTaskSpan, CLS_TASKS

BEST_DIR = "outputs/runs/run_001/best_model"
OUT_DIR  = "outputs/exports/fp16_span_model"

def folder_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total / (1024*1024)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    base_model = open(f"{BEST_DIR}/base_model.txt", "r", encoding="utf-8").read().strip()
    with open(f"{BEST_DIR}/label_maps.json", "r", encoding="utf-8") as f:
        maps = json.load(f)

    num_labels = {t: maps[t]["num_labels"] for t in CLS_TASKS}

    tokenizer = AutoTokenizer.from_pretrained(BEST_DIR, use_fast=True)

    model = TinyBertMultiTaskSpan(base_model, num_labels=num_labels)
    state = torch.load(f"{BEST_DIR}/multitask_span_state.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.half()

    tokenizer.save_pretrained(OUT_DIR)
    with open(f"{OUT_DIR}/base_model.txt", "w", encoding="utf-8") as f:
        f.write(base_model)

    with open(f"{OUT_DIR}/label_maps.json", "w", encoding="utf-8") as f:
        json.dump(maps, f, indent=2)

    torch.save(model.state_dict(), f"{OUT_DIR}/multitask_span_state_fp16.pt")

    print("✅ Exported FP16 span model to:", OUT_DIR)
    print(f"Folder size: {folder_size_mb(OUT_DIR):.2f} MB")

if __name__ == "__main__":
    main()
