import os
import json
import torch
from safetensors.torch import load_file, save_file

from model import TinyBertMultiTaskSpan, CLS_TASKS
from transformers import AutoTokenizer

# -------------------------
# Config
# -------------------------
BEST_DIR = "outputs/runs/run_001/best_model"
FP16_DIR = "outputs/exports/fp16_span_model"

def get_folder_size_mb(path: str) -> float:
    total = 0
    if not os.path.exists(path):
        return 0.0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)

def export_fp16():
    """Export trained model to FP16 format for deployment"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    with open(f"{BEST_DIR}/base_model.txt", "r", encoding="utf-8") as f:
        base_model = f.read().strip()
    
    with open(f"{BEST_DIR}/label_maps.json", "r", encoding="utf-8") as f:
        maps = json.load(f)
    
    # Load tokenizer from base model instead of saved dir
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    num_labels = {t: maps[t]["num_labels"] for t in CLS_TASKS}
    
    model = TinyBertMultiTaskSpan(base_model, num_labels=num_labels)
    
    # Load FP32 weights from safetensors
    state = load_file(f"{BEST_DIR}/model.safetensors")
    model.load_state_dict(state, strict=False)
    
    print(f"Loaded model from {BEST_DIR}")
    
    # Convert to FP16
    model = model.half()
    print("Converted model to FP16")
    
    # Create export directory
    os.makedirs(FP16_DIR, exist_ok=True)
    
    # Save FP16 model state
    fp16_state = {}
    for k, v in model.state_dict().items():
        fp16_state[k] = v.half() if v.dtype == torch.float32 else v
    
    torch.save(fp16_state, f"{FP16_DIR}/multitask_span_state_fp16.pt")
    print(f"Saved FP16 model state to {FP16_DIR}/multitask_span_state_fp16.pt")
    
    # Save tokenizer
    tokenizer.save_pretrained(FP16_DIR)
    print(f"Saved tokenizer to {FP16_DIR}")
    
    # Save label maps and base model info
    with open(f"{FP16_DIR}/label_maps.json", "w", encoding="utf-8") as f:
        json.dump(maps, f, ensure_ascii=False, indent=2)
    
    with open(f"{FP16_DIR}/base_model.txt", "w", encoding="utf-8") as f:
        f.write(base_model)
    
    # Report
    fp32_size = get_folder_size_mb(BEST_DIR)
    fp16_size = get_folder_size_mb(FP16_DIR)
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"FP32 model size: {fp32_size:.2f} MB")
    print(f"FP16 model size: {fp16_size:.2f} MB")
    print(f"Compression: {(1 - fp16_size/fp32_size)*100:.1f}%")
    print(f"Export path: {FP16_DIR}")
    print("="*60)

if __name__ == "__main__":
    export_fp16()
