# Korean Intent Recognition System

## Project Overview

This is a **multitask intent recognition system** that extracts structured information from Korean UI automation commands using a lightweight Korean BERT model (`kykim/bert-kor-base`).

## System Architecture

### Model: `TinyBertMultiTaskSpan`
A compact Korean BERT-based model with:
- **5 classification heads**: intent, target_type, attribute, spatial_relation, position
- **1 span extraction head**: spatial_reference (identifies token spans in commands)

### 6 Recognition Tasks

1. **Intent**: Action to perform (click, open, copy, paste, etc.)
2. **Target Type**: UI element type (button, menu, text, tab, textbox, etc.)
3. **Attribute**: Visual properties (red, blue, green, yellow, etc.)
4. **Spatial Relation**: Position relationships (above, below, left_of, right_of, etc.)
5. **Spatial Reference**: Target reference object (menu, button, cart, etc.)
6. **Position**: Sequential position (1st, 2nd, last, top, bottom, etc.)

### Example

**Korean Command**: "마지막 회색 탭 3번 클릭"
- Intent: `click`
- Target Type: `tab`
- Attribute: `gray`
- Position: `3rd`

## Quick Start

### 1. Setup
```bash
cd intent_tinybert_ko_10mb
pip install -r requirements.txt
```

### 2. Preprocess Data
```bash
python src/preprocess.py
```
Normalizes Korean text, maps Korean terms, creates label IDs.

### 3. Train Model
```bash
python src/train.py
```
Trains on Korean data using kykim/bert-kor-base.

### 4. Evaluate
```bash
python src/evaluate.py
```
Generates metrics, confusion matrices, classification reports.

### 5. Export for Deployment
```bash
python src/export_fp16.py
```
Converts to FP16 format (~10MB, 50% smaller).

### 6. Run Inference
```bash
python src/infer_json.py
```

## Key Features

### Korean-Specific Preprocessing
- **Hangul normalization** using Unicode NFC
- **Korean particle removal** (은, 는, 이, 가, 을, 를, 에, 에서, 로)
- **Term mapping** for UI elements, colors, spatial relations, positions
- **UTF-8 encoding** throughout

### Korean Term Mapping Examples

**UI Elements**:
- "텍스트 상자" → textbox
- "체크박스" → checkbox
- "라디오 버튼" → radiobutton

**Colors**:
- "빨간색" → red
- "파란색" → blue
- "초록색" → green

**Spatial Relations**:
- "왼쪽" → left_of
- "오른쪽" → right_of
- "아래" → below
- "위" → above

**Positions**:
- "첫번째" → 1st
- "마지막" → last
- "중간" → middle

## Architecture Differences from English Version

| Aspect | English | Korean |
|--------|---------|--------|
| Base Model | prajjwal1/bert-tiny | kykim/bert-kor-base |
| Folder | intent_tinybert_10mb | intent_tinybert_ko_10mb |
| Data | synthetic_intent_ui_10k.csv | synthetic_intent_ui_ko_10k.csv |
| Preprocessing | English normalization | Hangul + particle removal |
| Term Mapping | English terms | Korean terms |

## Project Structure

```
intent_tinybert_ko_10mb/
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── synthetic_intent_ui_ko_10k.csv
│   └── processed/
│       ├── data.csv
│       └── label_maps.json
├── src/
│   ├── model.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_fp16.py
│   └── infer_json.py
├── outputs/
│   ├── runs/run_001/
│   │   ├── best_model/
│   │   ├── checkpoints/
│   │   ├── classification_reports/
│   │   └── confusion_matrices/
│   └── exports/
│       └── fp16_span_model/
└── README.md
```

## Model Specifications

| Item | Value |
|------|-------|
| Base Model | kykim/bert-kor-base |
| Hidden Size | 768 |
| Layers | 12 |
| Vocab Size | 30,000+ (Korean tokens) |
| FP32 Size | ~20 MB |
| FP16 Size | ~10 MB |
| Max Sequence | 64 tokens |

## Training Configuration

- **Learning Rate**: 2e-5
- **Batch Size**: 32
- **Epochs**: 3
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup
- **Early Stopping**: patience=3
- **Data Split**: 90% train, 10% validation
- **FP16 Support**: Enabled on CUDA

## Inference Example

```python
from infer_json import load, predict
import json

tokenizer, model, device, id2label = load()

command = "마지막 탭 클릭"
result = predict(command, tokenizer, model, device, id2label)

print(json.dumps(result, ensure_ascii=False, indent=2))
# Output:
# {
#   "intent": "click",
#   "target_type": "tab",
#   "position": "last"
# }
```

## Important Notes

1. **Encoding**: All CSV files must be UTF-8 encoded
2. **Tokenizer**: Uses kykim/bert-kor-base tokenizer (supports Hangul decomposition)
3. **Preprocessing**: All Korean terms normalized consistently
4. **GPU**: Automatically uses CUDA if available
5. **Model Size**: FP16 export optimized for deployment

## Troubleshooting

### Encoding Issues
```python
# Ensure UTF-8 encoding for all file operations
df = pd.read_csv("file.csv", encoding='utf-8')
df.to_csv("file.csv", encoding='utf-8')
```

### Korean Tokenization
- kykim/bert-kor-base supports Hangul decomposition
- Use normalized input data from preprocessing

### GPU Memory
```python
# Reduce batch size in train.py if needed
per_device_train_batch_size = 16
```

## Output Files

After training and evaluation:
- `outputs/runs/run_001/best_model/` - Best trained model
- `outputs/runs/run_001/metrics_eval.json` - Evaluation metrics
- `outputs/runs/run_001/classification_reports/` - Per-task reports
- `outputs/runs/run_001/confusion_matrices/` - Confusion matrices
- `outputs/exports/fp16_span_model/` - Deployment-ready FP16 model

## Language Support

This Korean version maintains the exact same architecture and logic as the English version, with language-specific adaptations:
- Korean term mappings in preprocessing
- Hangul normalization
- Korean BERT base model
- Thai UTF-8 encoding throughout

All model logic, training dynamics, and evaluation metrics remain unchanged for consistency.
