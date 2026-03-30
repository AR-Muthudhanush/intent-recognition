# Korean Intent Recognition - Complete Setup & Execution Guide

## Overview

This guide walks through the complete process of setting up, training, evaluating, and deploying the Korean Intent Recognition system.

## Directory Structure

```
intent_tinybert_ko_10mb/          # Korean project root
├── README.md                      # English documentation
├── README_KO.md                   # Korean documentation (한국어)
├── SETUP_GUIDE.md                # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── data/
│   ├── raw/
│   │   └── synthetic_intent_ui_ko_10k.csv   # Korean training data (CSV)
│   └── processed/                 # Generated after preprocessing
│       ├── data.csv               # Processed & normalized data
│       └── label_maps.json        # Label ID mappings (JSON)
├── src/
│   ├── model.py                  # TinyBertMultiTaskSpan model definition
│   ├── preprocess.py             # Korean preprocessing logic
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── export_fp16.py            # FP16 export for deployment
│   └── infer_json.py             # Inference script
└── outputs/                       # Generated during training
    ├── runs/run_001/             # Training run directory
    │   ├── best_model/           # Best model weights & config
    │   ├── checkpoints/          # Intermediate checkpoints
    │   ├── classification_reports/# Evaluation reports
    │   ├── confusion_matrices/   # Confusion matrices (CSV)
    │   ├── metrics.json          # Training metrics
    │   └── metrics_eval.json     # Evaluation metrics
    └── exports/
        └── fp16_span_model/      # FP16 deployment model
            ├── multitask_span_state_fp16.pt
            ├── base_model.txt
            ├── label_maps.json
            ├── tokenizer.json
            ├── vocab.txt
            └── special_tokens_map.json
```

## Step-by-Step Execution

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB+ RAM (CPU) or 2GB+ VRAM (GPU)
- ~5GB disk space for models and outputs

### Step 1: Environment Setup

```bash
# Navigate to project directory
cd intent_tinybert_ko_10mb

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

**Expected Installation Time**: 3-5 minutes

### Step 2: Data Preparation

Korean training data is provided at `data/raw/synthetic_intent_ui_ko_10k.csv`

**File Format**:
```csv
command,intent,target_type,attribute,spatial_relation,spatial_reference,position
마지막 메뉴 열기,open,menu,없음,없음,없음,마지막
텍스트 복사,copy,text,없음,없음,없음,없음
...
```

**Columns**:
- `command`: Korean UI automation command
- `intent`: Action to perform
- `target_type`: UI element type
- `attribute`: Visual property (color, size, etc.)
- `spatial_relation`: Position relationship
- `spatial_reference`: Reference object
- `position`: Sequential position

Use "없음" (NONE) for missing values.

### Step 3: Data Preprocessing

Converts raw Korean data to normalized training format.

```bash
python src/preprocess.py
```

**Processing Pipeline**:
1. Load CSV with UTF-8 encoding
2. Normalize Hangul (Korean script)
3. Remove Korean particles (은, 는, 이, 가, 을, 를, 에, 에서, 로)
4. Map Korean terms to English labels:
   - UI elements: "텍스트 상자" → textbox
   - Colors: "빨간색" → red
   - Spatial relations: "왼쪽" → left_of
   - Positions: "첫번째" → 1st
5. Create label ID mappings
6. Generate processed CSV and label maps JSON

**Output Files**:
- `data/processed/data.csv` - Normalized training data
- `data/processed/label_maps.json` - Label ID dictionaries

**Expected Time**: < 1 minute

**Expected Output**:
```
✅ Saved: data/processed/data.csv
✅ Saved: data/processed/label_maps.json
Rows: 10000
intent: 12 labels
target_type: 15 labels
attribute: 8 labels
spatial_relation: 8 labels
position: 8 labels
spatial_reference: span-extraction (text), not classification
```

### Step 4: Model Training

Trains TinyBertMultiTaskSpan model on Korean data.

```bash
python src/train.py
```

**Training Configuration**:
- Base Model: `kykim/bert-kor-base` (Korean BERT)
- Epochs: 3
- Batch Size: 32
- Learning Rate: 2e-5
- Max Sequence Length: 64 tokens
- 90/10 train/validation split
- Early stopping with patience=3
- FP16 enabled on CUDA

**Output Directory**: `outputs/runs/run_001/`

**Expected Time**: 
- GPU (CUDA): 20-30 minutes
- CPU: 2-4 hours

**Expected Output Structure**:
```
outputs/runs/run_001/
├── best_model/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── base_model.txt
│   ├── label_maps.json
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── special_tokens_map.json
├── checkpoints/
│   ├── checkpoint-250/
│   ├── checkpoint-500/
│   └── ...
└── trainer_state.json
```

### Step 5: Model Evaluation

Evaluates trained model on validation set.

```bash
python src/evaluate.py
```

**Evaluation Process**:
1. Loads best trained model
2. Runs inference on 10% validation data
3. Computes metrics for each task
4. Generates classification reports
5. Creates confusion matrices

**Output Files**:
- `outputs/runs/run_001/metrics_eval.json` - Metrics summary
- `outputs/runs/run_001/classification_reports/{task}.txt` - Per-task reports
- `outputs/runs/run_001/confusion_matrices/{task}.csv` - Confusion matrices

**Expected Time**: 2-5 minutes

**Expected Metrics**:
```json
{
  "classification_metrics": {
    "intent": {"accuracy": 0.92, "f1": 0.91},
    "target_type": {"accuracy": 0.88, "f1": 0.87},
    "attribute": {"accuracy": 0.85, "f1": 0.84},
    "spatial_relation": {"accuracy": 0.83, "f1": 0.82},
    "position": {"accuracy": 0.90, "f1": 0.89}
  },
  "span_metrics": {
    "start_accuracy": 0.81,
    "end_accuracy": 0.79
  }
}
```

### Step 6: Export for Deployment

Converts model to FP16 format for production deployment.

```bash
python src/export_fp16.py
```

**Conversion Process**:
1. Loads FP32 trained model (~20 MB)
2. Converts to FP16 (half precision)
3. Reduces size to ~10 MB (50% compression)
4. Exports all necessary files for inference

**Output Directory**: `outputs/exports/fp16_span_model/`

**Expected Time**: < 1 minute

**Output Files**:
```
outputs/exports/fp16_span_model/
├── multitask_span_state_fp16.pt  # FP16 model weights
├── base_model.txt                # Model name reference
├── label_maps.json               # Label mappings
├── tokenizer.json                # Tokenizer config
├── vocab.txt                     # Korean vocabulary
└── special_tokens_map.json       # Special tokens
```

### Step 7: Run Inference

Performs predictions on new Korean commands.

```bash
python src/infer_json.py
```

**Example Usage**:
```python
from src.infer_json import load, predict
import json

# Load model
tokenizer, model, device, id2label = load()

# Make prediction
command = "마지막 회색 탭 3번 클릭"  # "click the 3rd gray tab"
result = predict(command, tokenizer, model, device, id2label)

print(json.dumps(result, ensure_ascii=False, indent=2))
```

**Example Output**:
```json
{
  "intent": "click",
  "target_type": "tab",
  "attribute": "gray",
  "position": "3rd"
}
```

## Korean Language Adaptations

### Term Mappings

#### UI Elements (target_type)
```python
{
  "텍스트 상자": "textbox",
  "입력란": "textbox",
  "체크박스": "checkbox",
  "확인란": "checkbox",
  "라디오 버튼": "radiobutton",
  "버튼": "button",
  "탭": "tab",
  "메뉴": "menu",
  "드롭다운": "dropdown",
  "목록": "list",
  "텍스트": "text",
  "링크": "link",
}
```

#### Colors (attribute)
```python
{
  "빨간색/빨강": "red",
  "파란색/파랑": "blue",
  "초록색/초록": "green",
  "노란색/노랑": "yellow",
  "검은색/검정": "black",
  "흰색/하양": "white",
  "회색": "gray",
}
```

#### Spatial Relations
```python
{
  "아래/밑": "below",
  "위/상단": "above",
  "왼쪽/좌측": "left_of",
  "오른쪽/우측": "right_of",
  "옆/곁/근처": "next_to",
  "안/내부": "inside",
  "왼쪽에서/왼쪽부터": "from_left",
  "오른쪽에서/오른쪽부터": "from_right",
}
```

#### Positions
```python
{
  "첫/첫번째": "1st",
  "두/두번째": "2nd",
  "세/세번째": "3rd",
  "네/네번째": "4th",
  "다섯/다섯번째": "5th",
  "마지막/끝": "last",
  "중간/중앙": "middle",
  "상단": "top",
  "하단": "bottom",
}
```

### Preprocessing Features
- **Hangul Normalization**: Unicode NFC normalization for Hangul consistency
- **Particle Removal**: Automatically removes Korean particles (은, 는, 이, 가, 을, 를, 에, 에서, 로)
- **Text Normalization**: Lowercase conversion, whitespace normalization
- **UTF-8 Encoding**: All operations use UTF-8 for proper Korean character handling

## Configuration Guide

### Modifying Training Parameters

Edit `src/train.py` to change:

```python
# Model
MODEL_NAME = "kykim/bert-kor-base"  # Korean BERT
MAX_LEN = 64                        # Max token length

# Training
num_train_epochs=3
per_device_train_batch_size=32
learning_rate=2e-5
weight_decay=0.01

# Evaluation
eval_steps=250
save_steps=250
```

### Reducing GPU Memory Usage

```python
# In train.py, reduce batch size
per_device_train_batch_size=16  # Default is 32
per_device_eval_batch_size=16
```

### Adjusting Preprocessing

Edit `src/preprocess.py` to add/modify term mappings:

```python
# Add more Korean UI element mappings
repl = {
    "텍스트 상자": "textbox",
    # Add new mappings here
}
```

## Performance Metrics Interpretation

### Classification Metrics
- **Accuracy**: Percentage of correct predictions
- **F1 Score**: Harmonic mean of precision and recall (weighted average)
- **Macro F1**: Average F1 across all classes
- **Weighted F1**: F1 weighted by class frequency

### Span Extraction Metrics
- **Start Accuracy**: Correct prediction of span start position
- **End Accuracy**: Correct prediction of span end position

### Confusion Matrix
- Shows misclassifications per class
- Helps identify which classes are confused with each other

## Troubleshooting

### Issue: Korean Text Not Displaying
**Solution**: Ensure UTF-8 encoding in editor and terminal
```bash
# Check file encoding
file -i data/raw/synthetic_intent_ui_ko_10k.csv

# Should output: charset=utf-8
```

### Issue: Out of Memory
**Solution**: Reduce batch size
```python
per_device_train_batch_size=16
per_device_eval_batch_size=16
```

### Issue: CUDA Out of Memory
**Solution**: Use CPU or reduce model size
```bash
# Force CPU usage (modify train.py)
device = torch.device("cpu")

# Or set CUDA_VISIBLE_DEVICES to limit GPUs
export CUDA_VISIBLE_DEVICES=0
```

### Issue: Korean Character Encoding Errors
**Solution**: Verify UTF-8 encoding in data files
```python
# When reading CSV
df = pd.read_csv("file.csv", encoding='utf-8')
```

## Performance Optimization

### Training Speed
- **GPU**: ~20-30 minutes (3 epochs)
- **CPU**: ~2-4 hours (3 epochs)

### Model Size
- **FP32**: ~20 MB
- **FP16**: ~10 MB (50% smaller)

### Inference Speed
- **CPU**: ~10-20ms per sample
- **GPU**: ~2-5ms per sample

## Output File Descriptions

### label_maps.json
Maps label strings to IDs and vice versa for each classification task:
```json
{
  "intent": {
    "label2id": {"click": 0, "open": 1, "copy": 2, ...},
    "id2label": {"0": "click", "1": "open", "2": "copy", ...},
    "num_labels": 12
  }
}
```

### metrics_eval.json
Evaluation metrics for all tasks:
```json
{
  "classification_metrics": {...},
  "span_metrics": {...},
  "inference_time_seconds": 45.2,
  "samples_evaluated": 1000
}
```

### Classification Reports
Detailed per-class metrics:
```
              precision    recall  f1-score   support

click           0.91      0.94      0.92       850
open            0.88      0.85      0.86       120
copy            0.95      0.97      0.96       210
...
```

### Confusion Matrices
CSV files showing misclassifications between classes.

## Next Steps

1. **Deploy the model**: Use the FP16 exported model in production
2. **Create API**: Build REST API using FastAPI or Flask
3. **Batch inference**: Process multiple commands efficiently
4. **Fine-tuning**: Retrain on domain-specific data
5. **Extend labels**: Add new intent types or UI elements

## Support & Resources

- **Korean BERT**: https://huggingface.co/kykim/bert-kor-base
- **Transformers**: https://huggingface.co/docs/transformers/
- **PyTorch**: https://pytorch.org/docs/stable/index.html

## License

Follow project license guidelines.
