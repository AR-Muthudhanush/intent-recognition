# 한국어 의도 인식 시스템 (Korean Intent Recognition)

## 프로젝트 개요 (Project Overview)

이 프로젝트는 한국어 UI 자동화 명령어에서 구조화된 정보를 추출하는 **멀티태스크 의도 인식 시스템**입니다.

This is a **multitask intent recognition system** that extracts structured information from Korean UI automation commands using a lightweight Korean BERT model.

## 시스템 아키텍처 (System Architecture)

### 모델: `TinyBertMultiTaskSpan`
한국어 BERT 기반 컴팩트 모델로 다음을 수행합니다:
- **5개의 분류 헤드 (Classification Heads)**: intent, target_type, attribute, spatial_relation, position
- **1개의 스팬 추출 헤드 (Span Extraction Head)**: spatial_reference (명령문 내 토큰 범위 인식)

### 인식 항목 (6 Recognition Tasks)

1. **Intent (의도)**: 수행할 동작
   - Examples: click, open, copy, paste, disable, clear, fill, select, hover, double_click

2. **Target Type (대상 유형)**: 작용할 UI 요소
   - Examples: button, menu, text, tab, textbox, checkbox, radiobutton, dropdown, item, link, icon

3. **Attribute (속성)**: 시각적 특성/필터
   - Examples: red, blue, green, yellow, black, white, gray

4. **Spatial Relation (공간 관계)**: 위치 관계
   - Examples: above, below, left_of, right_of, next_to, inside, from_left, from_right

5. **Spatial Reference (공간 참조)**: 상대 위치 대상
   - Examples: button, menu, cart, icon, message

6. **Position (위치)**: 순차적 위치
   - Examples: 1st, 2nd, 3rd, 4th, 5th, last, top, bottom, middle

### 예시 (Example)

**한국어 명령어**: "마지막 회색 탭 3번 클릭"
- Intent: `click`
- Target Type: `tab`
- Attribute: `gray`
- Position: `3rd`

## 프로젝트 구조 (Project Structure)

```
intent_tinybert_ko_10mb/
├── requirements.txt              # Python 의존성
├── data/
│   ├── raw/
│   │   └── synthetic_intent_ui_ko_10k.csv  # 원본 한국어 데이터 (10,000개 샘플)
│   └── processed/
│       ├── data.csv              # 전처리된 데이터
│       └── label_maps.json       # 레이블 매핑 및 ID 변환
├── src/
│   ├── model.py                  # TinyBertMultiTaskSpan 모델 정의
│   ├── preprocess.py             # 한국어 텍스트 정규화 및 데이터 처리
│   ├── train.py                  # 모델 학습 스크립트
│   ├── evaluate.py               # 평가 및 메트릭 계산
│   ├── export_fp16.py            # FP16 모델 내보내기 (배포용)
│   └── infer_json.py             # 추론 스크립트
└── outputs/
    ├── runs/run_001/
    │   ├── best_model/           # 최고 성능 모델
    │   ├── checkpoints/          # 학습 중 체크포인트
    │   ├── classification_reports/  # 분류 평가 보고서
    │   └── confusion_matrices/      # 혼동 행렬
    └── exports/
        └── fp16_span_model/      # FP16 배포 모델
```

## 사용 방법 (Usage Guide)

### 1. 환경 설정 (Environment Setup)

```bash
cd intent_tinybert_ko_10mb
pip install -r requirements.txt
```

### 2. 데이터 전처리 (Data Preprocessing)

한국어 데이터를 정규화하고 레이블을 ID로 변환합니다:

```bash
python src/preprocess.py
```

**처리 내용**:
- 한글(Hangul) 유니코드 정규화
- 한국어 조사(particles) 제거: 은, 는, 이, 가, 을, 를, 에, 에서, 로
- 한국어 UI 요소 용어 매핑 (예: "텍스트 상자" → "textbox")
- 한국어 색상 용어 매핑 (예: "빨간색" → "red")
- 한국어 공간 관계 용어 매핑 (예: "왼쪽" → "left_of")
- 한국어 순서 용어 매핑 (예: "첫번째" → "1st")
- **출력**: `data/processed/data.csv`, `data/processed/label_maps.json`

### 3. 모델 학습 (Model Training)

```bash
python src/train.py
```

**학습 설정**:
- 기본 모델: `kykim/bert-kor-base` (한국어 특화 BERT)
- 에포크: 3
- 배치 크기: 32
- 학습률: 2e-5
- 최대 시퀀스 길이: 64토큰
- Early stopping: patience=3
- FP16 지원 (CUDA 가능시)
- **출력**: `outputs/runs/run_001/best_model/`

### 4. 모델 평가 (Model Evaluation)

```bash
python src/evaluate.py
```

**평가 항목**:
- 정확도(Accuracy), F1 스코어 (각 분류 태스크별)
- 혼동 행렬 (Confusion matrices)
- 분류 보고서 (Classification reports)
- 스팬 추출 정확도
- **출력**: `outputs/runs/run_001/metrics_eval.json`, 분류 보고서, 혼동 행렬

### 5. 모델 내보내기 (Export for Deployment)

FP16 형식으로 변환하여 배포용 모델 생성 (약 50% 크기 감소):

```bash
python src/export_fp16.py
```

- **출력**: `outputs/exports/fp16_span_model/`
- 모델 크기: ~10MB (FP16)
- 배포용 준비 완료

### 6. 추론 (Inference)

```bash
python src/infer_json.py
```

**출력 예시**:
```json
{
  "intent": "click",
  "target_type": "tab",
  "attribute": "gray",
  "position": "3rd",
  "spatial_reference": "tab"
}
```

## 한국어 전처리 상세 (Korean-Specific Preprocessing Details)

### 한글 정규화 (Hangul Normalization)
- 유니코드 NFC 정규화 적용
- 한글 조사 자동 제거

### 한국어 용어 매핑 (Korean Term Mapping)

#### UI 요소 (UI Target Types)
- "텍스트 상자", "입력란" → textbox
- "체크박스", "확인란" → checkbox
- "라디오 버튼" → radiobutton
- "드롭다운" → dropdown
- "탭" → tab
- "메뉴" → menu

#### 색상 (Attributes)
- "빨간색/빨강" → red
- "파란색/파랑" → blue
- "초록색/초록" → green
- "노란색/노랑" → yellow
- "검은색/검정" → black
- "흰색/하양" → white
- "회색" → gray

#### 공간 관계 (Spatial Relations)
- "아래/밑" → below
- "위/상단" → above
- "왼쪽/좌측" → left_of
- "오른쪽/우측" → right_of
- "옆/곁/근처" → next_to
- "안/내부" → inside
- "왼쪽에서/왼쪽부터" → from_left
- "오른쪽에서/오른쪽부터" → from_right

#### 순서 (Positions)
- "첫/첫번째" → 1st
- "두/두번째" → 2nd
- "세/세번째" → 3rd
- "네/네번째" → 4th
- "다섯/다섯번째" → 5th
- "마지막/끝" → last
- "중간/중앙" → middle
- "상단" → top
- "하단" → bottom

## 모델 사양 (Model Specifications)

| 항목 | 값 |
|------|-----|
| 기본 모델 | kykim/bert-kor-base |
| 숨겨진 크기 | 768 |
| 레이어 수 | 12 |
| 어휘 크기 | 30,000+ (한국어 토큰) |
| FP32 크기 | ~20 MB |
| FP16 크기 | ~10 MB |
| 최대 시퀀스 | 64 토큰 |

## 한국어 모델의 특별 고려사항 (Korean-Specific Considerations)

1. **토크나이저 (Tokenizer)**: kykim/bert-kor-base의 BERT 토크나이저 사용
   - 한글 자모(jamo) 단위 분해 가능
   - 한국어 특화 어휘 포함

2. **텍스트 정규화 (Text Normalization)**
   - 한글 조사 자동 제거로 변수성 감소
   - 유니코드 정규화로 인코딩 일관성 보장

3. **스팬 추출 (Span Extraction)**
   - 한국어 공백 기반 토크나이제이션 지원
   - 문자 오프셋을 토큰 오프셋으로 정확히 변환

4. **평가 (Evaluation)**
   - 한국어 학습 데이터 사용
   - 한국어 UI 용어의 변수 매핑 적용

## 영어 버전과의 차이점 (Differences from English Version)

| 항목 | 영어 | 한국어 |
|------|------|--------|
| 기본 모델 | prajjwal1/bert-tiny | kykim/bert-kor-base |
| 폴더 | intent_tinybert_10mb | intent_tinybert_ko_10mb |
| 데이터 파일 | synthetic_intent_ui_10k.csv | synthetic_intent_ui_ko_10k.csv |
| 정규화 | 영어 전처리 | 한글 정규화 + 조사 제거 |
| 용어 매핑 | 영어 용어 | 한국어 용어 매핑 |

## 성능 메트릭 (Performance Metrics)

평가 후 생성되는 메트릭:
- **각 태스크별 정확도 및 F1 스코어**
- **스팬 추출 정확도** (시작 토큰, 끝 토큰)
- **혼동 행렬** (각 분류 태스크)
- **상세 분류 보고서** (정밀율, 재현율, F1)

## 주의사항 (Important Notes)

1. **데이터 인코딩**: 모든 CSV 파일은 UTF-8 인코딩으로 저장됨
2. **한국어 텍스트**: 정규화를 통해 일관성 있는 입력 처리
3. **모델 크기**: FP16 내보내기로 배포 최적화
4. **GPU 가능성**: CUDA 가용 시 자동으로 GPU 사용

## 문제 해결 (Troubleshooting)

### 인코딩 오류
```python
# CSV 읽기/쓰기 시 UTF-8 명시
df = pd.read_csv("file.csv", encoding='utf-8')
df.to_csv("file.csv", encoding='utf-8')
```

### 한국어 토크나이제이션 문제
- kykim/bert-kor-base 토크나이저는 한글 자모 분해 지원
- 정규화된 입력 데이터 사용 권장

### GPU 메모리 부족
```bash
# 배치 크기 감소 (train.py에서)
per_device_train_batch_size=16  # 기본값 32에서 감소
```

## 라이선스 (License)

프로젝트 라이선스에 따름
