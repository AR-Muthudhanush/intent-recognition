import os, json, re
import pandas as pd
import unicodedata

RAW_PATH = "data/raw/synthetic_intent_ui_ko_10k.csv"
OUT_CSV  = "data/processed/data.csv"
MAP_JSON = "data/processed/label_maps.json"

# Classification tasks (spatial_reference will be span-extracted, not classified)
CLS_TASKS = ["intent", "target_type", "attribute", "spatial_relation", "position"]
NONE = "NONE"

# Korean-specific particle and Hangul utilities
KOREAN_PARTICLES = ["은", "는", "이", "가", "을", "를", "에", "에서", "로"]
HANGUL_START = 0xAC00
HANGUL_END = 0xD7A3

def is_hangul(char):
    """Check if character is Hangul"""
    return HANGUL_START <= ord(char) <= HANGUL_END

def remove_korean_particles(text: str) -> str:
    """Remove trailing Korean particles that are similar to English articles"""
    for particle in KOREAN_PARTICLES:
        if text.endswith(particle):
            text = text[:-len(particle)].strip()
            break
    return text

def normalize_hangul(text: str) -> str:
    """Normalize Hangul text using Unicode normalization"""
    text = unicodedata.normalize('NFC', text)
    return text

def norm_text(x: str) -> str:
    """Normalize Korean text"""
    x = str(x).strip().lower()
    x = normalize_hangul(x)
    x = re.sub(r"\s+", " ", x)
    return x

def norm_label(x: str) -> str:
    """Normalize Korean label"""
    x = norm_text(x)
    if x in {"", "nan", "none", "null", "없음", "무"}:
        return NONE
    return x

def norm_target_type(x: str) -> str:
    """Normalize Korean UI target types"""
    x = norm_label(x)
    if x == NONE:
        return x
    
    # Map Korean UI element terms
    repl = {
        "텍스트 상자": "textbox",
        "텍스트박스": "textbox",
        "입력란": "textbox",
        "입력 필드": "textbox",
        "체크박스": "checkbox",
        "확인란": "checkbox",
        "라디오 버튼": "radiobutton",
        "라디오버튼": "radiobutton",
        "선택 버튼": "radiobutton",
        "버튼": "button",
        "탭": "tab",
        "메뉴": "menu",
        "드롭다운": "dropdown",
        "목록": "list",
        "텍스트": "text",
        "링크": "link",
    }
    return repl.get(x, x)

def norm_attribute(x: str) -> str:
    """Normalize Korean attribute descriptors"""
    x = norm_label(x)
    if x == NONE:
        return x
    
    # Map Korean color and property terms
    attr_map = {
        "빨간색": "red",
        "빨강": "red",
        "파란색": "blue",
        "파랑": "blue",
        "초록색": "green",
        "초록": "green",
        "노란색": "yellow",
        "노랑": "yellow",
        "검은색": "black",
        "검정": "black",
        "흰색": "white",
        "하양": "white",
        "회색": "gray",
        "회색": "gray",
    }
    return attr_map.get(x, x)

def norm_spatial_relation(x: str) -> str:
    """Normalize Korean spatial relationship terms"""
    x = norm_label(x)
    if x == NONE:
        return x
    
    # Map Korean spatial prepositions
    rel_map = {
        "아래": "below",
        "아래쪽": "below",
        "밑": "below",
        "위": "above",
        "위쪽": "above",
        "상단": "above",
        "왼쪽": "left_of",
        "왼쪽의": "left_of",
        "좌측": "left_of",
        "오른쪽": "right_of",
        "오른쪽의": "right_of",
        "우측": "right_of",
        "옆": "next_to",
        "곁": "next_to",
        "근처": "next_to",
        "안": "inside",
        "안쪽": "inside",
        "내부": "inside",
        "왼쪽에서": "from_left",
        "왼쪽부터": "from_left",
        "오른쪽에서": "from_right",
        "오른쪽부터": "from_right",
    }
    return rel_map.get(x, x.replace(" ", "_"))

def norm_position(x: str) -> str:
    """Normalize Korean position terms"""
    x = norm_label(x)
    if x == NONE:
        return x
    
    # Map Korean ordinal numbers
    pos_map = {
        "첫": "1st",
        "첫번째": "1st",
        "첫 번째": "1st",
        "1번째": "1st",
        "2번": "2nd",
        "두": "2nd",
        "두번째": "2nd",
        "두 번째": "2nd",
        "3번": "3rd",
        "세": "3rd",
        "세번째": "3rd",
        "세 번째": "3rd",
        "4번": "4th",
        "네": "4th",
        "네번째": "4th",
        "네 번째": "4th",
        "5번": "5th",
        "다섯": "5th",
        "다섯번째": "5th",
        "다섯 번째": "5th",
        "마지막": "last",
        "끝": "last",
        "중간": "middle",
        "중앙": "middle",
        "위": "top",
        "맨위": "top",
        "상단": "top",
        "아래": "bottom",
        "맨아래": "bottom",
        "하단": "bottom",
    }
    return pos_map.get(x, x)

def norm_spatial_reference(x: str) -> str:
    """
    Keep as TEXT (span target) for Korean. Light normalization to reduce noise.
    Do NOT turn into label ids.
    """
    x = norm_label(x)
    if x == NONE:
        return x
    
    # Remove Korean articles/particles from beginning
    x = remove_korean_particles(x)
    
    # Korean-specific text replacements
    x = x.replace("쇼핑 카트", "cart")
    x = x.replace("카트", "cart")
    x = x.replace("카트 아이콘", "cart")
    x = x.replace("와이파이", "wi-fi")
    x = x.replace("와이 파이", "wi-fi")
    x = re.sub(r"\s+", " ", x).strip()
    
    return x

def main():
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(RAW_PATH, encoding='utf-8')

    needed = ["command", "intent", "target_type", "attribute", "spatial_relation", "spatial_reference", "position"]
    df = df[needed].copy()

    df["command"] = df["command"].astype(str).str.strip()
    df = df[df["command"].str.len() > 0].reset_index(drop=True)

    # normalize Korean text
    df["intent"] = df["intent"].apply(norm_label)
    df["target_type"] = df["target_type"].apply(norm_target_type)
    df["attribute"] = df["attribute"].apply(norm_attribute)
    df["spatial_relation"] = df["spatial_relation"].apply(norm_spatial_relation)
    df["position"] = df["position"].apply(norm_position)

    # spatial_reference as text span supervision
    df["spatial_reference"] = df["spatial_reference"].apply(norm_spatial_reference)

    # Build label maps ONLY for classification tasks
    label_maps = {}
    for t in CLS_TASKS:
        labels = sorted(df[t].unique().tolist())
        if NONE not in labels:
            labels = [NONE] + labels
        label2id = {lbl: i for i, lbl in enumerate(labels)}
        id2label = {i: lbl for lbl, i in label2id.items()}
        label_maps[t] = {"label2id": label2id, "id2label": id2label, "num_labels": len(labels)}

    # Convert to ids
    for t in CLS_TASKS:
        df[t + "_id"] = df[t].map(label_maps[t]["label2id"]).astype(int)

    df.to_csv(OUT_CSV, index=False, encoding='utf-8')
    with open(MAP_JSON, "w", encoding="utf-8") as f:
        json.dump(label_maps, f, ensure_ascii=False, indent=2)

    print("✅ Saved:", OUT_CSV)
    print("✅ Saved:", MAP_JSON)
    print("Rows:", len(df))
    for t in CLS_TASKS:
        print(f"{t}: {label_maps[t]['num_labels']} labels")
    print("spatial_reference: span-extraction (text), not classification")

if __name__ == "__main__":
    main()
