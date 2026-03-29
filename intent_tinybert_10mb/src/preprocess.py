import os, json, re
import pandas as pd

RAW_PATH = "data/raw/synthetic_intent_ui_10k.csv"
OUT_CSV  = "data/processed/data.csv"
MAP_JSON = "data/processed/label_maps.json"

# Classification tasks (spatial_reference will be span-extracted, not classified)
CLS_TASKS = ["intent", "target_type", "attribute", "spatial_relation", "position"]
NONE = "NONE"

def norm_text(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x

def norm_label(x: str) -> str:
    x = norm_text(x)
    if x in {"", "nan", "none", "null"}:
        return NONE
    return x

def norm_target_type(x: str) -> str:
    x = norm_label(x)
    if x == NONE:
        return x
    repl = {
        "text box": "textbox",
        "text field": "textbox",
        "text-field": "textbox",
        "check box": "checkbox",
        "radio button": "radiobutton",
    }
    return repl.get(x, x)

def norm_attribute(x: str) -> str:
    x = norm_label(x)
    if x == NONE:
        return x
    x = x.replace("colour", "color")
    return x

def norm_spatial_relation(x: str) -> str:
    x = norm_label(x)
    if x == NONE:
        return x
    rel_map = {
        "below": "below",
        "under": "below",
        "above": "above",
        "over": "above",
        "left of": "left_of",
        "to the left of": "left_of",
        "right of": "right_of",
        "to the right of": "right_of",
        "next to": "next_to",
        "near": "next_to",
        "inside": "inside",
        "within": "inside",
        "from left": "from_left",
        "from the left": "from_left",
        "from right": "from_right",
        "from the right": "from_right",
    }
    return rel_map.get(x, x.replace(" ", "_"))

def norm_position(x: str) -> str:
    x = norm_label(x)
    if x == NONE:
        return x
    pos_map = {
        "first": "1st", "1st": "1st",
        "second": "2nd", "2nd": "2nd",
        "third": "3rd", "3rd": "3rd",
        "fourth": "4th", "4th": "4th",
        "fifth": "5th", "5th": "5th",
        "last": "last",
        "middle": "middle",
        "top": "top",
        "bottom": "bottom",
    }
    return pos_map.get(x, x)

def norm_spatial_reference(x: str) -> str:
    """
    Keep as TEXT (span target). Light normalization to reduce noise,
    but DO NOT turn into label ids.
    """
    x = norm_label(x)
    if x == NONE:
        return x
    x = re.sub(r"^(the|a|an)\s+", "", x)   # remove leading article
    x = x.replace("shopping cart", "cart")
    x = x.replace("cart icon", "cart")
    x = x.replace("wifi", "wi-fi")
    x = re.sub(r"\s+", " ", x).strip()
    return x

def main():
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    needed = ["command", "intent", "target_type", "attribute", "spatial_relation", "spatial_reference", "position"]
    df = df[needed].copy()

    df["command"] = df["command"].astype(str).str.strip()
    df = df[df["command"].str.len() > 0].reset_index(drop=True)

    # normalize
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

    df.to_csv(OUT_CSV, index=False)
    with open(MAP_JSON, "w", encoding="utf-8") as f:
        json.dump(label_maps, f, indent=2)

    print("✅ Saved:", OUT_CSV)
    print("✅ Saved:", MAP_JSON)
    print("Rows:", len(df))
    for t in CLS_TASKS:
        print(f"{t}: {label_maps[t]['num_labels']} labels")
    print("spatial_reference: span-extraction (text), not classification")

if __name__ == "__main__":
    main()
