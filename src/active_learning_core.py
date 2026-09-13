"""主動學習排序:從未標註圖片池挑「標了最有價值」的圖優先給人工處理。

依既有的固定 5 類 schema(見 vlm_annotate_core.CLASS_NAME_TO_ID)判斷
Gloves / No_Gloves 這兩個目前的短板類別。
"""
import glob
import os

from annotate_core import calculate_iou
from vlm_annotate_core import CLASS_NAME_TO_ID

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

HAND_CLASSES = {"Gloves", "No_Gloves"}

# 給模型很低的信心門檻去撈候選框,「值不值得標」由分數評估自己判斷,
# 不要讓模型自己先用高門檻濾掉猶豫的框(那些猶豫的框正是主動學習要的)。
_PREDICT_CONF_FLOOR = 0.1


def list_unlabeled_images(folder):
    if not folder or not os.path.isdir(folder):
        return []
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(paths))


def score_image(model, image_path, w_uncertainty, w_class, w_density, w_overlap=0.0):
    """對單張圖評分,回傳 dict(score, box_count, uncertainty, class_weight, density, overlap)。

    class_weight 跟 density 兩項目前算的是同一個計數(手部框總數)——
    規格上是兩個獨立的可調權重,分別代表「這個類別本身很重要」跟
    「這張圖手部框密度高很划算」兩種不同的調權角度,即使底層計數相同,
    UI 上還是讓使用者能各自調整,不是重複的 bug。
    """
    results = model(image_path, conf=_PREDICT_CONF_FLOOR, verbose=False)[0]
    boxes = results.boxes
    names = model.names

    uncertainty = 0
    hand_count = 0
    parsed = []  # (cls_name, conf, x1, y1, x2, y2)
    for box in boxes:
        conf = float(box.conf[0])
        cls_name = names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        parsed.append((cls_name, conf, x1, y1, x2, y2))
        if 0.3 <= conf <= 0.6:
            uncertainty += 1
        if cls_name in HAND_CLASSES:
            hand_count += 1

    overlap_anomaly = 0
    if w_overlap:
        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                cls_i, _, *box_i = parsed[i]
                cls_j, _, *box_j = parsed[j]
                if cls_i == cls_j:
                    continue
                if calculate_iou(box_i, box_j) > 0.3:
                    overlap_anomaly += 1

    score = (w_uncertainty * uncertainty + w_class * hand_count +
             w_density * hand_count + w_overlap * overlap_anomaly)

    return {
        "path": image_path,
        "score": score,
        "box_count": len(parsed),
        "uncertainty": uncertainty,
        "class_weight": hand_count,
        "density": hand_count,
        "overlap": overlap_anomaly,
    }


def rank_images(folder, model, weights, progress=None):
    """對資料夾內所有未標註圖片評分並依分數由高到低排序。

    weights: dict(uncertainty, class_weight, density, overlap=0.0)
    回傳 (ranked, failed);failed 是 [(path, error_message), ...],絕不靜默跳過。
    """
    images = list_unlabeled_images(folder)
    ranked = []
    failed = []
    total = len(images)
    for i, path in enumerate(images):
        if progress is not None:
            progress((i + 1) / max(1, total), desc=f"評分中... ({i + 1}/{total})")
        try:
            ranked.append(score_image(
                model, path,
                weights.get("uncertainty", 0.0),
                weights.get("class_weight", 0.0),
                weights.get("density", 0.0),
                weights.get("overlap", 0.0),
            ))
        except Exception as e:
            failed.append((path, str(e)))

    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked, failed


def export_filelist(paths, out_path):
    """把檔名(不含路徑)一行一個寫成 txt,方便拿去 Roboflow 搜尋。"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(os.path.basename(p) for p in paths))
    return out_path
