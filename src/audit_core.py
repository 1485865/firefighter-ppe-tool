"""標註品質稽核:找漏標、異常框、類別分布統計、同類別抽樣比對。

支援兩種既有目錄慣例:
  1. 標準 ultralytics 版型:<folder>/images/**/*.jpg 對 <folder>/labels/**/*.txt
     (tab_distill.py 訓練資料集就是這樣長)
  2. 本專案常見的「同資料夾同檔名」版型:<folder>/*.jpg + 同名 *.txt
     (tab_extractor.py / annotate_core.py 都這樣寫)
找不到 images/+labels/ 子資料夾就自動退回版型 2。
"""
import glob
import os
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlm_annotate_core import CLASS_NAME_TO_ID

ID_TO_CLASS = {v: k for k, v in CLASS_NAME_TO_ID.items()}
HAND_IDS = {CLASS_NAME_TO_ID["Gloves"], CLASS_NAME_TO_ID["No_Gloves"]}
HEAD_IDS = {CLASS_NAME_TO_ID["No_Helmet"], CLASS_NAME_TO_ID["helmet"]}
PERSON_ID = CLASS_NAME_TO_ID["person"]
HAND_CLASS_IDS = {CLASS_NAME_TO_ID["Gloves"], CLASS_NAME_TO_ID["No_Gloves"]}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_pairs(folder):
    """回傳 [(image_path, label_path), ...]。"""
    pairs = []
    images_dir = os.path.join(folder, "images")
    labels_dir = os.path.join(folder, "labels")
    if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
        for ext in IMG_EXTS:
            for img_path in glob.glob(os.path.join(images_dir, "**", f"*{ext}"), recursive=True):
                rel = os.path.relpath(img_path, images_dir)
                label_path = os.path.join(labels_dir, os.path.splitext(rel)[0] + ".txt")
                if os.path.exists(label_path):
                    pairs.append((img_path, label_path))
    else:
        for ext in IMG_EXTS:
            for img_path in glob.glob(os.path.join(folder, f"*{ext}")):
                label_path = os.path.splitext(img_path)[0] + ".txt"
                if os.path.exists(label_path):
                    pairs.append((img_path, label_path))
    return sorted(set(pairs))


def parse_label(label_path):
    """回傳 (boxes, parse_errors)。boxes: [(cls_id, cx, cy, w, h)](正規化座標)。"""
    boxes = []
    errors = []
    with open(label_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"第 {lineno} 行欄位數不對: {line!r}")
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, bw, bh = (float(x) for x in parts[1:])
            except ValueError:
                errors.append(f"第 {lineno} 行數值格式錯誤: {line!r}")
                continue
            boxes.append((cls_id, cx, cy, bw, bh))
    return boxes, errors


def _calc_iou_norm(b1, b2):
    """輸入正規化 (cx,cy,w,h),回傳 IoU。"""
    def to_xyxy(b):
        cx, cy, w, h = b
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    x1a, y1a, x2a, y2a = to_xyxy(b1)
    x1b, y1b, x2b, y2b = to_xyxy(b2)
    xi1, yi1 = max(x1a, x1b), max(y1a, y1b)
    xi2, yi2 = min(x2a, x2b), min(y2a, y2b)
    if xi2 < xi1 or yi2 < yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    return inter / max(1e-9, area_a + area_b - inter)


def audit_dataset(folder, area_max_pct=60.0, area_min_pct=0.1,
                   aspect_ratio_max=2.5, progress=None):
    """執行完整稽核,回傳 dict:
      missing_labels: [(image, person, hands, heads, gap), ...] 依缺口排序
      anomaly_boxes: [(image, line_no, reason), ...]
      class_counts: {class_name: count}
      class_areas: {class_name: [area_pct, ...]}
      per_image_counts: [instance_count, ...]
      failed: [(path, error_message), ...]
    """
    pairs = find_pairs(folder)
    missing_labels = []
    anomaly_boxes = []
    class_counts = {}
    class_areas = {}
    per_image_counts = []
    failed = []
    total = len(pairs)

    for i, (img_path, label_path) in enumerate(pairs):
        if progress is not None:
            progress((i + 1) / max(1, total), desc=f"稽核中... ({i + 1}/{total})")
        try:
            boxes, parse_errors = parse_label(label_path)
            for err in parse_errors:
                anomaly_boxes.append((img_path, "-", f"標註格式錯誤: {err}"))

            person_count = sum(1 for b in boxes if b[0] == PERSON_ID)
            hand_count = sum(1 for b in boxes if b[0] in HAND_IDS)
            head_count = sum(1 for b in boxes if b[0] in HEAD_IDS)
            expected_hands = person_count * 2
            expected_heads = person_count
            hand_gap = expected_hands - hand_count
            head_gap = expected_heads - head_count
            if (expected_hands > 0 and hand_count < expected_hands * 0.5) or \
               (expected_heads > 0 and head_count < expected_heads * 0.5):
                missing_labels.append({
                    "image": img_path, "person": person_count,
                    "hands": hand_count, "expected_hands": expected_hands,
                    "heads": head_count, "expected_heads": expected_heads,
                    "gap": max(hand_gap, 0) + max(head_gap, 0),
                })

            per_image_counts.append(len(boxes))

            for idx, (cls_id, cx, cy, bw, bh) in enumerate(boxes, 1):
                cls_name = ID_TO_CLASS.get(cls_id, f"未知類別({cls_id})")
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                area_pct = bw * bh * 100.0
                class_areas.setdefault(cls_name, []).append(area_pct)

                if area_pct > area_max_pct:
                    anomaly_boxes.append((img_path, idx, f"框過大({area_pct:.1f}% 畫面)"))
                elif area_pct < area_min_pct:
                    anomaly_boxes.append((img_path, idx, f"框過小({area_pct:.2f}% 畫面)"))

                if cls_id in HAND_CLASS_IDS and bh > 0:
                    ar = bw / bh
                    if ar > aspect_ratio_max or ar < (1.0 / aspect_ratio_max):
                        anomaly_boxes.append((img_path, idx, f"手套框長寬比異常(ratio={ar:.2f})"))

                if cx - bw / 2 < 0 or cy - bh / 2 < 0 or cx + bw / 2 > 1 or cy + bh / 2 > 1:
                    anomaly_boxes.append((img_path, idx, "框超出畫面邊界"))

            for a in range(len(boxes)):
                for b in range(a + 1, len(boxes)):
                    if boxes[a][0] != boxes[b][0]:
                        continue
                    if _calc_iou_norm(boxes[a][1:], boxes[b][1:]) > 0.95:
                        anomaly_boxes.append(
                            (img_path, f"{a+1}&{b+1}", "完全重疊的重複框"))
        except Exception as e:
            failed.append((img_path, str(e)))

    missing_labels.sort(key=lambda r: r["gap"], reverse=True)

    return {
        "missing_labels": missing_labels,
        "anomaly_boxes": anomaly_boxes,
        "class_counts": class_counts,
        "class_areas": class_areas,
        "per_image_counts": per_image_counts,
        "failed": failed,
        "total_images": len(pairs),
    }


def build_stats_figures(class_counts, class_areas):
    """回傳 (fig_counts, fig_areas)。"""
    fig_counts, ax1 = plt.subplots(figsize=(7, 4))
    if class_counts:
        labels, counts = list(class_counts.keys()), list(class_counts.values())
        ax1.bar(labels, counts, color="royalblue", edgecolor="black")
        ax1.set_title("各類別實例數")
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        fig_counts.tight_layout()
    else:
        ax1.text(0.5, 0.5, "沒有資料", ha="center", va="center")
        ax1.axis("off")

    fig_areas, ax2 = plt.subplots(figsize=(7, 4))
    if class_areas:
        labels = list(class_areas.keys())
        data = [class_areas[k] for k in labels]
        ax2.boxplot(data, labels=labels)
        ax2.set_title("各類別框面積佔比分布(%)")
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        fig_areas.tight_layout()
    else:
        ax2.text(0.5, 0.5, "沒有資料", ha="center", va="center")
        ax2.axis("off")

    return fig_counts, fig_areas


def sample_class_crops(folder, cls_name, n):
    """隨機抽 n 個指定類別的框,裁切出來給人目視比對標註標準是否一致。
    回傳 [(crop_bgr, caption), ...]。"""
    if cls_name not in CLASS_NAME_TO_ID:
        return []
    cls_id = CLASS_NAME_TO_ID[cls_name]
    pairs = find_pairs(folder)
    candidates = []  # (image_path, box)
    for img_path, label_path in pairs:
        boxes, _errors = parse_label(label_path)
        for box in boxes:
            if box[0] == cls_id:
                candidates.append((img_path, box))

    if not candidates:
        return []
    sample = random.sample(candidates, min(n, len(candidates)))

    crops = []
    for img_path, (cls_id, cx, cy, bw, bh) in sample:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        crops.append((crop, os.path.basename(img_path)))
    return crops


def export_filelist(paths, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(os.path.basename(p) for p in paths))
    return out_path
