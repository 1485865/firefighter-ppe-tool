"""跨模型共識標註(Ensemble Consensus):多來源(本機模型 + VLM)對同一張圖
各跑一次推論,用 IoU 空間聚類做多數決,過濾單一模型的誤判。

沿用既有的 _HARD 人工複核慣例(annotate_core.py / vlm_annotate_core.py):
_HARD 只加在 XML <name>,YOLO .txt 的 class id 永遠用正常類別 id
(CLASS_NAME_TO_ID,全專案固定 schema,不用各模型自己的 .names)。
"""
import glob
import os

import cv2

from annotate_core import calculate_iou
from vlm_annotate_core import CLASS_NAME_TO_ID, call_vlm_api

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

VOTE_STRICT = "標準模式(推薦)"
VOTE_LENIENT = "寬鬆模式(≥1 個來源同意即採用)"
VOTE_RULES = [VOTE_STRICT, VOTE_LENIENT]

MERGE_HIGHEST_CONF = "取信心最高者(推薦)"
MERGE_AVERAGE = "取平均"
MERGE_UNION = "取聯集框"
MERGE_MODES = [MERGE_HIGHEST_CONF, MERGE_AVERAGE, MERGE_UNION]


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def predict_local_model(model, image_path, conf=0.25):
    """本機 YOLO 模型推論,回傳 [(cls_name, conf, x1, y1, x2, y2), ...](像素座標)。"""
    res = model(image_path, conf=conf, verbose=False)[0]
    names = model.names
    out = []
    for box in res.boxes:
        cls_name = names[int(box.cls[0])]
        c = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append((cls_name, c, x1, y1, x2, y2))
    return out, res.orig_img.shape[1], res.orig_img.shape[0]  # boxes, w, h


def predict_vlm(image_path, api_key, w, h):
    """呼叫 VLM API,回傳跟本機模型同格式的 [(cls_name, conf, x1, y1, x2, y2), ...]。

    座標轉換邏輯比照 vlm_annotate_core.detections_to_yolo_and_xml(clamp 到
    [0,1]、x1>x2 互換、太小的框丟棄),但不寫檔——共識合併完才決定最終標籤。
    """
    data, _usage = call_vlm_api(image_path, api_key)
    out = []
    for det in data.get("detections", []):
        cls_name = det.get("class")
        bbox = det.get("bbox")
        if cls_name not in CLASS_NAME_TO_ID:
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
            conf = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        x1, x2 = min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)
        y1, y2 = min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)
        if (x2 - x1) < 0.001 or (y2 - y1) < 0.001:
            continue
        out.append((cls_name, conf, x1 * w, y1 * h, x2 * w, y2 * h))
    return out


def _merge_box(boxes, mode):
    """boxes: [(cls_name, conf, x1, y1, x2, y2), ...],回傳 (cls_name, x1, y1, x2, y2)。

    類別名一律取信心最高框的類別(座標合併方式只影響座標,不影響類別判定)。
    """
    best = max(boxes, key=lambda b: b[1])
    cls_name = best[0]
    if mode == MERGE_AVERAGE:
        n = len(boxes)
        x1 = sum(b[2] for b in boxes) / n
        y1 = sum(b[3] for b in boxes) / n
        x2 = sum(b[4] for b in boxes) / n
        y2 = sum(b[5] for b in boxes) / n
    elif mode == MERGE_UNION:
        x1 = min(b[2] for b in boxes)
        y1 = min(b[3] for b in boxes)
        x2 = max(b[4] for b in boxes)
        y2 = max(b[5] for b in boxes)
    else:  # MERGE_HIGHEST_CONF
        x1, y1, x2, y2 = best[2], best[3], best[4], best[5]
    return cls_name, x1, y1, x2, y2


def run_consensus_on_image(image_path, source_boxes, w, h, iou_threshold,
                            vote_rule, merge_mode):
    """source_boxes: dict(source_name -> [(cls_name, conf, x1,y1,x2,y2), ...])。

    回傳 dict(xml, txt_lines, stats),stats 含各票型計數與各來源獨有偵測數。
    """
    enabled_sources = list(source_boxes.keys())
    n_enabled = len(enabled_sources)

    flat = []  # (source, cls_name, conf, x1, y1, x2, y2)
    for src, boxes in source_boxes.items():
        for b in boxes:
            flat.append((src,) + tuple(b))

    n = len(flat)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            box_i, box_j = flat[i][3:7], flat[j][3:7]
            if calculate_iou(box_i, box_j) > iou_threshold:
                uf.union(i, j)

    groups = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(flat[i])

    stats = {
        "unanimous": 0, "majority": 0, "single": 0, "conflict": 0,
        "unique_by_source": {s: 0 for s in enabled_sources},
    }
    final_boxes = []  # (display_name, x1, y1, x2, y2)

    for members in groups.values():
        sources_present = set(m[0] for m in members)
        classes_present = set(m[1] for m in members)
        boxes_for_merge = [(m[1], m[2], m[3], m[4], m[5], m[6]) for m in members]
        cls_name, x1, y1, x2, y2 = _merge_box(boxes_for_merge, merge_mode)

        n_sources = len(sources_present)
        is_conflict = len(classes_present) > 1

        if vote_rule == VOTE_LENIENT:
            display_name = cls_name
            if is_conflict:
                stats["conflict"] += 1
            elif n_sources == n_enabled:
                stats["unanimous"] += 1
            elif n_sources > 1:
                stats["majority"] += 1
            else:
                stats["single"] += 1
        else:
            if is_conflict:
                display_name = f"{cls_name}_HARD"
                stats["conflict"] += 1
            elif n_sources == n_enabled:
                display_name = cls_name
                stats["unanimous"] += 1
            elif n_sources > 1:
                display_name = f"{cls_name}_HARD"
                stats["majority"] += 1
            else:
                display_name = f"{cls_name}_HARD"
                stats["single"] += 1

        if n_sources == 1:
            only_src = next(iter(sources_present))
            stats["unique_by_source"][only_src] = stats["unique_by_source"].get(only_src, 0) + 1

        final_boxes.append((display_name, cls_name, x1, y1, x2, y2))

    filename = os.path.basename(image_path)
    xml = (f"<annotation><filename>{filename}</filename>"
           f"<size><width>{w}</width><height>{h}</height></size>")
    txt_lines = []
    for display_name, base_cls, x1, y1, x2, y2 in final_boxes:
        cls_id = CLASS_NAME_TO_ID.get(base_cls)
        if cls_id is not None:
            cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            txt_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        xml += (f"<object><name>{display_name}</name><bndbox>"
                f"<xmin>{int(x1)}</xmin><ymin>{int(y1)}</ymin>"
                f"<xmax>{int(x2)}</xmax><ymax>{int(y2)}</ymax></bndbox></object>")
    xml += "</annotation>"

    return {
        "xml": xml, "txt_lines": txt_lines, "stats": stats,
        "draw_boxes": [(d, x1, y1, x2, y2) for d, _, x1, y1, x2, y2 in final_boxes],
    }


def draw_consensus_overlay(img_bgr, draw_boxes):
    """_HARD 框畫紅色(需要複核),正常框畫綠色。回傳 BGR 圖。"""
    out = img_bgr.copy()
    for display_name, x1, y1, x2, y2 in draw_boxes:
        color = (0, 0, 255) if display_name.endswith("_HARD") else (0, 200, 0)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(out, display_name, (int(x1), max(14, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def run_consensus_batch(folder, models, api_key, iou_threshold, vote_rule,
                         merge_mode, out_dir, progress=None):
    """models: dict(source_name -> YOLO model 物件),不含 VLM;
    VLM 是否參與由 api_key 是否為 None 決定(呼叫端已檢查 OPENAI_API_KEY)。

    輸出:對每張圖寫 out_dir/<name>.jpg(原圖複製)+ .txt + .xml,
    另存一張畫框預覽 out_dir/<name>_preview.jpg。
    回傳 (summary_stats, preview_items, failed)。
    """
    images = []
    for ext in IMG_EXTS:
        images.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        images.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    images = sorted(set(images))

    os.makedirs(out_dir, exist_ok=True)

    summary = {"unanimous": 0, "majority": 0, "single": 0, "conflict": 0,
               "unique_by_source": {}}
    preview_items = []
    failed = []
    total = len(images)

    for i, image_path in enumerate(images):
        if progress is not None:
            progress((i + 1) / max(1, total),
                     desc=f"共識標註中... ({i + 1}/{total})")
        try:
            source_boxes = {}
            w = h = None
            for name, model in models.items():
                boxes, iw, ih = predict_local_model(model, image_path)
                source_boxes[name] = boxes
                w, h = iw, ih
            if api_key:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError("圖片讀取失敗")
                h_, w_ = img.shape[:2]
                w, h = w or w_, h or h_
                source_boxes["VLM"] = predict_vlm(image_path, api_key, w, h)

            result = run_consensus_on_image(
                image_path, source_boxes, w, h, iou_threshold, vote_rule, merge_mode)

            for key in ("unanimous", "majority", "single", "conflict"):
                summary[key] += result["stats"][key]
            for src, cnt in result["stats"]["unique_by_source"].items():
                summary["unique_by_source"][src] = (
                    summary["unique_by_source"].get(src, 0) + cnt)

            filename = os.path.basename(image_path)
            base = os.path.splitext(filename)[0]
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError("圖片讀取失敗")
            cv2.imwrite(os.path.join(out_dir, filename), img_bgr)
            with open(os.path.join(out_dir, base + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result["txt_lines"]))
            with open(os.path.join(out_dir, base + ".xml"), "w", encoding="utf-8") as f:
                f.write(result["xml"])

            preview = draw_consensus_overlay(img_bgr, result["draw_boxes"])
            preview_path = os.path.join(out_dir, base + "_preview.jpg")
            cv2.imwrite(preview_path, preview)
            preview_items.append((preview_path, filename))
        except Exception as e:
            failed.append((image_path, str(e)))

    return summary, preview_items, failed
