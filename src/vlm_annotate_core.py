"""VLM(OpenAI Vision)自動標註核心邏輯。

annotate_batch() : 資料夾內一批圖片 → 並發呼叫 OpenAI Vision API → 依既有的
                    _HARD 人工複核慣例(見 annotate_core.py)寫出 .xml,
                    並額外寫出 YOLO 格式 .txt(固定 5 類 id,供直接訓練)。

class id 對照(專案固定,不可更動順序):
    0=Gloves  1=No_Gloves  2=No_Helmet  3=helmet  4=person
"""
import os
import re
import json
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

CLASS_NAME_TO_ID = {
    "Gloves": 0,
    "No_Gloves": 1,
    "No_Helmet": 2,
    "helmet": 3,
    "person": 4,
}

# 可用環境變數覆蓋模型(預設用便宜但視覺能力足夠的 mini 版)
VLM_MODEL = os.environ.get("OPENAI_VLM_MODEL", "gpt-4o-mini")

# 每百萬 token 的美元估算價格(粗估用,實際請以 OpenAI 官方牌價為準)
_PRICING_PER_M_TOKENS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

SYSTEM_PROMPT = """You are annotating images for a firefighter PPE compliance dataset.

TASK — follow this order strictly:
1. Find EVERY person in the image, including partially visible,
   back-facing, occluded, and background figures.
2. For EACH person, locate EVERY visible hand (up to 2 per person).
   Then classify each hand:
     - "Gloves"    → hand is wearing a glove of any type
     - "No_Gloves" → bare skin visible on the hand
3. For EACH person, locate the head. Then classify:
     - "helmet"    → wearing a helmet or hard hat
     - "No_Helmet" → head visible with no helmet (bare head,
                     cap, beanie, hood, or hair all count as No_Helmet)

CRITICAL RULES:
- Every visible hand MUST get exactly one box: either Gloves or
  No_Gloves. Never skip a hand because it is small, blurry,
  or partially occluded.
- Every visible head MUST get exactly one box: either helmet or
  No_Helmet.
- RECALL IS THE PRIORITY. If you are unsure whether something is
  a hand or a head, still output it with a low confidence score.
  A false positive is cheap; a miss is expensive.
- Do NOT label faces or people printed on posters, screens,
  monitors, mirrors, or reflections. Only real physical people.
- Bounding boxes must tightly enclose the object with no padding.
- One box per object. Do not output duplicate or stacked boxes
  on the same hand or head.

OUTPUT — JSON only, no prose, no markdown fences:
{
  "detections": [
    {"class": "person",    "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0},
    {"class": "No_Gloves", "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}
  ]
}
Coordinates are normalized to [0, 1] relative to image width and height.
If the image contains no people at all, return {"detections": []}.
"""


class VLMAnnotateError(Exception):
    """VLM 呼叫或回傳解析失敗時丟出,訊息帶原始回應片段方便除錯。"""


def resize_for_api(image_path, max_side=1568, quality=90):
    """縮到長邊 max_side、轉 JPEG(quality)。維持長寬比,不裁切,
    所以 API 回傳的 normalized [0,1] 座標對原圖依然成立。"""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def _strip_markdown_fence(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t.strip()


def call_vlm_api(image_path, api_key, model=None):
    """呼叫一次 OpenAI Vision API,回傳 (parsed_json, usage_dict)。
    失敗一律丟 VLMAnnotateError,不靜默吞掉。"""
    from openai import OpenAI  # 延遲匯入,openai 屬選用相依

    model = model or VLM_MODEL
    client = OpenAI(api_key=api_key)

    img_bytes = resize_for_api(image_path)
    b64 = base64.b64encode(img_bytes).decode("ascii")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        }
                    ],
                },
            ],
            max_tokens=4096,
        )
    except Exception as e:
        raise VLMAnnotateError(f"API 呼叫失敗: {e}") from e

    raw = resp.choices[0].message.content or ""
    cleaned = _strip_markdown_fence(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        snippet = cleaned[:200]
        raise VLMAnnotateError(f"回傳非合法 JSON({e}): {snippet!r}") from e

    if not isinstance(data, dict) or not isinstance(data.get("detections"), list):
        raise VLMAnnotateError(f"JSON 缺少 detections 陣列: {cleaned[:200]!r}")

    usage = resp.usage
    return data, {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }


def estimate_cost_usd(usage, model=None):
    model = model or VLM_MODEL
    price = _PRICING_PER_M_TOKENS.get(model)
    if not price:
        return 0.0
    return (usage["prompt_tokens"] / 1_000_000 * price["input"] +
            usage["completion_tokens"] / 1_000_000 * price["output"])


def detections_to_yolo_and_xml(detections, filename, w, h, conf_threshold):
    """把 VLM 偵測結果轉成 (txt_lines, xml_string, hard_count, box_count)。

    邊界處理: clamp 到 [0,1]、x1>x2 互換、w 或 h < 0.001 的框丟棄。
    信心 < conf_threshold 的框: XML 類別名加 _HARD 後綴(既有人工複核慣例,
    複核後才併回正常類別;YOLO txt 一律用固定的正常類別 id,不受影響)。
    """
    txt_lines = []
    xml = (f"<annotation><filename>{filename}</filename>"
           f"<size><width>{w}</width><height>{h}</height></size>")
    hard_count = 0
    box_count = 0

    for det in detections:
        cls_name = det.get("class")
        bbox = det.get("bbox")
        if cls_name not in CLASS_NAME_TO_ID:
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue

        try:
            conf = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        x1 = min(max(x1, 0.0), 1.0)
        x2 = min(max(x2, 0.0), 1.0)
        y1 = min(max(y1, 0.0), 1.0)
        y2 = min(max(y2, 0.0), 1.0)

        bw, bh = x2 - x1, y2 - y1
        if bw < 0.001 or bh < 0.001:
            continue

        cls_id = CLASS_NAME_TO_ID[cls_name]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        txt_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        is_hard = conf < conf_threshold
        if is_hard:
            hard_count += 1
        display_name = f"{cls_name}_HARD" if is_hard else cls_name

        xmin, ymin, xmax, ymax = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        xml += (f"<object><name>{display_name}</name><bndbox>"
                f"<xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
                f"<xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object>")
        box_count += 1

    xml += "</annotation>"
    return txt_lines, xml, hard_count, box_count


def annotate_one(image_path, api_key, conf_threshold, model=None):
    """單張圖:呼叫 API → 轉換 → 寫 .txt(YOLO) + .xml(含 _HARD 標記)。"""
    with Image.open(image_path) as im:
        w, h = im.size

    data, usage = call_vlm_api(image_path, api_key, model=model)
    txt_lines, xml, hard_count, box_count = detections_to_yolo_and_xml(
        data["detections"], os.path.basename(image_path), w, h, conf_threshold)

    base = os.path.splitext(image_path)[0]
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
    with open(base + ".xml", "w", encoding="utf-8") as f:
        f.write(xml)

    return {
        "path": image_path,
        "box_count": box_count,
        "hard_count": hard_count,
        "cost": estimate_cost_usd(usage, model=model),
    }


def annotate_batch(image_paths, api_key, conf_threshold, max_workers=4, model=None, progress_cb=None):
    """並發標註一批圖片。

    回傳 (results, failed):
      results: [{"path", "box_count", "hard_count", "cost"}, ...] 成功清單
      failed : [(path, error_message), ...] 失敗清單,絕不靜默吞掉
    """
    results = []
    failed = []
    total = len(image_paths)
    done = 0

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        future_to_path = {
            ex.submit(annotate_one, p, api_key, conf_threshold, model): p
            for p in image_paths
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            done += 1
            try:
                results.append(future.result())
            except VLMAnnotateError as e:
                failed.append((path, str(e)))
            except Exception as e:
                failed.append((path, f"未預期錯誤: {e}"))
            if progress_cb:
                progress_cb(done, total, os.path.basename(path))

    return results, failed
