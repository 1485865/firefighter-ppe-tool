"""稽核 Excel 嵌圖用:把標註框畫在縮圖上,讓組員不用開原圖就能核對。

- render_full_thumbnail(): 整張圖縮圖 + 畫出全部既有標註框(給漏標清單用,
  看框標在哪、哪裡明顯漏了)。
- render_crop_thumbnail(): 以單一異常框為中心,向外擴一圈裁切後縮圖,框本身
  用紅線標出(給異常框清單用,一眼判斷是不是誤報)。

線寬都是照「縮圖本身」的尺寸算,不是照原圖尺寸算,所以不管原圖解析度多高,
縮圖上的框永遠看得清楚。
"""
import os

from PIL import Image, ImageDraw, ImageFont

from audit_core import ID_TO_CLASS

# 各類別畫框顏色(RGB),同時給縮圖畫框跟「統計摘要」圖例共用。
CLASS_COLORS = {
    "Gloves": (0, 176, 80),
    "No_Gloves": (255, 0, 0),
    "No_Helmet": (255, 140, 0),
    "helmet": (0, 112, 192),
    "person": (146, 39, 143),
}
DEFAULT_CLASS_COLOR = (128, 128, 128)

_FONT_CACHE = {}


def _get_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for candidate in (r"C:\Windows\Fonts\msjh.ttc", r"C:\Windows\Fonts\arial.ttf"):
        try:
            font = ImageFont.truetype(candidate, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def render_full_thumbnail(image_path, boxes, out_path, max_side=200):
    """把整張圖縮成長邊 max_side px,畫上 boxes(YOLO 正規化座標)。
    回傳 (width, height) 像素;讀圖失敗回傳 None。"""
    img = Image.open(image_path)
    img = img.convert("RGB")
    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        return None
    scale = max_side / max(orig_w, orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    thumb = img.resize((new_w, new_h), Image.LANCZOS)
    draw = ImageDraw.Draw(thumb)

    line_w = max(2, round(min(new_w, new_h) / 100))
    font = _get_font(max(9, round(new_h / 16)))

    for cls_id, cx, cy, bw, bh in boxes:
        cls_name = ID_TO_CLASS.get(cls_id, "?")
        color = CLASS_COLORS.get(cls_name, DEFAULT_CLASS_COLOR)
        x1, y1 = (cx - bw / 2) * new_w, (cy - bh / 2) * new_h
        x2, y2 = (cx + bw / 2) * new_w, (cy + bh / 2) * new_h
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        label_y = y1 - (line_w + 11) if y1 - (line_w + 11) >= 0 else y1 + 2
        draw.text((max(0, x1 + 1), label_y), cls_name, fill=color, font=font)

    thumb.save(out_path, "PNG")
    return new_w, new_h


def render_crop_thumbnail(image_path, cx, cy, bw, bh, out_path, max_side=200, pad_ratio=0.5):
    """以框(YOLO 正規化座標)為中心向外擴 pad_ratio(預設 50%)裁切,框本身用
    紅線標出,再縮成長邊 max_side px。回傳 (width, height);失敗回傳 None。"""
    img = Image.open(image_path)
    img = img.convert("RGB")
    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        return None

    bx1, by1 = (cx - bw / 2) * orig_w, (cy - bh / 2) * orig_h
    bx2, by2 = (cx + bw / 2) * orig_w, (cy + bh / 2) * orig_h
    box_w, box_h = max(bx2 - bx1, 1.0), max(by2 - by1, 1.0)
    pad_x, pad_y = box_w * pad_ratio, box_h * pad_ratio

    crop_x1 = max(0, bx1 - pad_x)
    crop_y1 = max(0, by1 - pad_y)
    crop_x2 = min(orig_w, bx2 + pad_x)
    crop_y2 = min(orig_h, by2 + pad_y)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    crop_w, crop_h = crop.size
    scale = max_side / max(crop_w, crop_h)
    new_w = max(1, round(crop_w * scale))
    new_h = max(1, round(crop_h * scale))
    thumb = crop.resize((new_w, new_h), Image.LANCZOS)
    draw = ImageDraw.Draw(thumb)

    rel_x1 = (bx1 - crop_x1) * scale
    rel_y1 = (by1 - crop_y1) * scale
    rel_x2 = (bx2 - crop_x1) * scale
    rel_y2 = (by2 - crop_y1) * scale
    line_w = max(2, round(min(new_w, new_h) / 60))
    draw.rectangle([rel_x1, rel_y1, rel_x2, rel_y2], outline=(255, 0, 0), width=line_w)

    thumb.save(out_path, "PNG")
    return new_w, new_h
