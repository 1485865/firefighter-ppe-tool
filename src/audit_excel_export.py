"""把 audit_core.audit_dataset() 的結果匯出成 Excel(.xlsx),方便逐筆核對。

三個工作表:漏標清單 / 異常框清單 / 統計摘要。
"""
import os
import statistics
from urllib.request import pathname2url

from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

VERDICT_CHOICES = ["確認漏標", "誤報", "已修正", "待確認"]

# 異常原因上色:跟 audit_core.ANOMALY_CATEGORIES 對應,額外情況(如標註格式
# 錯誤)沒列在這裡的話會退回灰色,不影響匯出。
CATEGORY_COLORS = {
    "框過大": "FFC7CE",
    "框過小": "FFEB9C",
    "長寬比異常": "FCE4D6",
    "超出邊界": "E4DFEC",
    "重複框": "DDEBF7",
}
DEFAULT_CATEGORY_COLOR = "D9D9D9"

HEADER_FILL = PatternFill("solid", fgColor="4472C4")
HEADER_FONT = Font(color="FFFFFF", bold=True)
HYPERLINK_FONT = Font(color="0563C1", underline="single")

MAX_COL_WIDTH = 60


def _image_to_file_url(path):
    try:
        return "file:" + pathname2url(os.path.abspath(path))
    except Exception:
        return None


def _style_header(ws, n_cols):
    for col in range(1, n_cols + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(n_cols)}1"


def _autosize_columns(ws, n_cols, n_rows, sample_rows=200):
    """用前面幾百列估欄寬,避免 2000+ 列時逐列量測拖慢速度。"""
    widths = [0] * (n_cols + 1)
    check_rows = min(n_rows, sample_rows)
    for row in range(1, check_rows + 2):  # +1 涵蓋標題列
        for col in range(1, n_cols + 1):
            value = ws.cell(row=row, column=col).value
            if value is None:
                continue
            length = len(str(value))
            if length > widths[col]:
                widths[col] = length
    for col in range(1, n_cols + 1):
        ws.column_dimensions[get_column_letter(col)].width = min(
            MAX_COL_WIDTH, max(10, widths[col] + 2))


def _add_verdict_validation(ws, col_idx, n_rows):
    if n_rows <= 0:
        return
    dv = DataValidation(
        type="list",
        formula1='"' + ",".join(VERDICT_CHOICES) + '"',
        allow_blank=True,
    )
    ws.add_data_validation(dv)
    col_letter = get_column_letter(col_idx)
    dv.add(f"{col_letter}2:{col_letter}{n_rows + 1}")


def _write_missing_sheet(wb, missing_labels):
    ws = wb.create_sheet("漏標清單")
    headers = ["檔名", "完整路徑", "person 框數", "手部框數", "手部預期數",
               "手部缺口", "頭部框數", "頭部缺口", "嚴重度", "判定", "備註"]
    ws.append(headers)

    for r in missing_labels:
        full_path = os.path.abspath(r["image"])
        row = [
            os.path.basename(r["image"]), full_path, r["person"], r["hands"],
            r["expected_hands"], r["hand_gap"], r["heads"], r["head_gap"],
            r["gap"], "", "",
        ]
        ws.append(row)
        row_idx = ws.max_row
        url = _image_to_file_url(r["image"])
        if url:
            cell = ws.cell(row=row_idx, column=1)
            cell.hyperlink = url
            cell.font = HYPERLINK_FONT

    n_rows = len(missing_labels)
    _style_header(ws, len(headers))
    _add_verdict_validation(ws, headers.index("判定") + 1, n_rows)

    if n_rows > 0:
        severity_col = headers.index("嚴重度") + 1
        col_letter = get_column_letter(severity_col)
        rule = ColorScaleRule(
            start_type="min", start_color="63BE7B",
            mid_type="percentile", mid_value=50, mid_color="FFEB84",
            end_type="max", end_color="F8696B",
        )
        ws.conditional_formatting.add(f"{col_letter}2:{col_letter}{n_rows + 1}", rule)

    _autosize_columns(ws, len(headers), n_rows)
    return ws


def _write_anomaly_sheet(wb, anomaly_boxes):
    ws = wb.create_sheet("異常框清單")
    headers = ["檔名", "完整路徑", "類別", "異常原因", "框座標(cx,cy,w,h)",
               "框面積佔比(%)", "長寬比", "判定"]
    ws.append(headers)

    for a in anomaly_boxes:
        full_path = os.path.abspath(a["image"])
        if a["cx"] is not None:
            coord = f"({a['cx']:.4f}, {a['cy']:.4f}, {a['w']:.4f}, {a['h']:.4f})"
        else:
            coord = ""
        area_pct = round(a["area_pct"], 3) if a["area_pct"] is not None else ""
        ar = round(a["aspect_ratio"], 3) if a["aspect_ratio"] is not None else ""
        row = [
            os.path.basename(a["image"]), full_path, a["cls_name"] or "",
            a["category"], coord, area_pct, ar, "",
        ]
        ws.append(row)
        row_idx = ws.max_row

        url = _image_to_file_url(a["image"])
        if url:
            cell = ws.cell(row=row_idx, column=1)
            cell.hyperlink = url
            cell.font = HYPERLINK_FONT

        color = CATEGORY_COLORS.get(a["category"], DEFAULT_CATEGORY_COLOR)
        reason_cell = ws.cell(row=row_idx, column=headers.index("異常原因") + 1)
        reason_cell.fill = PatternFill("solid", fgColor=color)

    n_rows = len(anomaly_boxes)
    _style_header(ws, len(headers))
    _add_verdict_validation(ws, headers.index("判定") + 1, n_rows)
    _autosize_columns(ws, len(headers), n_rows)
    return ws


def _write_summary_sheet(wb, result):
    ws = wb.create_sheet("統計摘要")

    total_images = result["total_images"]
    n_missing = len(result["missing_labels"])
    anomaly_boxes = result["anomaly_boxes"]
    n_anomaly = len(anomaly_boxes)
    missing_pct = (n_missing / total_images * 100.0) if total_images else 0.0

    row = 1
    ws.cell(row=row, column=1, value="總圖數").font = Font(bold=True)
    ws.cell(row=row, column=2, value=total_images)
    row += 1
    ws.cell(row=row, column=1, value="疑似漏標張數")
    ws.cell(row=row, column=2, value=n_missing)
    ws.cell(row=row, column=3, value=f"{missing_pct:.2f}%")
    row += 2

    ws.cell(row=row, column=1, value="異常框總數").font = Font(bold=True)
    ws.cell(row=row, column=2, value=n_anomaly)
    row += 1

    cat_counts = {}
    for a in anomaly_boxes:
        cat_counts[a["category"]] = cat_counts.get(a["category"], 0) + 1
    ws.cell(row=row, column=1, value="異常原因").font = Font(bold=True)
    ws.cell(row=row, column=2, value="數量").font = Font(bold=True)
    ws.cell(row=row, column=3, value="佔比").font = Font(bold=True)
    row += 1
    for cat, count in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        pct = (count / n_anomaly * 100.0) if n_anomaly else 0.0
        ws.cell(row=row, column=1, value=cat)
        ws.cell(row=row, column=2, value=count)
        ws.cell(row=row, column=3, value=f"{pct:.2f}%")
        row += 1
    row += 1

    class_counts = result["class_counts"]
    class_areas = result["class_areas"]
    ws.cell(row=row, column=1, value="類別").font = Font(bold=True)
    ws.cell(row=row, column=2, value="實例數").font = Font(bold=True)
    ws.cell(row=row, column=3, value="面積佔比 min(%)").font = Font(bold=True)
    ws.cell(row=row, column=4, value="面積佔比 中位數(%)").font = Font(bold=True)
    ws.cell(row=row, column=5, value="面積佔比 max(%)").font = Font(bold=True)
    row += 1
    for cls_name, count in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        areas = class_areas.get(cls_name, [])
        ws.cell(row=row, column=1, value=cls_name)
        ws.cell(row=row, column=2, value=count)
        if areas:
            ws.cell(row=row, column=3, value=round(min(areas), 3))
            ws.cell(row=row, column=4, value=round(statistics.median(areas), 3))
            ws.cell(row=row, column=5, value=round(max(areas), 3))
        row += 1

    for col in range(1, 6):
        ws.column_dimensions[get_column_letter(col)].width = 20
    return ws


def export_excel(result, out_path):
    """把 audit_dataset() 的結果寫成 out_path(.xlsx)。回傳 out_path。"""
    wb = Workbook()
    wb.remove(wb.active)  # 預設會建一個空白 Sheet,先移掉

    _write_missing_sheet(wb, result["missing_labels"])
    _write_anomaly_sheet(wb, result["anomaly_boxes"])
    _write_summary_sheet(wb, result)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    wb.save(out_path)
    return out_path
