"""標註共用核心(extractor 與 batch_annotate 共用同一套強邏輯)。

annotate() : 單張圖 → 跑主模型(+可選輔助模型 ensemble 抓漏)→
                      conf 過濾 / IoU 合併 / padding 擴框 / HARD 困難樣本標記,
                      產出 XML、YOLO txt、繪框座標、EDA 數據與 ensemble 戰況。
build_eda_figures() : 把彙整後的統計畫成三張分析圖(標籤分佈 / 空間熱力 / 長寬比)。
"""
import cv2
import matplotlib
matplotlib.use("Agg") # 無視窗後端,執行緒中產圖才安全
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei','PingFang HK','sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / float(a1 + a2 - inter)


def annotate(source, filename, model1, model2, conf_thr, hard_thr, padding):
    """對單張圖做完整標註。

    source : 圖片路徑或 BGR np 陣列(ultralytics 兩者皆吃)。
    filename: 寫進 XML <filename> 的檔名。
    model1 : 主模型(決定哪些標籤可上桌)。
    model2 : 輔助模型或 None(ensemble 抓漏)。
    回傳 dict:w, h, xml, txt_lines, class_counts, eda_cx/cy/ar, draw_boxes, stats。
    """
    # 主模型認得的標籤白名單 + 名稱→類別 id(供寫 YOLO txt)
    allowed = list(model1.names.values())
    name_to_id = {v: k for k, v in model1.names.items()}

    res1 = model1(source, verbose=False)
    h, w, _ = res1[0].orig_img.shape

    final_boxes = [] # [cls_name, conf, x1, y1, x2, y2]
    for box in res1[0].boxes:
        conf = float(box.conf[0])
        if conf < conf_thr:
            continue
        cls_name = model1.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        final_boxes.append([cls_name, conf, x1, y1, x2, y2])

    main_box_count = len(final_boxes)
    duplicate_count = 0
    added_by_aux = 0
    aux_details = []

    # ----- 輔助模型 ensemble 抓漏 -----
    if model2 is not None:
        res2 = model2(source, verbose=False)
        for box in res2[0].boxes:
            conf = float(box.conf[0])
            if conf < conf_thr:
                continue
            cls_name = model2.names[int(box.cls[0])]
            if cls_name not in allowed:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            is_dup = False
            for eb in final_boxes[:main_box_count]:
                if calculate_iou([x1, y1, x2, y2], eb[2:6]) > 0.5:
                    is_dup = True
                    break
            if is_dup:
                duplicate_count += 1
            else:
                final_boxes.append([cls_name, conf, x1, y1, x2, y2])
                added_by_aux += 1
                aux_details.append(f"{cls_name}({conf:.2f})")

    # ----- 組裝 XML / txt / 繪框 / EDA -----
    xml = (f"<annotation><filename>{filename}</filename>"
           f"<size><width>{w}</width><height>{h}</height></size>")
    txt_lines = []
    class_counts = {}
    eda_cx, eda_cy, eda_ar = [], [], []
    draw_boxes = [] # (display_name, conf, x1, y1, x2, y2) 已 padding 的整數座標

    for cls_name, conf, x1, y1, x2, y2 in final_boxes:
        display_name = f"{cls_name}_HARD" if conf <= hard_thr else cls_name
        class_counts[display_name] = class_counts.get(display_name, 0) + 1

        # YOLO txt 用未 padding 的歸一化中心點(類別 id 走主模型命名空間)
        cls_id = name_to_id.get(cls_name)
        if cls_id is not None:
            ncx = ((x1 + x2) / 2) / w
            ncy = ((y1 + y2) / 2) / h
            nw = (x2 - x1) / w
            nh = (y2 - y1) / h
            txt_lines.append(f"{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")

        # 邊界擴張
        bw, bh = x2 - x1, y2 - y1
        p = padding / 100.0
        x1 = max(0, x1 - (bw * p / 2))
        y1 = max(0, y1 - (bh * p / 2))
        x2 = min(w, x2 + (bw * p / 2))
        y2 = min(h, y2 + (bh * p / 2))

        xml += (f"<object><name>{display_name}</name><bndbox>"
                f"<xmin>{int(x1)}</xmin><ymin>{int(y1)}</ymin>"
                f"<xmax>{int(x2)}</xmax><ymax>{int(y2)}</ymax></bndbox></object>")

        eda_cx.append(((x1 + x2) / 2) / w)
        eda_cy.append(1 - ((y1 + y2) / 2) / h) # Y 反轉供畫圖
        ar = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
        if ar > 0:
            eda_ar.append(ar)

        draw_boxes.append((display_name, conf, int(x1), int(y1), int(x2), int(y2)))

    xml +="</annotation>"

    return {
        "w": w,"h": h,"xml": xml,"txt_lines": txt_lines,
        "class_counts": class_counts,
        "eda_cx": eda_cx,"eda_cy": eda_cy,"eda_ar": eda_ar,
        "draw_boxes": draw_boxes,
        "stats": {
            "main_box_count": main_box_count,
            "duplicate_count": duplicate_count,
            "added_by_aux": added_by_aux,
            "aux_details": aux_details,
        },
    }


def draw_overlay(img_bgr, filename, draw_boxes):
    """在圖上印檔名 + 畫橘色框(供結果預覽)。回傳繪好的 BGR 圖。"""
    out = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 左上角檔名(黑底白字)
    ts = cv2.getTextSize(filename, font, 0.7, 2)[0]
    cv2.rectangle(out, (5, 5), (5 + ts[0] + 10, 5 + ts[1] + 15), (0, 0, 0), -1)
    cv2.putText(out, filename, (10, 5 + ts[1] + 8), font, 0.7, (255, 255, 255), 2)
    for display_name, conf, x1, y1, x2, y2 in draw_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(out, f"{display_name} {conf:.2f}", (x1, y1 - 5),
                    font, 0.5, (0, 165, 255), 2)
    return out


def build_eda_figures(class_counts, eda_cx, eda_cy, eda_ar):
    """把彙整統計畫成三張圖,回傳 (fig_stats, fig_heat, fig_ar)。"""
    # 1. 標籤數量長條圖
    fig_stats, ax = plt.subplots(figsize=(8, 4))
    if class_counts:
        labels, counts = list(class_counts.keys()), list(class_counts.values())
        bars = ax.bar(labels, counts, color='royalblue', edgecolor='black')
        ax.set_title("本次批次標註產生的標籤數量")
        ax.set_ylim(0, max(counts) * 1.15)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        fig_stats.tight_layout()
        for bar in bars:
            yv = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yv + (max(counts) * 0.01),
                    int(yv), ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5,"沒有偵測到任何標籤", ha='center', va='center')
        ax.axis('off')

    # 2. 空間分佈熱力圖
    fig_heat, ax2 = plt.subplots(figsize=(5, 5))
    if eda_cx:
        hb = ax2.hexbin(eda_cx, eda_cy, gridsize=15, cmap='YlOrRd', extent=[0, 1, 0, 1])
        ax2.set_title("空間分佈熱力圖 (Spatial Distribution)")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        fig_heat.colorbar(hb, ax=ax2, label='Density')
    else:
        ax2.text(0.5, 0.5,"無數據", ha='center', va='center')
        ax2.axis('off')

    # 3. 長寬比分佈
    fig_ar, ax3 = plt.subplots(figsize=(5, 5))
    if eda_ar:
        ax3.hist(eda_ar, bins=20, color='skyblue', edgecolor='black')
        ax3.set_title("框型長寬比分佈 (Aspect Ratio)")
        ax3.axvline(1.0, color='red', linestyle='dashed', linewidth=1)
    else:
        ax3.text(0.5, 0.5,"無數據", ha='center', va='center')
        ax3.axis('off')

    return fig_stats, fig_heat, fig_ar
