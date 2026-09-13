import gradio as gr
import os
import glob
import shutil
import random
import traceback
import yaml
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from notifier import send_alert
from utils import list_models, upload_to_roboflow, get_yolo_model

# ==========================================================
# 實驗室 V7：YOLO 知識蒸餾 (Knowledge Distillation)
# Mode 1：雙老師偽標籤蒸餾 (response-based，穩定，官方 API)
# PPE 老師(你們的 weights.pt)標 4 個 PPE 類，
# 官方老師(COCO yolo11x)標 person —— 各取所長。
# Mode 2：特徵蒸餾 (feature-based，實驗性，自訂訓練器)
# ==========================================================

DISTILL_ROOT ="runs_distill"
TRANSFER_DIR ="transfer_to_tab3"# 與第二頁(抽幀)共用的傳送門
EXTRACT_DIR ="extracted_frames"# 抽幀頁的原始輸出

# 官方預訓練底模（第一次使用會自動從網路下載）
PRETRAINED_TEACHERS = ["yolo11x.pt","yolo11l.pt","yolov8x.pt","yolov8l.pt"]
PRETRAINED_STUDENTS = ["yolo11n.pt","yolov8n.pt","yolo11s.pt","yolov8s.pt"]
# 專門用來標 person 的官方底模(COCO，person 通常比自訓模型準很多)
PERSON_TEACHERS = ["yolo11x.pt","yolo11l.pt","yolo11m.pt","yolov8x.pt"]
NO_PERSON_TEACHER ="(不使用，person 也交給 PPE 老師)"

# 預覽框顏色(BGR)
_PALETTE = [(0, 180, 255), (0, 255, 0), (0, 0, 255),
            (255, 128, 0), (255, 0, 255), (128, 0, 255), (0, 255, 255)]


def get_local_models():
    return list_models(with_placeholder=False)


def teacher_choices():
    return get_local_models() + PRETRAINED_TEACHERS


def student_choices():
    # 使用者偏好 nano，所以放最前面當預設
    return PRETRAINED_STUDENTS + get_local_models()


def ppe_teacher_choices():
    # PPE 老師應是你們自己訓練的模型,放最前面
    return get_local_models() + PRETRAINED_TEACHERS


def person_teacher_choices():
    return PERSON_TEACHERS + get_local_models() + [NO_PERSON_TEACHER]


def default_ppe_teacher():
    """優先挑名字像 weights 的本地模型當預設 PPE 老師。"""
    locals_ = get_local_models()
    for m in locals_:
        if "weight" in os.path.basename(m).lower():
            return m
    return locals_[0] if locals_ else PRETRAINED_TEACHERS[0]


def device_str():
    return "0" if torch.cuda.is_available() else "cpu"


def _find_person_id(names):
    """在 {id: name} 裡找名稱為 person 的類別 id,找不到回傳 None。"""
    for i, n in (names or {}).items():
        if str(n).strip().lower() =="person":
            return int(i)
    return None


def _resolve_schema(ppe_teacher, person_teacher):
    """決定最終要用的類別 schema 與各老師的 person id。

    - schema 固定採用 PPE 老師(你們的模型)的類別表,**不採用官方老師的 COCO 80 類**。
    - PPE 老師自己的 person 會被丟棄,改由官方老師負責(若有選)。
    """
    ppe_names = {int(k): v for k, v in ppe_teacher.names.items()}
    ppe_person_id = _find_person_id(ppe_names)
    coco_person_id = _find_person_id(person_teacher.names) if person_teacher else None

    schema = dict(ppe_names)
    if ppe_person_id is not None:
        target_person_id = ppe_person_id
    elif person_teacher is not None:
        # PPE 老師沒有 person 類,但要用官方補 → 在 schema 末端新增一類
        target_person_id = (max(schema) + 1) if schema else 0
        schema[target_person_id] ="person"
    else:
        target_person_id = None
    return ppe_names, ppe_person_id, coco_person_id, target_person_id, schema


def _collect_boxes(path, ppe_teacher, person_teacher, ppe_conf, person_conf,
                   ppe_person_id, coco_person_id, target_person_id, cache):
    """對單張圖跑兩位老師,回傳合併後的 [(cls, cx, cy, w, h)](皆為歸一化)。
    PPE 老師:取全部類別,但丟掉它自己的 person。
    官方老師:只取 person,並重映射成 schema 裡的 person id。
    """
    if path in cache:
        return cache[path]
    boxes = []
    r1 = ppe_teacher.predict(path, conf=float(ppe_conf), verbose=False)[0]
    if r1.boxes is not None:
        for b in r1.boxes:
            cls = int(b.cls[0])
            if ppe_person_id is not None and cls == ppe_person_id:
                continue # PPE 老師的 person 不要,改用官方
            cx, cy, w, h = b.xywhn[0].tolist()
            boxes.append((cls, cx, cy, w, h))
    if person_teacher is not None and coco_person_id is not None and target_person_id is not None:
        r2 = person_teacher.predict(path, conf=float(person_conf),
                                    classes=[coco_person_id], verbose=False)[0]
        if r2.boxes is not None:
            for b in r2.boxes:
                cx, cy, w, h = b.xywhn[0].tolist()
                boxes.append((target_person_id, cx, cy, w, h))
    cache[path] = boxes
    return boxes


def _draw_preview(path, boxes, names, save_path):
    """把合併後的框畫到圖上存檔,供 Gallery 肉眼檢查老師標得對不對。"""
    im = cv2.imread(path)
    if im is None:
        return None
    H, W = im.shape[:2]
    for (cls, cx, cy, w, h) in boxes:
        x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        color = _PALETTE[int(cls) % len(_PALETTE)]
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        cv2.putText(im, str(names.get(int(cls), cls)), (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(save_path, im)
    return save_path


def _write_voc_xml(xml_path, boxes, schema, w, h, filename):
    """把歸一化框寫成 Pascal VOC XML(供 Roboflow annotation_path 使用)。"""
    xml = (f"<annotation><filename>{filename}</filename>"
           f"<size><width>{w}</width><height>{h}</height></size>")
    for (cls, cx, cy, bw, bh) in boxes:
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        name = schema.get(int(cls), str(cls))
        xml += (f"<object><name>{name}</name><bndbox>"
                f"<xmin>{int(x1)}</xmin><ymin>{int(y1)}</ymin>"
                f"<xmax>{int(x2)}</xmax><ymax>{int(y2)}</ymax></bndbox></object>")
    xml +="</annotation>"
    with open(xml_path,"w", encoding="utf-8") as fh:
        fh.write(xml)


def receive_from_extractor():
    """讀第二頁傳送過來的圖(transfer_to_tab3);沒有就退而讀 extracted_frames。"""
    pats = ("*.jpg","*.jpeg","*.png")
    imgs = []
    for d in (TRANSFER_DIR, EXTRACT_DIR):
        for pat in pats:
            imgs += glob.glob(os.path.join(d, pat))
        if imgs:
            break
    return imgs if imgs else None


# ==========================================================
# Mode 1：雙老師偽標籤蒸餾
# ==========================================================
def _split_assignments(image_files):
    paths = [f.name for f in image_files]
    random.seed(42)
    random.shuffle(paths)
    if len(paths) <= 1:
        return paths, ([(p,"train") for p in paths] + [(p,"val") for p in paths]), True
    n_val = max(1, int(len(paths) * 0.2))
    val_paths, train_paths = paths[:n_val], paths[n_val:]
    return paths, ([(p,"train") for p in train_paths] +
                   [(p,"val") for p in val_paths]), False


def _load_teachers(log, gallery, ppe_teacher_path, person_teacher_path):
    """共用:載入兩位老師並算出 schema。回傳 (ppe_teacher, person_teacher, ids..., schema)。

    老師模型只拿來 .predict(),不會被訓練或保留 .track() 狀態,走快取安全——
    「預覽老師標註」在調信心門檻時常常連續點好幾次,快取能省掉重複讀檔。
    """
    ppe_teacher = get_yolo_model(ppe_teacher_path)
    person_teacher = None
    if (person_teacher_path and "找不到" not in str(person_teacher_path)
            and str(person_teacher_path) != NO_PERSON_TEACHER):
        person_teacher = get_yolo_model(person_teacher_path)
    ids = _resolve_schema(ppe_teacher, person_teacher)
    return (ppe_teacher, person_teacher) + ids


def preview_labels(image_files, ppe_teacher_path, person_teacher_path,
                   ppe_conf, person_conf, progress=gr.Progress()):
    """只標註不訓練,把前 30 張的標註結果畫出來給使用者檢查。
    同時把「原圖 + VOC XML」存成一份,供旁邊的 Roboflow 上傳按鈕使用。"""
    logs, gallery, upload_list = [], [], []

    def log(m):
        logs.append(m)
        return "\n".join(logs)

    try:
        if not image_files:
            yield log("請先上傳或接收圖片。"), gallery, upload_list
            return
        if not ppe_teacher_path or "找不到" in str(ppe_teacher_path):
            yield log("請選擇 PPE 老師模型(你們訓練的 weights.pt)。"), gallery, upload_list
            return

        yield log(f"載入 PPE 老師:{ppe_teacher_path} ..."), gallery, upload_list
        (ppe_teacher, person_teacher, ppe_names, ppe_person_id,
         coco_person_id, target_person_id, schema) = _load_teachers(
            log, gallery, ppe_teacher_path, person_teacher_path)
        yield log(f"類別 schema(採用 PPE 老師的,不用 COCO):{schema}"), gallery, upload_list
        if person_teacher is not None:
            yield log(f"person 改由官方老師負責:COCO id {coco_person_id} →"
                      f"schema id {target_person_id};PPE 老師的 person 已丟棄。"), gallery, upload_list
        else:
            yield log("未選官方 person 老師,person 仍由 PPE 老師負責。"), gallery, upload_list

        prev_dir = os.path.join(DISTILL_ROOT,"_preview")
        shutil.rmtree(prev_dir, ignore_errors=True)
        os.makedirs(prev_dir, exist_ok=True)
        raw_dir = os.path.join(DISTILL_ROOT,"_preview_raw")
        shutil.rmtree(raw_dir, ignore_errors=True)
        os.makedirs(raw_dir, exist_ok=True)

        paths = [f.name for f in image_files][:30]
        cache = {}
        for i, p in enumerate(paths):
            progress((i + 1) / len(paths), desc=f"老師標註預覽 {i + 1}/{len(paths)}")
            boxes = _collect_boxes(p, ppe_teacher, person_teacher, ppe_conf, person_conf,
                                   ppe_person_id, coco_person_id, target_person_id, cache)
            sp = os.path.join(prev_dir, f"prev_{i:04d}.jpg")
            _draw_preview(p, boxes, schema, sp)

            # 原圖(不畫框)+ VOC XML,存一份給 Roboflow 上傳用
            ext = os.path.splitext(p)[1] or ".jpg"
            raw_path = os.path.join(raw_dir, f"prev_{i:04d}{ext}")
            shutil.copy(p, raw_path)
            im = cv2.imread(raw_path)
            h, w = im.shape[:2] if im is not None else (0, 0)
            xml_path = os.path.splitext(raw_path)[0] +".xml"
            _write_voc_xml(xml_path, boxes, schema, w, h, os.path.basename(raw_path))
            upload_list.append(raw_path)

            n_person = sum(1 for b in boxes if b[0] == target_person_id)
            n_ppe = len(boxes) - n_person
            gallery = gallery + [(sp, f"{os.path.basename(p)} | PPE×{n_ppe} person×{n_person}")]
            yield log(f"{os.path.basename(p)}: PPE {n_ppe} 框, person {n_person} 框"), gallery, upload_list
        yield log(f"預覽完成(最多顯示 30 張)。確認標得對,可以按右邊「上傳到 Roboflow」,"
                  f"或往下開始正式蒸餾。"), gallery, upload_list
    except Exception as e:
        yield log(f"預覽時發生錯誤:{e}\n{traceback.format_exc()}"), gallery, upload_list


def run_dual_distill(image_files, ppe_teacher_path, person_teacher_path,
                     ppe_conf, person_conf, student_base,
                     epochs, imgsz, batch, out_name, progress=gr.Progress()):
    logs, gallery = [], []

    def log(m):
        logs.append(m)
        return "\n".join(logs)

    try:
        if not image_files:
            yield log("請先上傳或接收未標註圖片。"), gallery
            return
        if not ppe_teacher_path or "找不到" in str(ppe_teacher_path):
            yield log("請選擇 PPE 老師模型(你們訓練的 weights.pt)。"), gallery
            return
        out_name = (out_name or "student_distilled").strip().replace("","_")

        ds = os.path.join(DISTILL_ROOT, out_name,"dataset")
        for sub in ["images/train","images/val","labels/train","labels/val"]:
            os.makedirs(os.path.join(ds, sub), exist_ok=True)
        prev_dir = os.path.join(DISTILL_ROOT, out_name,"preview")
        shutil.rmtree(prev_dir, ignore_errors=True)
        os.makedirs(prev_dir, exist_ok=True)

        progress(0.03, desc="載入老師模型...")
        yield log(f"載入 PPE 老師:{ppe_teacher_path} ..."), gallery
        (ppe_teacher, person_teacher, ppe_names, ppe_person_id,
         coco_person_id, target_person_id, schema) = _load_teachers(
            log, gallery, ppe_teacher_path, person_teacher_path)
        yield log(f"固定使用專案 schema(不採用老師的 COCO 類別表):{schema}"), gallery
        if person_teacher is not None:
            yield log(f"person 由官方老師 {os.path.basename(str(person_teacher_path))} 負責"
                      f"(COCO id {coco_person_id} → id {target_person_id}),"
                      f"PPE 老師的 person 已丟棄。"), gallery
        else:
            yield log("未選官方 person 老師,person 由 PPE 老師負責(精度可能較差)。"), gallery

        # 切分 train/val
        paths, assignments, single = _split_assignments(image_files)
        if single:
            yield log("只有 1 張圖,讓它同時兼任訓練與驗證集(僅供測試流程)。"), gallery

        # 雙老師逐張標註(同張圖只推論一次,結果快取)
        cache, labeled = {}, set()
        total = len(assignments)
        for i, (p, split) in enumerate(assignments):
            progress(0.05 + 0.45 * (i / max(1, total)), desc=f"雙老師標註 {i + 1}/{total}")
            boxes = _collect_boxes(p, ppe_teacher, person_teacher, ppe_conf, person_conf,
                                   ppe_person_id, coco_person_id, target_person_id, cache)
            ext = os.path.splitext(p)[1] or ".jpg"
            base = f"img_{i:05d}"
            shutil.copy(p, os.path.join(ds, f"images/{split}", base + ext))
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"for (c, cx, cy, w, h) in boxes]
            if boxes:
                labeled.add(p)
            with open(os.path.join(ds, f"labels/{split}", base +".txt"),
                      "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))

            if len(gallery) < 24: # 順手存一批預覽圖讓使用者看老師成果
                sp = os.path.join(prev_dir, base +".jpg")
                _draw_preview(p, boxes, schema, sp)
                n_person = sum(1 for b in boxes if b[0] == target_person_id)
                n_ppe = len(boxes) - n_person
                gallery = gallery + [(sp, f"{base} | PPE×{n_ppe} person×{n_person}")]
                yield log(f"{os.path.basename(p)} → PPE {n_ppe} 框, person {n_person} 框"), gallery

        yield log(f"標註完成:共 {len(paths)} 張,其中 {len(labeled)} 張偵測到物件。"), gallery
        if not labeled:
            yield log("兩位老師都沒偵測到任何物件,請調低信心門檻或換老師模型。"), gallery
            return

        # 寫 data.yaml(用固定 schema)
        data_yaml = os.path.join(DISTILL_ROOT, out_name,"data.yaml")
        with open(data_yaml,"w", encoding="utf-8") as fh:
            yaml.safe_dump({
                "path": os.path.abspath(ds),
                "train":"images/train",
                "val":"images/val",
                "names": schema,
            }, fh, allow_unicode=True)
        yield log(f"已用專案 schema 產生資料集設定:{data_yaml}"), gallery

        # 訓練學生
        progress(0.55, desc="開始訓練學生模型...")
        yield log(f"開始訓練學生 {student_base}(epochs={epochs}, imgsz={imgsz})..."), gallery
        student = YOLO(student_base)
        student.train(
            data=data_yaml, epochs=int(epochs), imgsz=int(imgsz),
            batch=int(batch), device=device_str(),
            project=os.path.join(DISTILL_ROOT, out_name), name="train",
            exist_ok=True, verbose=False,
        )

        best = str(student.trainer.best)
        if os.path.exists(best):
            final = os.path.join("models", f"{out_name}.pt")
            shutil.copy(best, final)
            send_alert("雙老師蒸餾完成", f"學生模型已輸出:{final}")
            progress(1.0, desc="完成")
            yield log(f"蒸餾完成!學生模型已存到 {final}"
                      f"(維持你們的 PPE schema,可在其他分頁直接選用)"), gallery
        else:
            yield log("訓練跑完但找不到 best.pt,請檢查 runs_distill 資料夾。"), gallery

    except Exception as e:
        yield log(f"發生錯誤:{e}\n{traceback.format_exc()}"), gallery


# ==========================================================
# Mode 2：特徵蒸餾(自訂訓練器)
# 抓老師/學生送進 Detect 頭的三層金字塔特徵,用 1x1 conv 對齊
# 通道後算 MSE。架構無關,跨 YOLO 版本相對穩。
# ==========================================================
# --- 模組層級狀態:hook / loss 必須是「可被 pickle 用名稱參照」的物件,
# 否則 ultralytics 存檔時 pickle 整個模型會炸(閉包無法序列化)。
# 老師模型與特徵暫存放在這裡,不會被一起寫進 .pt 檔。
_KD_STUDENT_FEATS = {}
_KD_TEACHER_FEATS = {}
_KD = {"orig_loss": None,"model": None,"teacher": None,"alpha": 1.0}


def _kd_student_pre_hook(module, inp):
    for i, f in enumerate(inp[0]): # inp[0] = [P3, P4, P5] 特徵
        _KD_STUDENT_FEATS[i] = f


def _kd_teacher_pre_hook(module, inp):
    for i, f in enumerate(inp[0]):
        _KD_TEACHER_FEATS[i] = f


def _kd_loss(batch, preds=None):
    base, items = _KD["orig_loss"](batch, preds) # 學生前傳 → 填 _KD_STUDENT_FEATS
    img = batch["img"]
    teacher = _KD["teacher"]
    if next(teacher.parameters()).device != img.device:
        teacher.to(img.device)
    with torch.no_grad():
        teacher(img) # 老師前傳 → 填 _KD_TEACHER_FEATS
    kd = img.new_tensor(0.0)
    adapters = _KD["model"].kd_adapters
    for i in sorted(_KD_STUDENT_FEATS.keys()):
        s = adapters[i](_KD_STUDENT_FEATS[i])
        kd = kd + F.mse_loss(s, _KD_TEACHER_FEATS[i].detach())
    return base + _KD["alpha"] * kd, items


def build_distill_trainer(teacher_path, alpha):
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.modules import Detect

    class DistillTrainer(DetectionTrainer):
        TEACHER_PATH = teacher_path
        ALPHA = float(alpha)

        def setup_model(self):
            ckpt = super().setup_model() # 此時 self.model 還在 CPU

            # 載入老師(凍結、eval)
            teacher = YOLO(self.TEACHER_PATH).model.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

            # 在 Detect 頭掛 forward_pre_hook,攔截送進去的金字塔特徵
            _KD_STUDENT_FEATS.clear()
            _KD_TEACHER_FEATS.clear()
            for m in self.model.modules():
                if isinstance(m, Detect):
                    m.register_forward_pre_hook(_kd_student_pre_hook)
                    break
            for m in teacher.modules():
                if isinstance(m, Detect):
                    m.register_forward_pre_hook(_kd_teacher_pre_hook)
                    break

            # 用一次假前傳取得各層通道數,建立 1x1 conv adapter
            imgsz = self.args.imgsz if isinstance(self.args.imgsz, int) else self.args.imgsz[0]
            dummy = torch.zeros(1, 3, imgsz, imgsz)
            with torch.no_grad():
                self.model(dummy)
                teacher(dummy)
            adapters = []
            for i in sorted(_KD_STUDENT_FEATS.keys()):
                cs = _KD_STUDENT_FEATS[i].shape[1]
                ct = _KD_TEACHER_FEATS[i].shape[1]
                adapters.append(nn.Conv2d(cs, ct, 1))
            # 掛在 self.model 底下,稍後 build_optimizer 會一起優化它
            self.model.kd_adapters = nn.ModuleList(adapters)

            # 設定模組層級狀態,並把 model.loss 換成可序列化的模組函式
            _KD["orig_loss"] = self.model.loss
            _KD["model"] = self.model
            _KD["teacher"] = teacher
            _KD["alpha"] = self.ALPHA
            self.model.loss = _kd_loss
            return ckpt

    return DistillTrainer


def run_feature_distill(data_yaml, teacher_path, student_base, alpha,
                        epochs, imgsz, batch, out_name, progress=gr.Progress()):
    logs = []

    def log(m):
        logs.append(m)
        return "\n".join(logs)

    try:
        if not data_yaml or not os.path.exists(data_yaml):
            yield log("找不到 data.yaml。請填已標註資料集的 data.yaml 路徑,"
                      "或先用上面的「雙老師偽標籤蒸餾」產生一份。")
            return
        if not teacher_path or "找不到" in str(teacher_path):
            yield log("請選擇老師模型。")
            return
        out_name = (out_name or "student_feat_distilled").strip().replace("","_")

        progress(0.05, desc="準備自訂蒸餾訓練器...")
        yield log("建立特徵蒸餾訓練器(實驗性)...")
        Trainer = build_distill_trainer(teacher_path, alpha)

        yield log(f"開始特徵蒸餾:老師={teacher_path} → 學生={student_base}\n"
                  f"(alpha={alpha}, epochs={epochs}, imgsz={imgsz})")
        student = YOLO(student_base)
        student.train(
            trainer=Trainer,
            data=data_yaml, epochs=int(epochs), imgsz=int(imgsz),
            batch=int(batch), device=device_str(),
            amp=False, # 特徵蒸餾關閉混合精度,避免 adapter/老師的 half/float 型別衝突
            workers=0, # Windows 上避開多進程 dataloader 的 spawn pickle 問題
            project=os.path.join(DISTILL_ROOT, out_name), name="train",
            exist_ok=True, verbose=False,
        )

        best = str(student.trainer.best) # ultralytics 算好的 best.pt 真實路徑
        if os.path.exists(best):
            final = os.path.join("models", f"{out_name}.pt")
            shutil.copy(best, final)
            send_alert("特徵蒸餾完成", f"學生模型已輸出:{final}")
            progress(1.0, desc="完成")
            yield log(f"特徵蒸餾完成!學生模型已存到 {final}")
        else:
            yield log("訓練跑完但找不到 best.pt,請檢查 runs_distill 資料夾。")

    except Exception as e:
        yield log(f"發生錯誤(特徵蒸餾屬實驗性,若卡住可改用偽標籤蒸餾):"
                  f"{e}\n{traceback.format_exc()}")


# ==========================================================
# UI
# ==========================================================
def render_ui(global_api, global_ws, global_proj):
    if torch.cuda.is_available():
        dev = f"運算核心狀態: {torch.cuda.get_device_name(0)} (CUDA 加速啟用中)"
    else:
        dev ="運算核心狀態: CPU(蒸餾訓練建議用 NVIDIA GPU,否則會很慢)"
    gr.Markdown(f"### {dev}")
    gr.Markdown("## 實驗室 V7:YOLO 知識蒸餾 (Knowledge Distillation)")
    gr.Markdown("把又大又準的**老師模型**的知識,壓縮進又小又快的**學生模型**(預設 nano),"
                "適合部署到火場邊緣裝置做即時推論。")

    with gr.Tabs():
        # ---------- Mode 1：雙老師偽標籤蒸餾 ----------
        with gr.Tab("① 雙老師偽標籤蒸餾(推薦・穩)"):
            gr.Markdown(
                "**PPE 老師**(你們訓練的 `weights.pt`)負責標 Gloves / 消防帽等 PPE 類,"
                "**官方老師**(COCO `yolo11x`)負責標 person(通常比自訓模型準很多)。\n\n"
                "兩者各標各的、合併成同一份標籤,**固定使用你們的 5 類 schema**"
                "(不會被官方老師的 COCO 80 類汙染)。可先按「預覽」看老師標得對不對,再開始蒸餾。")
            with gr.Row():
                with gr.Column(scale=1):
                    p_imgs = gr.File(label="上傳未標註圖片(可多選)",
                                     file_count="multiple", file_types=["image"])
                    p_receive = gr.Button("接收抽幀圖片(來自第二頁 / extracted_frames)",
                                          variant="secondary")
                    with gr.Row():
                        p_ppe = gr.Dropdown(choices=ppe_teacher_choices(),
                                            value=default_ppe_teacher(),
                                            label="PPE 老師(你們的 weights.pt)",
                                            interactive=True)
                        p_person = gr.Dropdown(choices=person_teacher_choices(),
                                               value=PERSON_TEACHERS[0],
                                               label="person 老師(官方)",
                                               interactive=True)
                        p_refresh = gr.Button("", scale=0)
                    with gr.Row():
                        p_ppe_conf = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                               label="PPE 老師信心門檻")
                        p_person_conf = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                                  label="person 老師信心門檻")
                    p_student = gr.Dropdown(choices=student_choices(), value="yolo11n.pt",
                                            label="學生模型(底模)", interactive=True)
                    with gr.Row():
                        p_epochs = gr.Number(value=50, label="訓練輪數 epochs", precision=0)
                        p_imgsz = gr.Number(value=640, label="影像尺寸 imgsz", precision=0)
                        p_batch = gr.Number(value=16, label="批次 batch", precision=0)
                    p_name = gr.Textbox(value="student_distilled",
                                        label="輸出模型名稱(存到 models/<名稱>.pt)")
                    with gr.Row():
                        p_preview = gr.Button("預覽老師標註(不訓練)", variant="secondary")
                        p_run = gr.Button("開始雙老師蒸餾(標註+訓練)", variant="primary")
                    with gr.Row():
                        p_rf_upload = gr.Button("把這批預覽圖上傳到 Roboflow", variant="secondary")
                    p_rf_status = gr.Textbox(label="Roboflow 上傳狀態", interactive=False)
                with gr.Column(scale=1):
                    p_gallery = gr.Gallery(label="老師標註預覽(肉眼檢查老師成果)",
                                           columns=3, height=320, allow_preview=True)
                    p_log = gr.Textbox(label="進度報告", lines=14, max_lines=14)

            p_upload_state = gr.State([])

            def _refresh_m1():
                return (gr.update(choices=ppe_teacher_choices(), value=default_ppe_teacher()),
                        gr.update(choices=person_teacher_choices()),
                        gr.update(choices=student_choices()))

            p_refresh.click(fn=_refresh_m1, inputs=None,
                            outputs=[p_ppe, p_person, p_student])
            p_receive.click(fn=receive_from_extractor, inputs=None, outputs=[p_imgs])
            p_preview.click(fn=preview_labels,
                            inputs=[p_imgs, p_ppe, p_person, p_ppe_conf, p_person_conf],
                            outputs=[p_log, p_gallery, p_upload_state])
            p_rf_upload.click(fn=upload_to_roboflow,
                              inputs=[p_upload_state, global_api, global_ws, global_proj],
                              outputs=[p_rf_status])
            p_run.click(fn=run_dual_distill,
                        inputs=[p_imgs, p_ppe, p_person, p_ppe_conf, p_person_conf,
                                p_student, p_epochs, p_imgsz, p_batch, p_name],
                        outputs=[p_log, p_gallery])

        # ---------- Mode 2：特徵蒸餾 ----------
        with gr.Tab("② 特徵蒸餾(實驗性・強)"):
            gr.Markdown("**實驗性功能**:學生在訓練時直接模仿老師的中間特徵圖(MSE)。"
                        "精度通常更高,但需要**已標註好的資料集**(data.yaml),"
                        "且屬自訂訓練器,若報錯請改用偽標籤蒸餾。\n\n"
                        "小撇步:可以直接填上面雙老師偽標籤蒸餾自動產生的"
                        "`runs_distill/<名稱>/data.yaml`。")
            with gr.Row():
                with gr.Column(scale=1):
                    f_yaml = gr.Textbox(label="已標註資料集的 data.yaml 路徑",
                                        placeholder="例如 runs_distill/student_distilled/data.yaml")
                    with gr.Row():
                        f_teacher = gr.Dropdown(choices=teacher_choices(),
                                                value=teacher_choices()[0],
                                                label="老師模型", interactive=True)
                        f_student = gr.Dropdown(choices=student_choices(),
                                                value="yolo11n.pt",
                                                label="學生模型(底模)", interactive=True)
                        f_refresh = gr.Button("", scale=0)
                    f_alpha = gr.Slider(0.0, 5.0, value=1.0, step=0.1,
                                        label="蒸餾權重 alpha(特徵 MSE 的比重)")
                    with gr.Row():
                        f_epochs = gr.Number(value=50, label="訓練輪數 epochs", precision=0)
                        f_imgsz = gr.Number(value=640, label="影像尺寸 imgsz", precision=0)
                        f_batch = gr.Number(value=16, label="批次 batch", precision=0)
                    f_name = gr.Textbox(value="student_feat_distilled",
                                        label="輸出模型名稱(存到 models/<名稱>.pt)")
                    f_run = gr.Button("開始特徵蒸餾", variant="primary")
                with gr.Column(scale=1):
                    f_log = gr.Textbox(label="進度報告", lines=22, max_lines=22)

            f_refresh.click(
                fn=lambda: (gr.update(choices=teacher_choices()),
                            gr.update(choices=student_choices())),
                inputs=None, outputs=[f_teacher, f_student])
            f_run.click(fn=run_feature_distill,
                        inputs=[f_yaml, f_teacher, f_student, f_alpha,
                                f_epochs, f_imgsz, f_batch, f_name],
                        outputs=[f_log])
