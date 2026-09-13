"""Data Scaling Curve:同一份資料集用不同 fraction(25%/50%/75%/100%)各訓練
一輪,量化「資料量 vs 模型效能」的邊際效益,並線性外推估算還需要多少資料。

訓練環境固定 workers=0(先前遇過 pin memory 的 CUDA 錯誤)。
支援中斷後續跑:每個 fraction 的結果快取進 temp/scaling_curve/<run_name>/manifest.json,
key 用 (data_yaml, model_base, epochs, imgsz, batch, fraction) 的雜湊,參數不變就跳過重跑。
"""
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

TEMP_DIR = os.path.join(os.getcwd(), "temp", "scaling_curve")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _manifest_path(run_dir):
    return os.path.join(run_dir, "manifest.json")


def _load_manifest(run_dir):
    path = _manifest_path(run_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(run_dir, manifest):
    os.makedirs(run_dir, exist_ok=True)
    with open(_manifest_path(run_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _cache_key(data_yaml, model_base, epochs, imgsz, batch, fraction):
    raw = f"{os.path.abspath(data_yaml)}|{model_base}|{epochs}|{imgsz}|{batch}|{fraction}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cuda_available():
    import torch
    return torch.cuda.is_available()


def estimate_train_image_count(data_yaml):
    """粗估 train 集圖片數,供外推估算換算成「還需要多少張圖」。抓不到就回傳 None。"""
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        base = data.get("path") or os.path.dirname(os.path.abspath(data_yaml))
        train_rel = data.get("train", "images/train")
        train_dir = train_rel if os.path.isabs(train_rel) else os.path.join(base, train_rel)
        if os.path.isdir(train_dir):
            return sum(1 for f in os.listdir(train_dir) if f.lower().endswith(IMG_EXTS))
    except Exception:
        return None
    return None


def run_one_fraction(data_yaml, model_base, fraction, epochs, imgsz, batch,
                      run_dir, epoch_progress_cb=None):
    """跑一輪指定 fraction 的訓練+驗證。每個 fraction 都要全新模型物件,
    不能用 utils.get_yolo_model() 快取(訓練會直接改動權重,快取會互相汙染)。
    回傳 dict(overall_map50, per_class_map50)。"""
    student = YOLO(model_base)

    if epoch_progress_cb is not None:
        def _on_epoch_end(trainer):
            epoch_progress_cb(trainer.epoch + 1, trainer.epochs)
        student.add_callback("on_train_epoch_end", _on_epoch_end)

    name = f"frac_{int(fraction * 100)}"
    student.train(
        data=data_yaml, fraction=fraction, epochs=int(epochs), imgsz=int(imgsz),
        batch=int(batch), workers=0, device=("0" if _cuda_available() else "cpu"),
        project=run_dir, name=name, exist_ok=True, verbose=False,
    )
    metrics = student.val(data=data_yaml)

    names = student.names
    ap50 = metrics.box.ap50
    per_class = {names[i]: float(ap50[i]) for i in range(len(ap50))}
    overall = float(metrics.box.map50)

    return {"overall_map50": overall, "per_class_map50": per_class}


def run_scaling_curve(data_yaml, model_base, fractions, epochs, imgsz, batch,
                       run_name, progress=None):
    """generator,每次 yield (log_message, results_so_far)。

    results_so_far: [{"fraction", "overall_map50", "per_class_map50"}, ...]。
    """
    run_dir = os.path.join(TEMP_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    manifest = _load_manifest(run_dir)

    results = []
    total = len(fractions)
    for i, fraction in enumerate(fractions):
        key = _cache_key(data_yaml, model_base, epochs, imgsz, batch, fraction)
        if key in manifest:
            results.append({"fraction": fraction, **manifest[key]})
            if progress is not None:
                progress((i + 1) / total, desc=f"fraction={fraction} 已有快取,跳過")
            yield f"fraction={fraction} 已有快取結果,跳過重跑。", list(results)
            continue

        def epoch_cb(epoch, total_epochs, _i=i, _fraction=fraction):
            if progress is not None:
                frac_progress = epoch / max(1, total_epochs)
                progress((_i + frac_progress) / total,
                          desc=f"訓練 fraction={_fraction}({epoch}/{total_epochs} epoch)")

        yield f"開始訓練 fraction={fraction}...", list(results)
        try:
            metrics = run_one_fraction(data_yaml, model_base, fraction, epochs,
                                        imgsz, batch, run_dir, epoch_progress_cb=epoch_cb)
        except Exception as e:
            yield f"fraction={fraction} 訓練失敗: {e}", list(results)
            continue

        manifest[key] = metrics
        _save_manifest(run_dir, manifest)
        results.append({"fraction": fraction, **metrics})
        yield (f"fraction={fraction} 完成: overall mAP50={metrics['overall_map50']:.3f}",
               list(results))

    yield "全部 fraction 訓練完成。", list(results)


def extrapolate_target(results, target_map, total_train_images=None):
    """線性回歸 (fraction, overall_map50),估算達到 target_map 需要的 fraction。"""
    if len(results) < 2:
        return "資料點不足(至少要 2 個 fraction 才能估算),請先完成訓練。"

    fractions = np.array([r["fraction"] for r in results])
    maps = np.array([r["overall_map50"] for r in results])

    if maps.max() >= target_map:
        return f"目前資料量下已經達到目標 mAP50={target_map:.2f},不需要更多資料。"

    slope, intercept = np.polyfit(fractions, maps, 1)
    if slope <= 0:
        return "目前曲線斜率 <= 0,無法用線性外推估算(可能已經飽和或資料量太少)。"

    target_fraction = (target_map - intercept) / slope
    extra_fraction = target_fraction - 1.0

    msg = (
        f"線性回歸: mAP50 ≈ {slope:.4f} × fraction + {intercept:.4f}\n"
        f"估算達到 mAP50={target_map:.2f} 需要 fraction ≈ {target_fraction:.2f}"
        f"(相當於現有資料量的 {target_fraction * 100:.0f}%)\n"
    )
    if total_train_images:
        extra_images = max(0.0, extra_fraction) * total_train_images
        msg += f"換算約需再新增 {extra_images:.0f} 張訓練圖片。\n"
    msg += "\n⚠️ 注意:這是線性外推,實際邊際效益遞減,真正需要的資料量通常比這個估算更多。"
    return msg


HAND_CLASSES = {"Gloves", "No_Gloves"}


def build_curve_figures(results):
    """回傳 (fig_overall, fig_per_class)。results 依 fraction 由小到大排序後畫。"""
    ordered = sorted(results, key=lambda r: r["fraction"])
    fractions = [r["fraction"] for r in ordered]

    fig_overall, ax1 = plt.subplots(figsize=(6, 4))
    if ordered:
        overall = [r["overall_map50"] for r in ordered]
        ax1.plot(fractions, overall, marker="o", color="royalblue")
        ax1.set_xlabel("資料量比例(fraction)")
        ax1.set_ylabel("整體 mAP50")
        ax1.set_title("整體 mAP50 對資料量")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "尚無資料", ha="center", va="center")
        ax1.axis("off")

    fig_per_class, ax2 = plt.subplots(figsize=(6, 4))
    if ordered:
        class_names = list(ordered[0]["per_class_map50"].keys())
        for cls_name in class_names:
            ys = [r["per_class_map50"].get(cls_name, 0.0) for r in ordered]
            is_hand = cls_name in HAND_CLASSES
            ax2.plot(fractions, ys, marker="o",
                     linewidth=3 if is_hand else 1.5,
                     label=cls_name + ("(短板)" if is_hand else ""))
        ax2.set_xlabel("資料量比例(fraction)")
        ax2.set_ylabel("mAP50")
        ax2.set_title("各類別 mAP50 對資料量(Gloves/No_Gloves 加粗)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "尚無資料", ha="center", va="center")
        ax2.axis("off")

    return fig_overall, fig_per_class
