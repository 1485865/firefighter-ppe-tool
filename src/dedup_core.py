"""去重演算法共用底層。

兩套演算法都保留,給不同時機用:
  A. hist_similarity — 灰階直方圖相關性,「快」。給抽幀串流中即時去重。
  B. feature_dedup — MobileNetV2 特徵向量 + 餘弦相似度,「準」。給事後精煉。

MobileNet 改成「第一次用到才載入」(延遲),避免每次開機都吃記憶體
(就算使用者根本沒用到特徵去重)。
"""
import cv2
import torch
import torch.nn.functional as F


# ==========================================================
# 演算法 A:灰階直方圖相關性(快)
# ==========================================================
def hist_similarity(img1, img2):
    """回傳 [-1, 1] 的直方圖相關性,越接近 1 越像。"""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h1 = cv2.calcHist([g1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([g2], [0], None, [256], [0, 256])
    cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


# ==========================================================
# 演算法 B:MobileNetV2 特徵向量 + 餘弦相似度(準)
# ==========================================================
_feat_model = None
_feat_preprocess = None


def _ensure_feature_model():
    """延遲載入 MobileNetV2(eval 模式,有 GPU 就上 GPU)。"""
    global _feat_model, _feat_preprocess
    if _feat_model is None:
        import torchvision.models as models
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        _feat_preprocess = weights.transforms()
        m = models.mobilenet_v2(weights=weights).eval()
        if torch.cuda.is_available():
            m = m.cuda()
        _feat_model = m
    return _feat_model, _feat_preprocess


def extract_feature(img_path):
    """讀一張圖 → 正規化後的特徵向量;失敗回傳 None。"""
    from PIL import Image
    model, preprocess = _ensure_feature_model()
    try:
        img = Image.open(img_path).convert('RGB')
        tensor = preprocess(img).unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        with torch.no_grad():
            feat = model(tensor)
        return F.normalize(feat, p=2, dim=1).squeeze(0)
    except Exception:
        return None


def feature_dedup(img_paths, threshold_pct, progress=None):
    """用特徵向量做事後去重。

    threshold_pct: 0~100,越高越嚴格(越容易判定為不同圖)。
    回傳 (unique_paths, removed_count)。

    比對用一次矩陣乘法算出「這張圖跟目前所有已保留圖片」的相似度,
    而不是逐一 for 迴圈 torch.dot().item()——後者每比對一對就要同步一次
    GPU,對幾百張圖的批次而言,同步次數是 O(n^2),換成矩陣乘法後
    每張新圖只需要一次同步,批次越大差距越明顯。
    """
    unique = []
    kept_stack = None  # (N, D),L2 正規化過的已保留圖片特徵向量
    total = len(img_paths)
    threshold = threshold_pct / 100.0
    for i, p in enumerate(img_paths):
        if progress is not None:
            progress((i + 1) / total, desc=f"提取特徵向量中... ({i + 1}/{total})")
        feat = extract_feature(p)
        if feat is None:
            continue
        is_dup = False
        if kept_stack is not None:
            sims = kept_stack @ feat
            if torch.max(sims).item() > threshold:
                is_dup = True
        if not is_dup:
            unique.append(p)
            row = feat.unsqueeze(0)
            kept_stack = row if kept_stack is None else torch.cat([kept_stack, row], dim=0)
    return unique, total - len(unique)
