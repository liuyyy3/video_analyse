# encoding: utf-8
# @File  : Image_diff_complete.py
# @Author: Xinghui
# @Date  : 2025/09/09/10:42

# 更适用于车厢环境的遗留物检测，先收集一张空的、没有遗留物的车厢照片（图一），作为基础背景
# 另外等到终点站乘客下车后，启动程序，抓取一张当前的“空车厢图片”（图二）
# 讲图二于图一做差，多出来的人或物框选起来送入 YOLO
# 此为精简输出的完整版程序，适配 RK3588板卡上用 NPU推理的情况

import os
import cv2
import numpy as np
import json
import sys
import time
from rknn.api import RKNN
import argparse

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
# sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))
# 路径直接写死
sys.path.append('/home/tom/Projects/rknn_model_zoo')

from py_utils.rknn_executor import RKNN_model_container
from py_utils.coco_utils import COCO_test_helper


# 复用 yolo11.py 里的常量/后处理(包含 dfl + 无锚后处理)
#    注意：保持与你板上脚本同名模块导入路径一致；若不在 PYTHONPATH，可把 yolo11.py 放到同目录后 `import yolo11`
from yolo11 import post_process, IMG_SIZE, CLASSES  # IMG_SIZE=(640,640)


BASELINE_PATH = '/home/tom/model/leftover/carriage4.png'
CLEAN_PATH = '/home/tom/model/leftover/carriage4-2.png'

ROI_JSON_PATH = '/home/tom/model/leftover/carriage4_half_door.json'

# 程序需要输出的文件夹以及文件
OUT_DIR = './debug_out'
OUT_RAW_SIDE = os.path.join(OUT_DIR, "1_raw_side_by_side.png")  # 精简版要保留
OUT_PREP_SIDE = os.path.join(OUT_DIR, "2_prep_side_by_side.png")  # 精简版要保留
OUT_CAND_OVERLAY = os.path.join(OUT_DIR, "4_cand_overlay.png")  # 候选框叠加图  # 精简版要保留
OUT_CAND_EXPAND_OVERLAY = os.path.join(OUT_DIR, "5_cand_expand_overlay.png")  # 扩展框叠加图  # 精简版要保留
OUT_YOLO_OVERLAY = os.path.join(OUT_DIR, "6_yolo_overlay.png")
CROP_DIR = os.path.join(OUT_DIR, "corps")


RKNN_MODEL_PATH = "/home/tom/model/yolo11n_relu.rknn"
rknn = RKNN_model_container(RKNN_MODEL_PATH, target='rk3588', device_id=None)

# 全局 RKNN 会话（供后面的推理函数复用）
rknn = RKNN()

# 添加调试信息，调试信息成功后将下面两行的注释删掉
# assert rknn.load_rknn(RKNN_MODEL_PATH) == 0, 'load_rknn failed'
# assert rknn.init_runtime(target='rk3588') == 0, 'init_runtime failed'


# 调试信息如下

ret = rknn.load_rknn(RKNN_MODEL_PATH)
if ret != 0:
    exit("load rknn failed")
ret = rknn.init_runtime(target='rk3588')
if ret != 0:
    exit("init runtime failed")
print("✅ NPU 初始化完成")




# 差分/清理参数 （根据分辨率进行调整 ）
ABS_THR = 29  # 绝对差阈值，数值越小越敏感（噪声也可能更多）
GAUSS_KSIZE = 5  # 高斯平滑核尺寸（奇数），减小噪声影响
MOG2_VAR_THR = 16  # MOG2阈值（越大越不敏感）
MIN_AREA = 600  # 连通域最小有效面积(px)
OPEN_K = 5  # 开运算核尺寸（去小噪点）
CLOSE_K = 7  # 闭运算核尺寸（填小空洞）

# 扩张与裁剪参数
EXPAND_RATIO = 1.1  # 扩张比例：1.0=不扩，1.3=宽高都放大30%（推荐 1.2~1.5）
MIN_CROP = 32  # 裁剪最小边（像素），太小会让YOLO看不清

YOLO_CONF_THR = 0.25

# 只允许上报的“遗留物”类别（COCO 名称；可按需增减）
ALLOWED_LABELS = {
    "backpack", "handbag", "suitcase",  # 各种包
    "cell phone",                       # 手机
    "bottle", "cup",                    # 瓶/杯
    "umbrella",                         # 雨伞
    "book", "laptop",                   # 书/电脑（可选）
    "person"                            # 赖着不走的乘客
}

# 确保输出文件夹路径存在，不存在就创建一个，
def ensure_dir (path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

# 读取图片
def imread_color(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    assert img is not None, f"读取失败: {path}"
    return img

# 缩放图片统一尺寸
def resize_to(img, target_wh):
    target_w, target_h = target_wh
    # 插值策略：放大/微调常用 INTER_LINEAR，缩小时可考虑 INTER_AREA
    inter = cv2.INTER_LINEAR if (img.shape[1] <= target_w and img.shape[0] <= target_h) else cv2.INTER_AREA
    return cv2.resize(img, (target_w, target_h), interpolation = inter)

# 抗光照处理
def preprocess_clean_bgr(img, clip_limit=2.0, tile_grid_size=(8,8)):
    # BGR -> YCrCb
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)

    # 创建 CLAHE对象并英语到亮度通道 Y
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    y_eq = clahe.apply(y)

    # 合并 转回 BGR
    ycc_eq = cv2.merge([y_eq, cr, cb])
    img_out = cv2.cvtColor(ycc_eq, cv2.COLOR_YCrCb2BGR)
    return img_out

# 将两张图片水平对齐拼接
def hstack_same_height (img_left, img_right):
    h0 = img_left.shape[0]
    # 将右图缩放到跟左图一样的尺寸
    scale = h0 / img_right.shape[0]
    new_w = max(1, int(img_right.shape[1] * scale))
    right_resized = cv2.resize(img_right, (new_w, h0), interpolation=cv2.INTER_LINEAR)
    # 水平拼接
    side = np.hstack((img_left, right_resized))
    return side

# 适用 ORB提取关键点与子描述
def orb_detect_and_describe(img_gray, nfeatures = 2000):
    orb = cv2.ORB_create(nfeatures=nfeatures)  # 创建 ORB；nfeatures 控制希望的关键点上限。
    kps, des = orb.detectAndCompute(img_gray, None)
    return kps, des

# 暴力匹配器
def match_orb_descriptors(des1, des2, keep_top = 400):
    if des1 is None or des2 is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    return matches[:keep_top]

# 把图二坐标映射到图一坐标
def estimate_homography(kps1, kps2, matches, ransac_reproj_thr=4.0, confidence = 0.99):
    if len(matches) < 8:
        return None, None  # 估计H至少需要4对，但稳妥起见<8就放弃
    # 从匹配对中取对应坐标（KeyPoint.pt 是 (x, y)）
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    # 注意参数顺序：把 pts2 映射到 pts1，所以是 (src=pts2, dst=pts1)
    H, mask = cv2.findHomography(
        pts2, pts1,  # 源点、目标点
        cv2.RANSAC,  # RANSAC方法稳健估计
        ransac_reproj_thr,  # 重投影误差阈值（像素）
        maxIters=2000,  # RANSAC最多迭代次数
        confidence=confidence  # 置信度
    )
    return H, mask

def warp_to_baseline(img_to_warp, H, out_shape_hw):
    if H is None:
        H = np.eye(3, dtype=np.float64)  # 对齐失败就不变形（继续流程，也能跑）
    Hh, Hw = out_shape_hw  # out_shape_hw 是 (高, 宽)
    warped = cv2.warpPerspective(img_to_warp, H, (Hw, Hh))  # size顺序必须 (宽, 高)
    print(f"[warp] baseline(H,W)={out_shape_hw}, warp dsize=(W,H)=({Hw},{Hh})")
    return warped

def draw_matches_image(img1, kps1, img2, kps2, matches, max_draw=80):
    draw = matches[:max_draw]
    vis = cv2.drawMatches(
        img1, kps1, img2, kps2,  # 左右图与各自的关键点
        draw, None,  # 要画的匹配，输出画布
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS  # 不画没有匹配的点
    )
    return vis

# 图片转灰度
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def absdiff_mask(imgA, imgB, thr = ABS_THR, gauss_ksize = GAUSS_KSIZE):
    if gauss_ksize and gauss_ksize >= 3:
        imgA = cv2.GaussianBlur(imgA, (gauss_ksize, gauss_ksize), 0)
        imgB = cv2.GaussianBlur(imgB, (gauss_ksize, gauss_ksize), 0)
    diff = cv2.absdiff(imgA, imgB)
    _, diff_bin = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    return diff_bin, diff

# MOG2背景建模
def mog2_foreground(img_ref_gray, img_now_gray, var_thr = MOG2_VAR_THR):
    mog = cv2.createBackgroundSubtractorMOG2(history = 50, varThreshold = var_thr, detectShadows = False)
    _ = mog.apply(img_ref_gray)    # 建背景
    fg = mog.apply(img_now_gray)   # 取前景
    _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)   # 去除小阴影
    return fg_bin

def clean_mask(mask, open_k = OPEN_K, close_k = CLOSE_K):
    # 生成结构元素
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    # 开运算 = 先腐蚀(erode)后膨胀(dilate)
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations = 1)
    # 闭运算 = 先膨胀(dilate)后腐蚀(erode)
    mask3 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, k_close, iterations = 2)
    return mask3

def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def expand_box(x, y, w, h, ratio, W, H, min_crop = MIN_CROP):
    # 计算中心点
    cx = x + w / 2.0
    cy = y + h / 2.0
    # 扩张后的的宽高
    new_w, new_h = w * ratio, h * ratio
    # 确保最小尺寸
    new_w = max(new_w, min_crop)
    new_h = max(new_h, min_crop)
    # 根据中心点推理四个坐标
    x1 = int(round(cx - new_w / 2.0))
    y1 = int(round(cy - new_h / 2.0))
    x2 = int(round(cx + new_w / 2.0))
    y2 = int(round(cy + new_h / 2.0))
    # 夹紧到图像边缘
    x1 = clamp(x1, 0, W - 1)
    y1 = clamp(y1, 0, H - 1)
    x2 = clamp(x2, 0, W - 1)
    y2 = clamp(y2, 0, H - 1)
    # 防止夹紧后越界
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)

    return x1, y1, x2, y2

def crop_by_box(img, box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return img[y1:y2, x1:x2].copy()

# 安全拿类别名
def get_class_names(model, results = None):
    names = None
    if results is not None and hasattr(results, "names"):
        names = results.names
    if names is None and hasattr(model, "names"):
        names = model.names
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    if not isinstance(names, dict):
        names = {i: str(i) for i in range(100)}
    return names

def _pts_int32(pts):
    return np.array(pts,dtype=np.int32)

def _fill_rect(mask, p1, p2, val=255):
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    mask[y1:y2, x1:x2] = val

def mask_from_labelme(json_path, img_shape, label_name):
    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    want = label_name.strip().lower()
    for shp in data.get("shapes", []):
        lab = shp.get("label","").strip().lower()
        if lab != want:
            continue
        pts = shp.get("points", [])
        st = shp.get("shape_type", "polygon")
        if st == "polygon" and len(pts) >= 3:
            cv2.fillPoly(mask, [_pts_int32(pts)], 255)
        elif st == "rectangle" and len(pts) >= 2:
            _fill_rect(mask, pts[0], pts[1], 255)
    return mask

def build_final_mask(img_shape, json_path, roi_label="ROI", ignore_label = "ignore"):
    roi_mask = mask_from_labelme(json_path, img_shape, roi_label)
    ign_mask = mask_from_labelme(json_path, img_shape, ignore_label)

    if roi_mask.any():
        # final = ROI AND (NOT ignore): 只再 ROI内做检测，并且抠掉 ignore区域的不进行检测
        final_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(ign_mask))
    else:
        # 没有画出 ROI的话，就用全图来剪掉 ignore区域
        H, W = img_shape[:2]
        full = np.full((H, W), 255, np.uint8)
        final_mask = cv2.bitwise_and(full, cv2.bitwise_not(ign_mask))
    return roi_mask, ign_mask, final_mask

def box_mask_ratio(box_xyxy, mask255):
    x1, y1, x2, y2 = map(int, box_xyxy)
    H, W = mask255.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W - 1, x2)
    y2 = min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = mask255[y1:y2, x1:x2]
    return float((crop == 255).sum()) / float(crop.size)

def letterbox_640(bgr, new_shape = (640, 640), pad_color = (0, 0, 0)):
    tw, th = new_shape
    h, w = bgr.shape[:2]
    scale = min(tw / float(w), th / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation = cv2.INTER_LINEAR)
    out = np.full((th, tw, 3), pad_color, dtype=np.uint8)
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    out[dy:dy + nh, dx:dx + nw] = resized
    return out, scale, dx, dy


def rknn_yolo11_infer_640(bgr640):
    """
    传入：bgr640  (H=640, W=640, C=3, uint8)
    目标：与 yolo11.py 一致 —— NHWC、RGB、uint8、[0~255]，不做 /255 归一化
    """

    # 与官方给出的 yolo11.py完全一致的图像处理办法
    # BGR -> RGB，保持 dtype=uint8
    rgb = cv2.cvtColor(bgr640, cv2.COLOR_BGR2RGB).astype(np.uint8)

    # 直接 NHWC 喂给 RKNN；默认 data_format = 'nhwc'，保险起见我们显式写上
    # ★ 调试②：统计推理耗时（毫秒）
    t0 = time.perf_counter()
    outputs = rknn.inference(inputs = [rgb], data_format = 'nhwc')
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[RKNN][infer] time={dt_ms:.2f} ms")

    # 后处理保持与 yolo11.py 一致
    boxes, classes, scores = post_process(outputs)

    if boxes is None:
        return []

    dets = []
    for(x1, y1, x2, y2), cls_id, sc in zip(boxes, classes, scores):
        label = CLASSES[int(cls_id)] if int(cls_id) < len(CLASSES) else str(int(cls_id))
        label = label.strip()
        dets.append((
            label, float(sc), (float(x1), float(y1), float(x2), float(y2))
        ))
    return dets


def yolo_on_crop_fixed640_RKNN(crop_bgr, crop_xyxy_in_full, conf_thr = 0.3, allowed = ALLOWED_LABELS):
    # 【可选】整个函数计时（如果你想看）
    t0_total = time.perf_counter()

    canvas, scale, dx, dy = letterbox_640(crop_bgr, (IMG_SIZE[0], IMG_SIZE[1]))

    # ★ 调试①：打印送入 YOLO 的图片尺寸（期望 (640, 640, 3)）
    print(f"[DEBUG] YOLO input canvas.shape={canvas.shape}, dtype={canvas.dtype}")

    dets = rknn_yolo11_infer_640(canvas)

    best = None
    for label, conf, (x1, y1, x2, y2) in dets:

        if conf < conf_thr:
            continue
        if label not in allowed:
            continue

        x1c = (x1 - dx) / max(scale, 1e-6)
        y1c = (y1 - dy) / max(scale, 1e-6)
        x2c = (x2 - dx) / max(scale, 1e-6)
        y2c = (y2 - dy) / max(scale, 1e-6)

        Hc, Wc = crop_bgr.shape[:2]
        x1c = float(np.clip(x1c, 0, Wc - 1))
        y1c = float(np.clip(y1c, 0, Hc - 1))
        x2c = float(np.clip(x2c, 0, Wc - 1))
        y2c = float(np.clip(y2c, 0, Hc - 1))

        if (best is None) or (conf > best[1]):
            best = (label, conf, (x1c, y1c, x2c, y2c))

    # 【可选】如果还想看这个函数总耗时（不是必须）
    total_ms = (time.perf_counter() - t0_total) * 1000.0
    print(f"[DEBUG] yolo_on_crop_fixed640_RKNN total={total_ms:.2f} ms")

    if best is None:
        return None

    off_x, off_y = crop_xyxy_in_full[0], crop_xyxy_in_full[1]
    gx1 = int(round(best[2][0] + off_x))
    gy1 = int(round(best[2][1] + off_y))
    gx2 = int(round(best[2][2] + off_x))
    gy2 = int(round(best[2][3] + off_y))
    return {
        "label": best[0],
        "conf": best[1],
        "bbox_xyxy": [gx1, gy1, gx2, gy2],
        "bbox_xyxy_crop":[float(v) for v in best[2]]
    }


def log_one_result(idx, crop_path, det, fallback_xyxy):
    """
    控制台打印单个候选的检测结果。
    参数：
      idx            : 候选编号（从0开始）
      crop_path      : 裁剪小图的保存路径
      det            : yolo_on_crop_fixed640_RKNN() 的返回字典；None 表示未识别到
      fallback_xyxy  : 这个候选在整图上的原始框 (x1,y1,x2,y2)，用于 Unknown 时也能打印坐标
    """
    if det is None:
        print(f"[{idx:02d}] Unknown | box={fallback_xyxy} | crop='{crop_path}'")
        return

    label = det["label"]
    conf  = det["conf"]
    box_g = det["bbox_xyxy"]        # 全图上的框
    box_c = det.get("bbox_xyxy_crop", None)  # 裁剪图上的框（可选）
    print(f"[{idx:02d}] {label:<12} conf={conf:.3f} | box_full={box_g} | box_crop={box_c} | crop='{crop_path}'")




def main():
    ensure_dir(OUT_DIR)

    # 记录程序开始时间
    global_start = time.perf_counter()

    # 先读图，统一尺寸，水平拼接
    print("读图 + 统一尺寸 + 水平拼接")
    img0 = imread_color(BASELINE_PATH)
    img1 = imread_color(CLEAN_PATH)
    # 以图一为尺寸标准，将图二和以后的图片都强制缩放到这个尺寸
    h0, w0 = img0.shape[:2]
    img1_resized = resize_to(img1, (w0, h0))

    # 并排可视化，将两张图片水平拼接
    side_raw = np.hstack((img0, img1_resized))
    cv2.imwrite(OUT_RAW_SIDE, side_raw)

    print(f" -原图尺寸: baseline={img0.shape}, current(resized) = {img1_resized.shape}")
    print(f" -已输出: {OUT_RAW_SIDE}")

    print("抗光预处理 + 处理后并排保存")
    img0_p = preprocess_clean_bgr(img0, clip_limit=2.0, tile_grid_size=(8, 8))
    img1_p = preprocess_clean_bgr(img1_resized, clip_limit=2.0, tile_grid_size=(8, 8))

    # 并排可视化
    side_prep = hstack_same_height(img0_p, img1_p)
    cv2.imwrite(OUT_PREP_SIDE, side_prep)

    print(f" -已输出: {OUT_PREP_SIDE}")

    print("轻量配准 ORB + RANSAC H")

    # 转灰度
    g0 = cv2.cvtColor(img0_p, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(img1_p, cv2.COLOR_BGR2GRAY)

    # 提取 ORB关键点与描述子
    kps0, des0 = orb_detect_and_describe(g0, nfeatures=2000)
    kps1, des1 = orb_detect_and_describe(g1, nfeatures=2000)

    # 特征匹配（交叉检查）
    matches = match_orb_descriptors(des0, des1, keep_top=400)
    print(f" -匹配到的点对数量 = {len(matches)}")

    # RANSAC 估计单应性 H （把图二映射到图一）
    H, mask_inliers = estimate_homography(kps0, kps1, matches, ransac_reproj_thr=4.0, confidence=0.99)
    inliers = int(mask_inliers.sum()) if mask_inliers is not None else 0
    print(f" - RANSAC 内点数={inliers} / {len(matches)}")
    print(f" - H = \n{H if H is not None else np.eye(3)}")

    # 用 H将图二对齐到图一的坐标系
    print(f"[shape] img0_p(H,W)={img0_p.shape[:2]}, img1_p(H,W)={img1_p.shape[:2]}")

    img1_aligned = warp_to_baseline(img1_p, H, out_shape_hw=(h0, w0))  # 注意参数(out_shape_hw)是(高,宽)
    img1_raw_aligned = warp_to_baseline(img1_resized, H, out_shape_hw=(h0, w0))  # 将未作增强的待检测图映射到基图坐标
    print(f"[shape] img1_aligned(H,W)={img1_aligned.shape[:2]}")
    print(f"[shape] img1_raw_aligned(H,W)={img1_raw_aligned.shape[:2]}")

    print(" 4: 差分与二值化")
    # 取对齐后的图二与图一，转灰度,差分只需要强度信息
    g0p = to_gray(img0_p)
    g1a = to_gray(img1_aligned)

    # 绝对差 + 阈值（二值化），快速且直观
    diff_bin, diff_gray = absdiff_mask(g0p, g1a, thr=ABS_THR, gauss_ksize=GAUSS_KSIZE)

    # 再用 MOG2 做一份前景掩码（鲁棒对光照/小抖动），作为第二重筛选
    fg_mog2 = mog2_foreground(g0p, g1a, var_thr=MOG2_VAR_THR)

    # 两者做“与”运算，只保留共同认可的变化区域（减少误报）
    mask_and = cv2.bitwise_and(diff_bin, fg_mog2)

    # 形态学清理：去除小碎点、补上小空洞
    mask_clean = clean_mask(mask_and, open_k=OPEN_K, close_k=CLOSE_K)
    # 读取 JSON文件并套用 final_mask。从 JSON 生成 ROI/ignore/final 掩膜（注意尺寸用的是“用于检测的图”，也就是对齐后的 img1_aligned）
    roi_mask, ign_mask, final_mask = build_final_mask(img1_aligned.shape, ROI_JSON_PATH,
                                                      roi_label="ROI", ignore_label="ignore")

    # 将 final_mask 与“差分清理后的 mask”再做一次 AND
    mask_clean = cv2.bitwise_and(mask_clean, final_mask)

    # 提取连通域做候选框（下一步会丢给YOLO；这一步先把候选框画出来看效果）。连通域 → 候选框（记得用正确的 findContours）
    cnts_info = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
    # overlay = img1_aligned.copy()
    overlay = img1_raw_aligned.copy()
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area <MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # 算出该框在 roi_mask 里的覆盖率 roi_ratio（在白色区域的像素比例）。框级别二次过滤（强烈建议启用）
        roi_ratio = box_mask_ratio((x, y, x+w, y+h), roi_mask)
        ign_ratio = box_mask_ratio((x, y, x+w, y+h), ign_mask)

        if roi_ratio < 0.6 or ign_ratio > 0.2:
            continue

        candidates.append((x, y, w, h))
        # 画候选框以便送进yolo做识别，蓝色画框
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 102, 0), 2)
    cv2.imwrite(OUT_CAND_OVERLAY, overlay)

    print(f" - 候选数量: {len(candidates)}")
    print(f" - 输出: {OUT_CAND_OVERLAY}")

    print(" 5 候选扩张 + 裁剪")
    if not os.path.isdir(CROP_DIR):
        os.makedirs(CROP_DIR, exist_ok=True)
    Hh, Ww = img1_aligned.shape[:2]

    # 针对 PHASE 4 产出的 candidates 列表（里面是 (x,y,w,h)）
    expanded_boxes = []  # 保存扩张后的框 (x1,y1,x2,y2)
    crop_paths = []  # 保存每个裁剪小图的路径

    # 遍历每一个候选框，进行扩张、裁剪与可视化
    # overlay5 = img1_aligned.copy()
    overlay5 = img1_raw_aligned.copy()
    for idx, (x, y, w, h) in enumerate(candidates):
        x1, y1, x2, y2 = expand_box(
            x, y, w, h,
            ratio=EXPAND_RATIO,
            W=Ww, H=Hh,
            min_crop=MIN_CROP
        )
        expanded_boxes.append((x1, y1, x2, y2))

        # 裁剪该区域的小图，存盘，供你肉眼检查 & 供下一步 YOLO 使用
        crop = crop_by_box(img1_raw_aligned, (x1, y1, x2, y2))
        crop_name = f"crop_{idx:02d}.png"
        crop_path = os.path.join(CROP_DIR, crop_name)
        cv2.imwrite(crop_path, crop)
        crop_paths.append(crop_path)

        # 可视化：在整图上画扩张后的框，并标一个序号
        cv2.rectangle(overlay5, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            overlay5, f"id = {idx}", (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
        )

    # 保存扩张后的候选叠加图
    cv2.imwrite(OUT_CAND_EXPAND_OVERLAY, overlay5)

    # 把候选信息做成一个json，供后续使用
    cand_records = []
    for idx, ((x, y, w, h), (x1, y1, x2, y2), cpath) in enumerate(zip(candidates, expanded_boxes, crop_paths)):
        cand_records.append({
            "id": idx,
            "orig_bbox_xywh": [int(x), int(y), int(w), int(h)],
            "expanded_bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "crop_path": os.path.abspath(cpath)

        })

    print(f" - 扩张候选数: {len(expanded_boxes)}")
    print(f" - 裁剪目录: {CROP_DIR}")
    print(f" - 叠加图: {OUT_CAND_EXPAND_OVERLAY}")


    # model = load_yolo_model(YOLO_MODEL_PATH)
    # model = rknn_lite.load_rknn(RKNN_MODEL_PATH)
    # 在整图上准备一张叠加画布 overlay6
    # overlay6 = img1_aligned.copy()
    overlay6 = img1_raw_aligned.copy()
    det_results = []
    recognized_cnt = 0
    unknown_cnt =0

    for i, (x1, y1, x2, y2) in enumerate(expanded_boxes):
        crop_bgr = crop_by_box(img1_raw_aligned, (x1, y1, x2, y2))
        det = yolo_on_crop_fixed640_RKNN(crop_bgr, (x1, y1, x2, y2), conf_thr=YOLO_CONF_THR, allowed=ALLOWED_LABELS)

        cp = crop_paths[i] if i < len(crop_paths) else ""
        log_one_result(i, cp, det, (x1, y1, x2, y2))

        if det is None:
            unknown_cnt += 1
            cv2.rectangle(overlay6, (x1, y1), (x2, y2), (77, 166, 255), 2)
            cv2.putText(overlay6, "Unknown", (x1, max(0, y1 - 6)), 0, 0.6, (77, 166, 255), 2)
            continue

        recognized_cnt += 1
        det_results.append(det)
        gx1, gy1, gx2, gy2 = det["bbox_xyxy"]
        label, conf = det["label"], det["conf"]
        cv2.rectangle(overlay6, (gx1, gy1), (gx2, gy2), (255, 26, 209), 2)  # 紫色
        cv2.putText(overlay6, f"{label} {conf:.2f}", (gx1, max(0, gy1 - 6)), 0, 0.6, (255, 26, 209), 2)

    # 保存叠加可视化图片
    cv2.imwrite(OUT_YOLO_OVERLAY, overlay6)
    # 上报 json图
    payload = {
        "baseline_image": os.path.abspath(BASELINE_PATH),
        "current_image": os.path.abspath(CLEAN_PATH),
    }

    print(f" - 输出叠加图: {OUT_YOLO_OVERLAY}")

    # 记录程序整体结束时间并计算总耗时
    total_ms = (time.perf_counter() - global_start) * 1000.0
    print(f"[TOTAL] 整个流程总耗时: {total_ms:.2f} ms")

    # —— 新增：控制台统计小结 ——
    total_cand = len(expanded_boxes)
    print(f"[SUMMARY] candidates={total_cand}, recognized={recognized_cnt}, unknown={unknown_cnt}")

if __name__ =="__main__":
    main()

