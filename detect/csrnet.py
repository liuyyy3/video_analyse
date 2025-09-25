# encoding: utf-8
# @File  : csrnet.py
# @Author: Xinghui
# @Date  : 2025/09/11/09:57

import cv2
import numpy as np
from rknn.api import RKNN
import matplotlib.pyplot as plt
import time

RKNN_MODEL = '/home/tom/activityMonitor/video_analyse/detect/model/CSRNet_2_All.rknn'
IMG_PATH = '/home/tom/model/IMG_crowd/crowd_3.png'
IMG_SIZE = (640, 480)  # W, H


def csrnet_load_model(rknn_path: str):
    """
    返回已初始化好的 rknn 对象。
    """
    rknn = RKNN()
    print("[CSRNet] loading:", rknn_path)
    rknn.load_rknn(rknn_path)
    rknn.init_runtime(target='rk3588')
    print("[CSRNet] ready.")
    return rknn

def csrnet_infer_frame(rknn: RKNN, frame_bgr, input_size=(640, 480)):
    """
    对单帧做推理。
    返回: (count_int, density_map(h,w,float32), infer_ms, vis_input_bgr)
    - vis_input_bgr: 实际送入模型的那张 BGR（已resize），用于保存“原帧”
    """
    # 预处理：按你原脚本的流程来（BGR->RGB/归一化/NCHW）
    W, H = input_size
    img_bgr = cv2.resize(frame_bgr, (W, H))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (img_rgb - mean) / std
    x = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)  # [1,3,H,W]

    import time
    t0 = time.time()
    outputs = rknn.inference(inputs=[x], data_format='nchw')
    infer_ms = int((time.time() - t0) * 1000)

    # 假设输出是 [1,1,h,w]
    dm = outputs[0][0, 0].astype(np.float32)
    count = float(dm.sum())
    return int(count + 0.5), dm, infer_ms, img_bgr


def main():
    # 1. Load and resize image (直接拉伸，无padding)
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        print('Error: Image not found!')
        return

    img_bgr = cv2.resize(img_bgr, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 2. 归一化（与训练一致）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_input = (img_rgb - mean) / std
    img_input = np.transpose(img_input, (2, 0, 1))  # [C, H, W]
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)  # [1, 3, H, W]

    # 3. 初始化 RKNN
    rknn = RKNN()
    rknn.load_rknn(RKNN_MODEL)
    rknn.init_runtime(target='rk3588')

    # 4. 推理
    start = time.time()
    outputs = rknn.inference(inputs=[img_input], data_format='nchw')
    if outputs is None:
        print("Error: RKNN inference failed.")
        return
    density_map = outputs[0][0, 0]  # [H, W]
    count = np.sum(density_map)
    end = time.time()
    print(f"RKNN 推理耗时：{(end - start)*1000:.2f} ms")

    # 5. 显示结果
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f"Input Image (Estimated Count: {count:.1f})")
    axs[0].axis('off')

    im = axs[1].imshow(density_map, cmap='jet')
    axs[1].set_title("Predicted Density Map")
    axs[1].axis('off')

    plt.colorbar(im, ax=axs[1])
    plt.tight_layout()
    plt.show()

    rknn.release()

if __name__ == '__main__':
    main()
