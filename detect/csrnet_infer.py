# encoding: utf-8
# @File  : csrnet_infer.py
# @Author: Xinghui
# @Date  : 2025/09/11/09:57

# 人员密度检测

import os
import cv2
import time
import numpy as np
from datetime import datetime
from .check_rtsp import check_rtsp_stream
from .csrnet import csrnet_load_model, csrnet_infer_frame

# 固定模型路径（写死）
RKNN_MODEL = "/home/tom/activityMonitor/video_analyse/detect/model/CSRNet_2_All.rknn"
SAVE_DIR = "/home/tom/Flask/crowd"  # 截图保存的文件夹
DELAY_SEC = 1.0                           # 每秒抓一帧
INPUT_SIZE = (640, 480)                    # 送入模型的尺寸 (W,H)

os.makedirs(SAVE_DIR, exist_ok=True)

def ts_str():
    return datetime.now().strftime("%Y%m%d%H%M%S")

def density_to_heatmap(density_map: np.ndarray):
    dm = density_map.copy()
    dm -= dm.min()
    if dm.max() > 0:
        dm = dm / dm.max()
    dm = (dm * 255).astype(np.uint8)
    dm_color = cv2.applyColorMap(dm, cv2.COLORMAP_JET)
    return dm_color


# 备选的软解码回退方案
def frames_optimized(rtsp_url: str, fps=1):
    """优化的软解码方案"""
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"无法打开 RTSP 流: {rtsp_url}")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, fps)

    last_frame_time = 0
    frame_interval = 1.0 / fps if fps > 0 else 0
    try:
        while True:
            current_time = time.time()
            if current_time - last_frame_time >= frame_interval:
                # 清空缓存区，只保留最新的
                for _ in range(int(cap.get(cv2.CAP_PROP_BUFFERSIZE))+ 1):
                    if not cap.grab():
                        break

                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    last_frame_time = current_time
                    yield frame
                else:
                    # 重新连接逻辑
                    print("读取帧失败，尝试重连...")
                    time.sleep(1)
                    cap.release()
                    cap = cv2.VideoCapture(rtsp_url)
                    if not cap.isOpened():
                        print(f"重新连接失败: {rtsp_url}")
                        break
            else:
                time.sleep(0.01) # 减少 CPU占用
    finally:
        if cap.isOpened():
            cap.release()



# 新增统一的帧源
from .ffmpeg_hw_reader import read_frames_hw   # VPU硬解，NV12用Python转成BGR，更省带宽/IO

def frames_from_vpu (rtsp_url: str, fps=1, first_timeout=10.0, debug=True):
    """使用硬件加速读取帧"""
    try:
        print("使用硬件加速解码...")
        for frame in read_frames_hw(rtsp_url, fps=fps, first_timeout=first_timeout, debug=debug):
            yield frame
        return
    except Exception as e:
        print(f"硬件加速失败: {e}")
        print("切换到优化软件解码...")
    # 回退方案
    for frame in frames_optimized(rtsp_url, fps=fps):
        yield frame

def main(rtsp_url=None):
    # —— 交互式输入 RTSP ——
    if rtsp_url is None:
        rtsp_url = input("请输入 RTSP 地址：").strip()

    if not check_rtsp_stream(rtsp_url):
        print("RTSP 检查失败，推出程序")
        return
    print("RTSP 检查通过，开始加载模型")

    # —— 加载模型（写死路径） ——
    rknn = csrnet_load_model(RKNN_MODEL)

    print("开始处理，VPU硬解码，每秒抽一帧...")
    try:
        for frame in frames_from_vpu(rtsp_url, fps=1, first_timeout=10.0, debug=True):

            # —— 推理 ——（送入模型的就是这张 resize 后的图）
            count, dm, inf_ms, vis_bgr = csrnet_infer_frame(
                rknn, frame, input_size=INPUT_SIZE
            )
            # 生成热力图，以及保存检测图和热力图
            stamp = ts_str()
            raw_name = f"Crowd_{count}_{stamp}.jpg"
            map_name = f"Map_{count}_{stamp}.jpg"

            raw_path = os.path.join(SAVE_DIR, raw_name)
            map_path = os.path.join(SAVE_DIR, map_name)

            heat = density_to_heatmap(dm)

            # 两张图保存到同一目录
            cv2.imwrite(raw_path, vis_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            cv2.imwrite(map_path, heat, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            print(f"[ 人员密度检测 ] 人数:{count}  推理耗时:{inf_ms:.1f}ms  => {os.path.basename(raw_path)} / {os.path.basename(map_path)}")

    except KeyboardInterrupt:
        print("用户中断，推出。")
    finally:
        rknn.release()

if __name__ == "__main__":
    main()




