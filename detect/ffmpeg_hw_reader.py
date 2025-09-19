# encoding: utf-8
# @File  : ffmpeg_hw_reader.py
# @Author: Xinghui
# @Date  : 2025/09/15/16:19

import subprocess
import os
import shlex
import numpy as np
import cv2
import select
import time
import threading

def probe_wh(rtsp: str, default=(1280, 720)):
    try:
        cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 {shlex.quote(rtsp)}"
        out = subprocess.check_output(cmd, shell=True, text=True).strip()
        w, h = map(int, out.split(','))
        return w, h
    except Exception:
        return default


def _build_cmd_hw(rtsp: str, codec: str = "h264", fps: int = 1, debug=False):
    """硬件加速命令 - 使用自动检测"""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-rtsp_transport", "tcp",
        "-loglevel", "error",
        "-hwaccel", "auto",
        "-i", rtsp,
        "-an",
        "-pix_fmt", "nv12",  # 硬件友好格式
        "-f", "rawvideo",
        "pipe:1",
    ]

    return cmd


def read_frames_hw(rtsp: str, fps: int = 1, first_timeout: float = 8.0, debug=False):
    """yield: HxWx3 uint8 的 BGR 帧（首帧超时 + 错误回显）"""
    # 用 ffprobe 拿分辨率（你已有 probe_wh 的话也可直接调用）

    w, h = probe_wh(rtsp)
    frame_bytes = (w * h * 3) // 2  # NV12: 1.5 * W * H
    cmd = _build_cmd_hw(rtsp, debug=debug)

    if debug:
        print("硬件加速命令:", " ".join(shlex.quote(x) for x in cmd))
        print(f"每帧大小: {frame_bytes} 字节")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 3
    )

    # 启动 stderr 读取线程
    stderr_lines = []
    def read_stderr():
        while proc.poll() is None:
            try:
                line = proc.stderr.readline()
                if line:
                    line_str = line.decode('utf-8', 'ignore').strip()
                    stderr_lines.append(line_str)
                    if debug and any(x in line_str.lower() for x in ['hw', 'accel', 'gpu']):
                        print(f"[HW] {line_str}")
            except:
                break


    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()

    try:
        fd = proc.stdout.fileno()
        deadline = time.time() + first_timeout
        frame_count = 0
        last_frame_time = 0
        frame_interval = 1.0 / max(0.1, fps)

        while True:
            buf = bytearray()
            start_time = time.time()

            while len(buf) < frame_bytes:
                if proc.poll() is not None:
                    err = "\n".join(stderr_lines[-5:])
                    raise RuntimeError(f"FFmpeg 已退出，最后错误:\n{err}")

                if time.time() > deadline:
                    err = "\n".join(stderr_lines[-5:])
                    raise TimeoutError(f"首帧超时，没有收到数据。最后错误:\n{err}")

                r, _, _ = select.select([fd], [], [], 0.5)
                if fd in r:
                    try:
                        to_read = min(frame_bytes - len(buf), 131072)
                        chunk = os.read(fd, to_read)
                        if not chunk:
                            break
                        buf.extend(chunk)
                    except BlockingIOError:
                        time.sleep(0.01)
                    except Exception as e:
                        if debug:
                            print(f"读取异常: {e}")
                        break
                else:
                    time.sleep(0.01)

            if len(buf) < frame_bytes:
                if debug:
                    print(f"帧数据不完整: {len(buf)}/{frame_bytes}")
                continue
            # 控制帧率
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                continue
            last_frame_time = current_time
            frame_count += 1

            if debug and frame_count % 5 == 1:
                print(f"帧 {frame_count}, 读取时间: {time.time() - start_time:.3f}s")



            # NV12 转成 BGR
            try:
                yuv_array = np.frombuffer(buf, dtype=np.uint8)
                bgr_frame = cv2.cvtColor(yuv_array.reshape((h * 3 // 2, w)), cv2.COLOR_YUV2BGR_NV12)
                yield bgr_frame
            except Exception as e:
                if debug:
                    print(f"帧转换错误: {e}")
                continue
            # 重置超时时间
            deadline = time.time() + 5.0

    except Exception as e:
        if debug:
            print(f"读取器错误: {e}")
            if stderr_lines:
                print("最后错误信息:", stderr_lines[-3:])
        raise

    finally:
        if proc and proc.poll() is None:

            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except:
                try:
                    proc.kill()
                except:
                    pass


# 测试函数
def test_hw_reader():
    """测试硬件加速读取器"""
    url = "rtsp://192.168.9.140:8554/1"
    print("测试硬件加速读取器...")

    try:
        start_time = time.time()
        frames_received = 0

        for i, frame in enumerate(read_frames_hw(url, fps=1, debug=True)):
            print(f"✓ 帧 {i}: {frame.shape}, 耗时: {time.time() - start_time:.2f}s")
            frames_received += 1

            if frames_received >= 5:  # 测试5帧
                break

        print(f"✓ 测试成功! 平均帧率: {frames_received / (time.time() - start_time):.2f}fps")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


if __name__ == "__main__":
    test_hw_reader()

