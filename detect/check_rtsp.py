# encoding: utf-8
# @File  : check_rtsp.py.py
# @Author: Xinghui
# @Date  : 2025/09/11/10:03


import cv2
import numpy
import ffmpeg
import subprocess

def check_rtsp_stream(rtsp_url, timeout=5):
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width, height, r_frame_rate, codec_name',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        rtsp_url
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        output = result.stdout.decode().strip()
        if output:
            print("RTSP流可以访问，视频信息如下: ")
            print(output)
            return True
        else:
            print("ffprobe没有返回有效信息，RTSP流可能不存在或无法播放")
            print(result.stderr.decode())
            return False

    except subprocess.TimeoutExpired:
        print(f"ffprobe超时，RTSP流无响应 (>{timeout}s)")
        return False
    except Exception as e:
        print(f"出现异常: {e}")
        return False

if __name__ == "__main__":
    rtsp_url = input("请输入RTSP地址: ")
    check_rtsp_stream(rtsp_url)



