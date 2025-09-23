# encoding: utf-8
# @File  : config.py
# @Author: Xinghui
# @Date  : 2025/09/17/10:50

# 读取环境变量/默认配置

import os
from pathlib import Path

BASE = Path(os.getenv('VA_BASE_DIR', "/home/tom/video_analyse"))

class Config:
    # Node服务根地址
    NODE_BASE = os.getenv('NODE_BASE', "http://192.168.9.60:8080")  # NODE_BASE要换成node的实际地址，方便访问
    TASK_FETCH_URL = f"{NODE_BASE}/task/alg_task_fetch"  # 前端的新增|修改任务的接口，从这里获取 enable的信息
    REPORT_URL = f"{NODE_BASE}/warning/alg_alarm_fetch"  # 上报警告给 node后端

    # 鉴定权限，如果 node需要
    TOKEN = os.getenv("NODE_TOKEN", "")  # 前端的告警上报接口，从这里上报告警信息，node需要校验 token

    PAGE_SIZE = int(os.getenv("PAGE_SIZE", "10"))
    PAGE_NUM = int(os.getenv("PAGE_NUM", "1"))

    # 轮询频率（秒）
    POLL_INTERVAL = float(os.getenv('POLL_INTERVAL', "5.0"))

    # 告警画面 & 检测结果目录
    # SNAP_DIR = Path(os.getenv("SNAP_DIR", BASE / "snapshots"))
    SNAP_DIR = Path(os.getenv("SNAP_DIR", "/home/tom/activitiMonitor/public/static/"))

    # RESULT_DIR = Path(os.getenv("RESULT_DIR", BASE / "results"))

