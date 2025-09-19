# encoding: utf-8
# @File  : reporter.py
# @Author: Xinghui
# @Date  : 2025/09/17/16:49

# 把结果 POST 给 Node

import requests, json, time
from pathlib import Path
from app.core.config import Config

def report_alarm(session_id: str, warnings_type:str, image_path:str, info:dict):
    """
    把一条结果 POST 给 Node 的 /warning/alg_alarm_fetch
    warning_type: 比如 "拥挤度" / "遗留物"
    image_path:   传你保存的快照相对/绝对路径（按你们约定）
    info:         算法详细结果（dict），会被序列化为 JSON 字符串
    """
    payload = {
        "AlgTaskSession": session_id,
        "warningType": [warnings_type],
        "imgPath": image_path,
        "warningInfo": json.dump(info, ensure_ascii=False),
        "time": int(time.time()),
    }
    headers = {"Authorization": f"Bearer {Config.TOKEN}"} if Config.TOKEN else {}
    r = requests.post(Config.REPORT_URL, json=payload, headers=headers, timeout=5)
    r.raise_for_status()
    return r.json()

