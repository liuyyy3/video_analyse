# encoding: utf-8
# @File  : reporter.py
# @Author: Xinghui
# @Date  : 2025/09/17/16:49

# 把结果 POST 给 Node

import requests, json, time
from pathlib import Path
from app.core.config import Config

def report_alarm_payload(
    media_name: str,
    media_url: str,
    result_type,              # str 或 [str]
    img_path: str,
    user_data: dict,          # ← 现在是 dict（对象），不是数组
    uploadstatus: str = "1",  # "1"已上报 / "2"未上报
    result_desc: str = "",
    upload_reason: str = "",
    uploadvideo_path: str = ""
):
    """
    按你们的 README 要求构造并上报到 /warning/alg_alarm_fetch 的载荷。
    """
    payload = {
        "MediaName": str(media_name or ""),
        "MediaUrl":  str(media_url or ""),
        "ResultType": result_type if isinstance(result_type, list) else [str(result_type)],
        "ResultDescription": result_desc,
        "imgPath": str(img_path or ""),
        "UploadReason": upload_reason,
        "Uploadstatus": str(uploadstatus),
        "Uploadvideo_path": str(uploadvideo_path or ""),
        "UserData": user_data or {},   # ← 对象
        "time": int(time.time()),      # 可选，便于排查
    }
    headers = {"Authorization": f"Bearer {Config.TOKEN}"} if Config.TOKEN else {}
    r = requests.post(Config.REPORT_URL, json=payload, headers=headers, timeout=8)
    r.raise_for_status()
    return r.json()

