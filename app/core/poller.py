# encoding: utf-8
# @File  : poller.py
# @Author: Xinghui
# @Date  : 2025/09/17/16:48

# 定时去 Node 拉 配置/控制

import threading, time, requests
from app.core.config import Config
from app.core.sessions import ensure_session, apply_config, start_selected, stop_all

import os
import json
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "10"))
PAGE_NUM = int(os.getenv("PAGE_NUM",  "1"))
KEYWORD = os.getenv("KEYWORD", "")

class Poller(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_evt = threading.Event()

    def run(self):
        while not self.stop_evt.is_set():
            try:
                self.tick()
            except Exception as e:
                print("[poller] error:", e)
            time.sleep(Config.POLL_INTERVAL)

    def stop(self):
        self.stop_evt.set()

    def tick(self):
        # === 1) 拉任务列表（你们后端的数据结构可能是：一个数组，包含每个任务的状态）===
        headers = {"Authorization": f"Bearer {Config.TOKEN}"} if Config.TOKEN else {}
        params = {"pageSize": PAGE_SIZE, "pageNum": PAGE_NUM, "keyword": KEYWORD}
        r = requests.get(Config.TASK_FETCH_URL, headers=headers, params=params, timeout=8)
        r.raise_for_status()
        payload = r.json()
        tasks = payload.get("data") or payload
        if isinstance(tasks, dict):
            tasks = [tasks]

        for t in tasks:
            sid = str(t.get("AlgTaskSession", "default"))
            media_name = t.get("MediaName") or ""
            media_url = t.get("MetadataUrl") or t.get("MediaUrl") or ""
            raw_ud = t.get("UserData", [])
            if isinstance(raw_ud, str):
                try:
                    user_data = json.loads(raw_ud)
                except Exception:
                    user_data = []
            else:
                user_data = raw_ud
            sess = ensure_session(sid)
            apply_config(sess, media_name, media_url, user_data, running_now=sess.running)

            # 统一起停（只对于被勾选的算法做一层停止启动的判断）
            ctrl = t.get("ControlCommand", None)
            if ctrl is not None:
                should_run = (int(ctrl) == 1)
                if should_run and not sess.running:
                    sess.running = True
                    start_selected(sess)
                elif (not should_run) and sess.running:
                    sess.running = False
                    stop_all(sess)




