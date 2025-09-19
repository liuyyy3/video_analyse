# encoding: utf-8
# @File  : poller.py
# @Author: Xinghui
# @Date  : 2025/09/17/16:48

# 定时去 Node 拉 配置/控制

import threading, time, requests
from app.core.config import Config
from app.core.sessions import ensure_session, apply_config, start_selected, stop_all

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
        # TODO：按你们实际返回的 JSON 结构改这段解析
        headers = {"Authorization": f"Bearer {Config.TOKEN}"} if Config.TOKEN else {}
        r = requests.get(Config.TASK_FETCH_URL, headers=headers, timeout=5)
        r.raise_for_status()
        tasks = r.json().get("data") or r.json()

        for t in (tasks if isinstance(tasks, list) else [tasks]):
            sid = str(t.get("AlgTaskSession", "default"))
            media_name = t.get("MediaName") or ""
            media_url = t.get("MedaiUrl") or ""
            user_data = t.get("UaerData", [])
            ctrl = t.get("ControlCommand", None)

            sess = ensure_session(sid)

            apply_config(sess, media_name, media_url, user_data, running_now=sess.running)

            if ctrl is not None:
                should_run = (ctrl == 1)
                if should_run and not sess.running:
                    sess.running = True
                    start_selected(sess)
                elif (not should_run) and sess.running:
                    sess.running = False
                    stop_all(sess)

