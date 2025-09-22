# encoding: utf-8
# @File  : poller.py
# @Author: Xinghui
# @Date  : 2025/09/17/16:48
import json
# 定时去 Node 拉 配置/控制

import threading
import time
import requests
from typing import Dict, Any, Set

from app.core.config import Config
from app.core.sessions import ensure_session, apply_config, start_selected, stop_all

from app.adapters.csrnet_adapter import CsrnetAdapter
from app.adapters.leftover_adapter import LeftoverAdapter

def _norm_alg(item: dict) -> str:
    # 把任务里的算法配置项映射成内部键：csrnet / leftover
    name = (item.get("name") or "").lower()
    base = (item.get("baseAlgname") or "").lower()
    if "拥挤" in base or "p2pnetconfig" in name or "crowd" in name:
        return "csrnet"
    if "遗留" in base or "leftover" in name or "yolo" in name:
        return "leftover"
    return ""

class SessionState:
    def __init__(self, sid: str):
        self.sid = sid
        self.media_name = ""
        self.media_url = ""
        self.selected: Set[str] = set()
        self.opts: Dict[str, dict] = {}
        self.adapters: Dict[str, Any] = {
            "csrnet": CsrnetAdapter(),
            "leftover": LeftoverAdapter(),
        }

    def apply(self, media_name: str, media_url: str, usre_data, running: bool):
        if isinstance(usre_data, str):
            try: user_data = json.load(usre_data)
            except Exception: user_data = []
        if not isinstance(user_data, list):
            user_data = []

        new_sel: Set[str] = set()
        new_opts: Dict[str, dict] = {}
        for it in user_data:
            key = _norm_alg(it or {})
            if not key:
                continue
            new_opts[key] = it
            if bool((it or {}).get("enable", False)):
                new_sel.add(key)

        old_sel = set(self.selected)
        self.selected = new_sel
        self.opts = new_opts
        self.media_name = media_name or ""
        self.media_url = media_url or ""

        if not running:
            for k in old_sel:
                self.adapters.get(k, None) and self.adapters[k].stop()
            return
        # 运行：新增的启动，取消的停止
        for k in (new_sel - old_sel):
            self.adapters[k].start(
                session_id=self.sid,
                media_name=self.media_name,
                media_url=self.media_url,
                **self.opts.get(k, {})
            )
        for k in (old_sel - new_sel):
            self.adapters[k].stop()



class Poller(threading.Thread):
    def __init__(self, interval: float = 5.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_evt = threading.Event()
        self.sessions: Dict[str, SessionState] = {}

    def stop(self):
        self.stop_evt.set()
        for s in self.sessions.values():
            for a in s.adapters.values():
                try:
                    a.stop()
                except:
                    pass

    def run(self):
        while not self.stop_evt.is_set():
            try:
                self.tick()
            except Exception as e:
                print("[poller] error:", e)
            self.stop_evt.wait(self.interval)

    def tick(self):
        params = {"pageSize": Config.PAGE_SIZE, "pageNum": Config.PAGE_NUM, "keyword": ""}
        headers = {"Authorization": f"Bearer {Config.TOKEN}"} if Config.TOKEN else {}
        url = Config.TASK_FETCH_URL
        print(f"[poller] GET {url} params={params}")
        r = requests.get(url, params=params, headers=headers, timeout=8)
        print(f"[poller] <- {r.status_code}")
        if r.status_code != 200:
            print("[poller] err:", r.text[:200])
            return

        resp = r.json()
        rows = resp.get("data") or resp.get("rows") or resp.get("list") or []
        if isinstance(resp.get("data"), dict):
            rows = resp["data"].get("list") or resp["data"].get("data") or rows
        print(f"[poller] rows={len(rows)}")

        for t in rows:
            sid = str(t.get("AlgTaskSession") or t.get("id") or t.get("TaskDesc") or "default")
            run = int(t.get("ControlCommand", 0)) == 1
            mname = str(t.get("MediaName") or "")
            murl = str(t.get("MetadataUrl") or t.get("MediaUrl") or "")
            udata = t.get("UserData") or []
            print(f"[poller] task sid={sid} run={run} media={mname} url={murl!r}")

            sess = self.sessions.get(sid)
            if not sess:
                sess = SessionState(sid)
                self.sessions[sid] = sess

            before = set(sess.selected)
            sess.apply(mname, murl, udata, run)
            after = set(sess.selected)

            added = after - before
            removed = before - after
            if run and added:
                print(f"[poller] start {added} for sid={sid}")
            if removed:
                print(f"[poller] stop {removed} for sid={sid}")
            if not run and before:
                print(f"[poller] stop all for sid={sid}")





