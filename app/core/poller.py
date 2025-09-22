# encoding: utf-8
# @File  : poller.py
# @Author: Xinghui
# @Date  : 2025/09/17/16:48

# 定时去 Node 拉 配置/控制
import os
import threading
import time
import json
import ast
import requests
from typing import Dict, Any, Set

from app.core.config import Config
from app.adapters.csrnet_adapter import CsrnetAdapter
from app.adapters.leftover_adapter import LeftoverAdapter

VERBOSE = os.getenv("VERBOSE", "0") == "1"

# app/core/poller.py 里
def _norm_alg(item: dict) -> str:
    """
    把 UserData 里的一条算法配置映射为内部键：
      - 'csrnet'   人员拥挤/人群密度  (p2pnetconfig / crowd / csrnet / 含“拥挤”“密度”“人群”)
      - 'leftover' 遗留物            (p2pnetconfig2 / leftover / yolo / 含“遗留”“遗留物”)
    匹配尽量宽松，避免因为命名差异识别失败。
    """
    name = str(item.get("name") or "").strip().lower()
    base = str(item.get("baseAlgname") or "").strip()
    base_l = base.lower()

    # —— csrnet（拥挤/密度）——
    if any(k in name for k in ("p2pnetconfig", "csrnet", "crowd")):
        return "csrnet"
    if any(kw in base for kw in ("拥挤", "密度", "人群", "人员拥挤")):
        return "csrnet"

    # —— leftover（遗留物）——
    if any(k in name for k in ("p2pnetconfig2", "leftover", "yolo")):
        return "leftover"
    if any(kw in base for kw in ("遗留", "遗留物")):
        return "leftover"

    return ""



def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y", "on")
    if isinstance(v, (int, float)):
        return v != 0
    return False


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

    def apply(self, media_name: str, media_url: str, user_data, running: bool):
        # --- ① 解析 UserData：兼容字符串 JSON / Python 字面量 / 各种包裹 ---
        raw_ud = user_data
        if not raw_ud:
            raw_ud = []
        if isinstance(raw_ud, dict):
            raw_ud = (raw_ud.get("list") or raw_ud.get("data")
                      or raw_ud.get("UserData") or raw_ud.get("userdata")
                      or [])
        if isinstance(raw_ud, str):
            s = raw_ud.strip()
            if os.getenv("VERBOSE", "0") == "1":
                print(f"[task {self.sid}] raw UserData (len={len(s)}): {s[:160]}")
            if s:
                try:
                    user_data = json.loads(s)
                except Exception as e1:
                    try:
                        user_data = ast.literal_eval(s)
                    except Exception as e2:
                        print(f"[task {self.sid}] UserData parse failed: {e1} | {e2}")
                        user_data = []
            else:
                user_data = []
        else:
            user_data = raw_ud

        if isinstance(user_data, str):
            user_data = [user_data]

        if not isinstance(user_data, list):
            user_data = []

        if os.getenv("VERBOSE", "0") == "1":
            print(f"[task {self.sid}] ControlCommand={int(running)} media={media_name} url={media_url!r}")
            for i, it in enumerate(user_data):
                print("  - alg[{:d}]: name={!r}, baseAlgname={!r}, enabled={}, confThresh={}, normalRange={}".format(
                    i, it.get("name"), it.get("baseAlgname"),
                    it.get("enabled"), it.get("confThresh"), it.get("normalRange")))

        new_sel, new_opts = set(), {}
        for it in user_data:
            if not isinstance(it, dict):
                continue
            key = _norm_alg(it)
            if VERBOSE:
                print((f"     -> mapped key={key!r}"))
            if not key:
                continue
            new_opts[key] = it
            if bool(it.get("enable", False)):
                new_sel.add(key)

        old_sel = set(self.selected)
        self.selected = new_sel
        self.opts = new_opts
        self.media_name = media_name or ""
        self.media_url = media_url or ""

        # ③ 决策：只有 ControlCommand==1 且存在 enabled:true 的算法才会启动
        if not running:
            if old_sel:
                if os.getenv("VERBOSE", "0") == "1":
                    print(f"[task {self.sid}] stop all (ControlCommand=0)")
            for k in old_sel:
                self.adapters.get(k) and self.adapters[k].stop()
            return

        if running and not new_sel:
            if os.getenv("VERBOSE", "0") == "1":
                print(f"[task {self.sid}] no enabled algorithms -> nothing to start")
            # 若之前有运行中的、但现在全被取消勾选，也要停
            for k in (old_sel - new_sel):
                self.adapters.get(k) and self.adapters[k].stop()
            return

        # ④ 热更新：新增勾选→启动；取消勾选→停止
        for k in (new_sel - old_sel):
            if os.getenv("VERBOSE", "0") == "1":
                print(
                    f"[task {self.sid}] START {k} media={self.media_name} url={self.media_url!r} opts={self.opts.get(k)}")
            self.adapters[k].start(
                session_id=self.sid,
                media_name=self.media_name,
                media_url=self.media_url,
                **(self.opts.get(k) or {})
            )
        for k in (old_sel - new_sel):
            if VERBOSE:
                print(f"[task {self.sid}] STOP {k} (disabled)")
            self.adapters.get(k) and self.adapters[k].stop()


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

        if VERBOSE:
            print(f"[poller] GET {url} params={params}")
        r = requests.get(url, params=params, headers=headers, timeout=8)
        if VERBOSE:
            print(f"[poller] <- {r.status_code}")
        if r.status_code != 200:
            print("[poller] err:", r.text[:200])
            return

        resp = r.json()
        rows = resp.get("data") or resp.get("rows") or resp.get("list") or []
        if isinstance(resp.get("data"), dict):
            rows = resp["data"].get("list") or resp["data"].get("data") or rows
        if VERBOSE:
            print(f"[poller] rows={len(rows)}")

        for t in rows:
            sid = str(t.get("AlgTaskSession") or t.get("id") or t.get("TaskDesc") or "default")
            run = int(t.get("ControlCommand", 0)) == 1
            mname = str(t.get("MediaName") or "")
            murl = str(t.get("MetadataUrl") or t.get("MediaUrl") or "")
            udata = t.get("UserData") or []
            if VERBOSE:
                print(f"[poller] task sid={sid} run={run} media={mname} url={murl!r}")

            sess = self.sessions.get(sid) or SessionState(sid)
            self.sessions[sid] = sess
            # 这里的 apply 内部会根据 ControlCommand & enabled 作出“启动/停止”的决定，并打印原因
            sess.apply(mname, murl, udata, run)

