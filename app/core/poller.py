# encoding: utf-8
# @File  : poller.py
# @Author: Xinghui
# @Date  : 2025/09/17/16:48

# 定时去 Node 拉 配置/控制
import os
import re
import threading
import time
import json
import ast
import requests
from typing import Dict, Any, Set

from app.core.config import Config
from app.utils.name_map import norm_alg_item
from app.adapters.csrnet_adapter import CsrnetAdapter
from app.adapters.leftover_adapter import LeftoverAdapter

# 只从 MediaName 中解析，"3 - rtsp://x.x.x.x:8553/8"
# 不解析/不回退 MediaUrl
_DASH_CLASS = r'[\-\u2012-\u2015\u2212\uFE58\uFE63\uFF0D]'  # 常见各类破折号


VERBOSE = os.getenv("VERBOSE", "0") == "1"

# app/core/poller.py 里
def _norm_alg(item: dict) -> str:
    key = norm_alg_item(item)
    if key:
        return key

    return ""



def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y", "on")
    if isinstance(v, (int, float)):
        return v != 0
    return False


def stream_from_media_name(media_name: str):
    name = (media_name or "").strip()
    if not name:
        return ("", None)

    m = re.search(r'(rtsp[s]?://\S+)', name, re.I)
    if m:
        left = name[:m.start()].rstrip()
        left = re.sub(fr'{_DASH_CLASS}\s*$', '', left).strip()
        return (left or name, m.group(1).strip())

    # 从 - 后面分割出需要的 rtsp地址
    norm = re.sub(_DASH_CLASS, '_', name)
    if '-' in norm:
        left, right = norm.strip('-', 1)
        right = right.strip()
        if right.lower().startswith('rtsp://'):
            return (left.strip() or name, right)
    return (name, None)



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
                print(f"[task {self.sid}] raw UserData (len={len(s)}): {s[:200]}")
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

        if isinstance(user_data, dict):
            user_data = [user_data]

        if not isinstance(user_data, list):
            user_data = []

        # 先解析 MediaName 得到 URL
        clean_name, chosen_url = stream_from_media_name(media_name)

        if os.getenv("VERBOSE", "0") == "1":
            print(f"[task {self.sid}] ControlCommand={int(running)} media={chosen_url or media_name} url={media_url!r}")
            for i, it in enumerate(user_data):
                if isinstance(it, dict):
                    print("  - alg[{:d}]: name={!r}, baseAlgname={!r}, enabled={}, confThresh={}, normalRange={}".format(
                        i, it.get("name"), it.get("baseAlgname"),
                        it.get("enabled"), it.get("confThresh"), it.get("normalRange")))

        new_sel, new_opts = set(), {}
        enabled_seen, mapped_seen = [], []  # 仅用于调试总结

        for it in user_data:
            if not isinstance(it, dict):
                continue
            key = _norm_alg(it)
            if os.getenv("VERBOSE", "0") == "1":
                print(f"    -> mapped key={key!r}  (raw name={it.get('name')!r}, baseAlgname={it.get('baseAlgname')!r})")
            if key:
                mapped_seen.append(key)
                new_opts[key] = it
                flag = _to_bool(it.get("enabled", False))
                if os.getenv("VERBOSE", "0") == "1":
                    print(f"       enabled raw={it.get('enabled')!r} -> {_to_bool(it.get('enabled', False))}")
                if flag:
                    new_sel.add(key)
                    if os.getenv("VERBOSE", "0") == "1":
                        print(f"       -> enabled -> add '{key}'")
                else:
                    enabled_seen.append(f"{key}=False")

        if os.getenv("VERBOSE", "0") == "1":
            print(f"[task {self.sid}] computed new_sel={sorted(list(new_sel))}, mapped={mapped_seen}")

        old_sel = set(self.selected)
        # self.selected = new_sel
        self.opts = new_opts
        self.media_name = clean_name or ""
        self.media_url = chosen_url or ""

        if not self.media_url:
            if os.getenv("VERBOSE", "0") == "1":
                print(f"[task {self.sid}] MediaName 未解析到 rtsp:// 地址，跳过本轮。raw={media_name!r}")
            return

        # ③ 决策：只有 ControlCommand==1 且存在 enabled:true 的算法才会启动
        if not running:
            if old_sel:
                if os.getenv("VERBOSE", "0") == "1":
                    print(f"[task {self.sid}] stop all (ControlCommand=0)")
            for k in old_sel:
                self.adapters.get(k) and self.adapters[k].stop()
                # 进入停止状态后要清空运行列表，避免再次启动的时候无法出发 start()
            self.selected = set()
            return

        if running and not new_sel:
            if os.getenv("VERBOSE", "0") == "1":
                print(f"[task {self.sid}] no enabled algorithms -> nothing to start")
            self.selected = set()
            return

        # ④ 热更新：新增勾选→启动；取消勾选→停止
        for k in (new_sel - old_sel):
            if os.getenv("VERBOSE", "0") == "1":
                # print(f"[task {self.sid}] START {k} media={self.media_name} url={self.media_url!r} opts={self.opts.get(k)}")
                print(f"[task {self.sid}] START {k} media={self.media_url} opts={self.opts.get(k)}")

            self.adapters[k].start(
                session_id=self.sid,
                media_name=self.media_name,
                media_url=self.media_url,
                **(self.opts.get(k) or {})
            )

        for k in (old_sel - new_sel):
            if os.getenv("VERBOSE", "0") == "1":
                print(f"[task {self.sid}] STOP {k} (disabled)")
            self.adapters.get(k) and self.adapters[k].stop()
        self.selected = new_sel


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

