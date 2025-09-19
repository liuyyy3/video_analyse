# encoding: utf-8
# @File  : sessions.py
# @Author: Xinghui
# @Date  : 2025/09/17/16:48

# 会话状态 & 启停调度（关键：enabled ∧ ControlCommand）

from dataclasses import dataclass, field
from typing import Dict, Any
from app.adapters.csrnet_adapter import CsrnetAdapter
from app.adapters.leftover_adapter import LeftoverAdapter

class Session:
    sid: str
    media_name: str = ""
    media_url: str = ""
    selected: set[str] = field(default_factory=set)  # 勾选的算法集合
    running: bool = False  # ControlCommand == 1 ?
    opts: dict[str, dict] = field(default_factory=dict)
    algs: dict[str, Any] = field(default_factory=lambda:{
        "csrnet": CsrnetAdapter(),
        "leftover": LeftoverAdapter(),
    })

SESSIONS: dict[str, Session] = {}

def ensure_session(sid: str) -> Session:
    if sid not in SESSIONS:
        SESSIONS[sid] = Session(sid = sid)
    return SESSIONS[sid]

def start_selected(sess: Session):
    # 只启动运行被勾选的算法
    for key in sess.selected:
        sess.algs[key].start(media_name = sess.media_name, media_url = sess.media_url, **sess.opt.get(key, {}))

def stop_all(sess: Session):
    for key, adapter in sess.algs.items():
        adapter.stop()

def apply_config(sess:Session, media_name: str, media_url: str, items:list, running_now: bool):
    # 把 Node 拉来的“任务配置”应用到本地：更新勾选集合/参数。
    # 如果当前 running_now=True，做热更新：新增则启动，被取消则停止。
    old = set(sess.selected)
    new = set()
    new_opts: dict[str, dict] = {}

    from app.utils.name_map import norm_alg
    for it in items:
        key = norm_alg(it.get("name"))
        if not key:
            continue
        new.add(key)
        new_opts[key] = it

    sess.media_name, sess.media_url = media_name, media_url
    sess.selected = new
    sess.opts.update(new_opts)

    if running_now:
        for key in new - old:  # 新增勾选 → 立即启动
            sess.algs[key].start(media_name = media_name, media_url = media_url, **sess.opts.get(key, {}))
        for key in old - new:  # 取消勾选 → 立即停止
            sess.algs[key].stpo()




