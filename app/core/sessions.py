# encoding: utf-8
# @File  : sessions.py
# @Author: Xinghui
# @Date  : 2025/09/17/16:48

# 会话状态 & 启停调度（关键：enabled ∧ ControlCommand）

import json
from dataclasses import dataclass, field
from typing import Dict, Any, Set
from app.adapters.csrnet_adapter import CsrnetAdapter
from app.adapters.leftover_adapter import LeftoverAdapter
from app.utils.name_map import norm_alg_item


@dataclass
class Session:
    sid: str
    media_name: str = ""
    media_url: str = ""
    selected: Set[str] = field(default_factory=set)  # 勾选的算法集合（PY38 用 typing.Set）
    running: bool = False                            # ControlCommand == 1 ?
    opts: Dict[str, dict] = field(default_factory=dict)
    algs: Dict[str, Any] = field(default_factory=lambda: {
        "csrnet": CsrnetAdapter(),
        "leftover": LeftoverAdapter(),
    })

SESSIONS: Dict[str, Session] = {}

def ensure_session(sid: str) -> Session:
    if sid not in SESSIONS:
        SESSIONS[sid] = Session(sid=sid)
    return SESSIONS[sid]

def start_selected(sess: Session):
    """只启动被勾选的算法"""
    for key in sess.selected:
        adapter = sess.algs.get(key)
        if adapter is None:
            continue
        opts = sess.opts.get(key, {})
        adapter.start(
            session_id=sess.sid,
            media_name=sess.media_name,
            media_url=sess.media_url,
            **opts
        )

def stop_all(sess: Session):
    for _key, adapter in sess.algs.items():
        adapter.stop()

def apply_config(sess: Session, media_name: str, media_url: str, items, running_now: bool):
    """
    把 Node 拉来的“任务配置”应用到本地：更新勾选集合/参数。
    如果当前 running_now=True，做热更新：新增则启动，被取消则停止。
    """
    # 兼容后端把 UserData 当成字符串 JSON 的情况
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except Exception:
            items = []

    if not isinstance(items, list):
        items = []

    old = set(sess.selected)
    new: Set[str] = set()
    all_opts: Dict[str, dict] = {}

    for it in items:
        name = (it or {}).get("name")
        key = norm_alg_item(it)
        if not key:
            continue
        all_opts[key] = it
        if bool(it.get("enabled", False)):
            new.add(key)

    sess.media_name, sess.media_url = media_name, media_url
    sess.selected = new
    sess.opts = all_opts

    if running_now:
        # 新增勾选 → 立即启动
        for key in (new - old):
            adapter = sess.algs.get(key)
            if adapter is None:
                continue
            adapter.start(
                session_id=sess.sid,
                media_name=media_name,
                media_url=media_url,
                **sess.opts.get(key, {})
            )
        # 取消勾选 → 立即停止
        for key in (old - new):
            adapter = sess.algs.get(key)
            if adapter is None:
                continue
            adapter.stop()




