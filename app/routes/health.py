# encoding: utf-8
# @File  : health.py
# @Author: Xinghui
# @Date  : 2025/09/17/16:47

# 健康检查

from flask import Buleprint, jsonify
bp = Buleprint("health", __name__)

@bp.get("/health")
def health():
    return jsonify({"ok": True})



