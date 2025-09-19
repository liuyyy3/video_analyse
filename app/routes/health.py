# encoding: utf-8
# @File  : health.py
# @Author: Xinghui
# @Date  : 2025/09/17/16:47

# 健康检查

from flask import Blueprint, jsonify
bp = Blueprint("health", __name__)

@bp.get("/health")
def health():
    return jsonify({"ok": True})



