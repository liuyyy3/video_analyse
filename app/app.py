# encoding: utf-8
# @File  : app.py
# @Author: Xinghui
# @Date  : 2025/09/17/10:17

# 注册蓝图、加载配置，启动后端的主程序代码
# 兼容两种运行方式：
    # 1) python3 -m app.app
    # 2) python3 app/app.py   （必须在项目根目录执行）

from flask import Flask, jsonify
import os
import atexit
from typing import Optional
from .routes.health import bp as health_bp
from app.core.poller import Poller
from .core.config import Config

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"ok", True})

# 启动轮询获取 json信息的线程
poller = Poller(interval=5.0)
poller.start()
print("[boot] poller started, target:", os.getenv("NODE_BASE", "http://192.168.9.60:8080"))

def _bye():
    try:
        poller.stop()
        print("[shutdown] poller stopped")
    except Exception:
        pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8888")), threaded=True)




