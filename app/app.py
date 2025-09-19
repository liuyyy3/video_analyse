# encoding: utf-8
# @File  : app.py
# @Author: Xinghui
# @Date  : 2025/09/17/10:17

# 注册蓝图、加载配置，启动后端的主程序代码

from flask import Flask
import os
from typing import Optional
from .routes.health import bp as health_bp
from .core.poller import Poller
from .core.config import Config

poller: Optional[Poller] = None

# 兼容两种运行方式：
# 1) python3 -m app.app
# 2) python3 app/app.py   （必须在项目根目录执行）
try:
    from .routes.health import bp as health_bp
except ImportError:
    from app.routes.health import bp as health_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(health_bp)
    # TODO：这里再注册你其他路由 / 挂后台线程等
    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8888")), threaded=True)




