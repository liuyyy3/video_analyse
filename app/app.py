# encoding: utf-8
# @File  : app.py
# @Author: Xinghui
# @Date  : 2025/09/17/10:17

# 注册蓝图、加载配置，启动后端的主程序代码

from flask import Flask
from app.routes.health import bp as health_bp
from app.core.poller import Poller
from app.core.config import Config

poller: Poller | None = None

def creat_app():
    app = Flask(__name__)
    app.register_blueprint(health_bp)
    return app

app = creat_app()

@app.before_first_request
def start_backgroung():
    global poller
    if poller is None:
        poller = Poller()
        poller.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, threaded=True)




