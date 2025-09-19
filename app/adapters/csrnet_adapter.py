# encoding: utf-8
# @File  : csrnet_adapter.py
# @Author: Xinghui
# @Date  : 2025/09/17/10:51

# 包一层线程启动/停止（用你现成 csrnet_infer）

import threading
import re
import time
from typing import Optional
from app.core.config import Config
from app.core.reporter import report_alarm_payload

from detect.csrnet_infer import csrnet_load_model, csrnet_infer_frame, frames_from_vpu, INPUT_SIZE
import cv2

def safe_name(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', s or 'noname')

class CsrnetAdapter:
    def __init__(self):
        self.enabled = False
        self.thread: Optional[threading.Thread] = None
        self.stop_evt = threading.Event()
        self.model = None
        self.session_id = "default"
        # 运行参数
        self.min_normal = 10
        self.max_normal = 20
        self.conf_thresh = 0.5
        self.media_name = ""
        self.media_url  = ""
        self.base_algname = "人员拥挤检测"
        self.alg_name = "p2pnetconfig"

    def start(self, session_id: str, media_name: str, media_url: str, **opts):
        if self.enabled:
            return
        self.enabled = True
        self.stop_evt.clear()

        self.session_id = session_id
        self.media_name = media_name or ""
        self.media_url  = media_url or ""

        # 读取任务里的参数
        self.base_algname = str(opts.get("baseAlgname", "人员拥挤检测"))
        self.alg_name     = str(opts.get("name", "p2pnetconfig"))
        rng = (opts.get("normalRange") or {})
        self.min_normal = int(rng.get("min", 10))
        self.max_normal = int(rng.get("max", 20))
        self.conf_thresh = float(opts.get("confThresh", 0.5))

        if self.model is None:
            # TODO: 换成你的 rknn 模型路径
            self.model = csrnet_load_model("/home/tom/model/CSRNet_2_test.rknn")

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.enabled:
            return
        self.enabled = False
        self.stop_evt.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def _loop(self):
        safe_media = safe_name(self.media_name)
        snap_dir = Config.SNAP_DIR / "csrnet" / f"{self.session_id}__media_{safe_media}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        for bgr in frames_from_vpu(self.media_url, fps=1, first_timeout=10.0, debug=False):
            if self.stop_evt.is_set():
                break

            count, dm, inf_ms, vis_bgr = csrnet_infer_frame(self.model, bgr, input_size=INPUT_SIZE)
            if count < self.min_normal:
                label = "spare"
            elif count > self.max_normal:
                label = "crowded"
            else:
                label = "normal"

            ts = int(time.time())
            snap_path = snap_dir / f"{ts}.jpg"
            cv2.imwrite(str(snap_path), vis_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            # 组织 UserData（对象）
            user_data = {
                "count": int(count),
                "label": label,
                "infer_ms": float(inf_ms),
            }

            # ResultType：建议与 UI 一致，比如“拥挤度”或“人员拥挤检测”
            try:
                report_alarm_payload(
                    media_name=self.media_name,
                    media_url=self.media_url,
                    result_type=self.base_algname,           # ["人员拥挤检测"]
                    img_path=str(snap_path),
                    user_data=user_data,
                    uploadstatus="1",                         # 已上报图片
                    result_desc=""                            # 需要的话可填写
                )
            except Exception as e:
                print("[csrnet] report error:", e)


