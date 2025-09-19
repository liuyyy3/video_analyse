# encoding: utf-8
# @File  : csrnet_adapter.py
# @Author: Xinghui
# @Date  : 2025/09/17/10:51

# 包一层线程启动/停止（用你现成 csrnet_infer）
import threading
import time
from typing import Optional, Dict, Any
from app.core.config import Config
from app.core.reporter import report_alarm

from detect.csrnet_infer import csrnet_load_model, csrnet_infer_frame, frames_from_vpu, INPUT_SIZE
import cv2

class CsrnetAdapter:
    def __init__(self):
        self.enabled = False
        self.thread: Optional[threading.Thread] = None
        self.stop_evt = threading.Event()
        self.model = None

    def start(self, session_id: str, media_name: str, media_url: str, *opts):
        if self.enabled:  # 幂等
            return
        self.enabled = True
        self.stop_evt.clear()
        self.session_id = session_id

        rng = (opts.get("normalRange") or {})
        self.min_normal = int(rng.get("min", 10))
        self.max_normal = int(rng.get("max", 20))

        self.conf_thresh = float(opts.get("confThresh", 0.5))

        if self.model is None:
            self.model = csrnet_load_model("/home/tom/model/CSRNet_2_All.rknn")
        self.thread = threading.Thread(
            target=self._loop, args=(media_name, media_url), daemon=True
        )
        self.thread.start()

    def stop(self):
        if not self.enabled:
            return
        self.enabled = False
        self.stop_evt.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def _loop(self, media_name: str, media_url: str):
        # sid = "default"
        snap_dir = Config.SNAP_DIR / "csrnet" / (media_name or "noname")
        snap_dir.mkdir(parents=True, exist_ok=True)

        for bgr in frames_from_vpu(media_url, fps=1, first_timeout=10.0, debug=False):
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

            # 将结果上报给 node端
            info = {"count": int(count), "label": label, "infer_ms": float(inf_ms),
                    "normalRange": {"min": self.min_normal, "max": self.max_normal}}
            try:
                report_alarm(self.session_id, "拥挤度", str(snap_path), info)
            except Exception as e:
                print("[csrnet] report error:", e)








