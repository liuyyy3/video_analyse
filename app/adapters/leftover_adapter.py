# encoding: utf-8
# @File  : leftover_adapter.py
# @Author: Xinghui
# @Date  : 2025/09/17/10:51

# 遗留物检测

import threading
import time
import shutil
from pathlib import Path
from typing import Optional
from app.core.config import Config
from app.core.reporter import report_alarm

import detect.Image_diff_RK as leftover

class LeftoverAdapter:
    def __init__(self):
        self.enable = False
        self.thread: Optional[threading.Thread] = None
        self.stop_evt = threading.Event()

    def start(self, media_name: str, media_url: str, **opts):
        if self.enable:
            return
        self.enable = True
        self.stop_evt.clear()
        self.thread = threading.Thread(target=self._run_once, args=(media_name,), daemon=True)
        self.thread.start()

    def stop(self):
        self.enabled = False
        self.stop_evt.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def _run_once(self, media_name: str):
        try:
            leftover.main()
            overlay = Path("./debug_out/6_yolo_overlay.png")
            ts = int(time.time())
            dst_dir = Config.SNAP_DIR / "leftover" / (media_name or "noname")
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{ts}.png"
            if overlay.exists():
                shutil.copy2(overlay, dst)

            try:
                report_alarm("default", "遗留物", str(dst), {"found": overlay.exists()})
            except Exception as e:
                print("[leftover] report error:", e)
        finally:
            self.enable = False



