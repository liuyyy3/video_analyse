# encoding: utf-8
# @File  : leftover_adapter.py
# @Author: Xinghui
# @Date  : 2025/09/17/10:51

# 遗留物检测

import threading
import time
import shutil
import re
import json
from pathlib import Path
from typing import Optional, List

from app.core.config import Config
from app.core.reporter import report_alarm_payload

import detect.Image_diff_RK as leftover


def safe_name(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', s or 'noname')

class LeftoverAdapter:
    def __init__(self):
        self.enabled = False
        self.thread: Optional[threading.Thread] = None
        self.stop_evt = threading.Event()
        self.session_id = "default"
        self.conf_thresh = 0.5
        self.media_name = ""
        self.media_url  = ""
        self.base_algname = "遗留物检测"
        self.alg_name = "leftover_detect"

    def start(self, session_id: str, media_name: str, media_url: str, **opts):
        if self.enabled:
            return
        self.enabled = True
        self.stop_evt.clear()

        self.session_id = session_id
        self.media_name = media_name or ""
        self.media_url  = media_url or ""
        self.base_algname = str(opts.get("baseAlgname", "遗留物检测"))
        self.alg_name     = str(opts.get("name", "leftover_detect"))
        self.conf_thresh  = float(opts.get("confThresh", 0.5))

        self.thread = threading.Thread(target=self._run_once, daemon=True)
        self.thread.start()

    def stop(self):
        self.enabled = False
        self.stop_evt.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def _read_classes_from_debug_out(self) -> List[str]:
        """
        从 debug_out 里尽量读取 YOLO 的类别结果。
        优先 results.json，其次 labels.txt，实在没有返回 []。
        """
        classes: list[str] = []

        res_json = Path("./debug_out/results.json")
        if res_json.exists():
            try:
                data = json.loads(res_json.read_text(encoding="utf-8"))
                objs = data.get("objects") or []
                classes = [str(o.get("label", "unknown")) for o in objs if o is not None]
                # 过滤空白
                classes = [c for c in classes if c]
            except Exception:
                pass

        if not classes:
            labels_txt = Path("./debug_out/labels.txt")
            if labels_txt.exists():
                try:
                    classes = [ln.strip() for ln in labels_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
                except Exception:
                    pass
        return classes

    def _run_once(self):
        try:
            # 1) 跑你现有离线流程，生成 ./debug_out/6_yolo_overlay.png（或你真实的输出）
            leftover.main()

            # 2) 快照目录
            safe_media = safe_name(self.media_name)
            dst_dir = Config.SNAP_DIR / "leftover" / f"{self.session_id}__media_{safe_media}"
            dst_dir.mkdir(parents=True, exist_ok=True)

            overlay = Path("./debug_out/6_yolo_overlay.png")
            ts = int(time.time())
            dst = dst_dir / f"{ts}.png"
            has_pic = overlay.exists()
            if has_pic:
                shutil.copy2(overlay, dst)

            # 3) 组装 classes & count
            classes = self._read_classes_from_debug_out()
            if not classes:
                if has_pic:
                    # 有结果图但没解析到类别 → unknown 兜底，视为 1 个
                    classes = ["unknown"]
                    count = 1
                else:
                    classes = []
                    count = 0
            else:
                count = len(classes)

            # 4) 最小 UserData
            user_data = {
                "count": int(count),
                "classes": classes
            }

            # 5) 上报
            try:
                report_alarm_payload(
                    media_name=self.media_name,
                    media_url=self.media_url,
                    result_type=self.base_algname,          # 例如 "遗留物检测"
                    img_path=str(dst if has_pic else ""),
                    user_data=user_data,
                    uploadstatus=("1" if has_pic else "2"),
                    upload_reason=("" if has_pic else "无可用结果图/未检测到目标")
                )
            except Exception as e:
                print("[leftover] report error:", e)

        finally:
            self.enabled = False




