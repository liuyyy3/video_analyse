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
from datetime import datetime

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
        self.period_sec = 60  # 每60秒跑一次


    def start(self, session_id: str, media_name: str, media_url: str, **opts):
        if self.enabled:
            return
        self.enabled = True
        self.stop_evt.clear()

        self.session_id = session_id
        self.media_name = media_name or ""
        self.media_url = media_url or ""
        self.base_algname = str(opts.get("baseAlgname", "遗留物检测"))
        self.alg_name = str(opts.get("name", "leftover_detect"))
        self.conf_thresh = float(opts.get("confThresh", 0.5))
        self.period_sec = float(opts.get("periodSec", 60))  # 可以由后端配置

        # self.thread = threading.Thread(target=self._run_once, daemon=True)  # 只跑一次
        self.thread = threading.Thread(target=self._run_loop, daemon=True)  # 每一分钟跑一次
        self.thread.start()

    def stop(self):
        self.enabled = False
        self.stop_evt.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def _read_classes_from_debug_out(self) -> (List[str], int):

        # 从 ./debug_out/results.json 读取 objects 列表，返回 (classes, count)
        results = Path("/home/tom/activityMonitor/video_analyse/app/utils/results.json")
        classes: List[str] = []
        if results.exists():
            try:
                data = json.loads(results.read_text(encoding="utf-8"))
                for o in (data.get("objects") or []):
                    lab = str(o.get("label", "unknown")).strip()
                    if lab:
                        classes.append(lab)
            except Exception as e:
                print("[leftover] bad results.json:", e)
        count = len(classes)
        return classes, count

    def _run_once(self):
        try:
            # 跑现有离线流程，生成 ./debug_out/6_yolo_overlay.png
            leftover.main()

            # 快照目录
            safe_media = safe_name(self.media_name)
            dst_dir = Config.SNAP_DIR / f"{self.session_id}__media_{safe_media}" / "leftover"
            dst_dir.mkdir(parents=True, exist_ok=True)

            overlay = Path("./debug_out/6_yolo_overlay.png")
            has_pic = overlay.exists()

            # 组装 classes & count
            classes, count = self._read_classes_from_debug_out()

            # dst = dst_dir / overlay.name

            if has_pic:
                ts = time.strftime("%Y%m%d_%H%M%S")
                dst = dst_dir / f"leftover_{count}_{ts}{overlay.suffix}"
                try:
                    shutil.copy2(overlay, dst)
                except Exception as exc:
                    print(f"[leftover] failed to copy overlay: {exc}")
            else:
                dst = None

            # 最小 UserData
            user_data = {
                "count": int(count),
                "classes": classes
            }

            # 上报
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
            pass

    def _run_loop(self):
        # 立即跑一次，然后每 period_sec 秒跑一次；随时响应 stop_evt
        first = True
        while not self.stop_evt.is_set():
            if not first:
                # 等待 periodSec；期间若收到 stop 则退出
                if self.stop_evt.wait(self.period_sec):
                    break
            first = False
            try:
                self._run_once()
            except Exception as e:
                print("[leftover] loop error:", e)
        self.enabled = False


    # 工具函数：结果读取 / 目录 / 静态路径 ----

    def _as_static(self, p: Path) -> str:
        # 把磁盘路径换成 Node 能访问的 /static 相对路径
        try:
            rel = p.relative_to(Config.SNAP_DIR)
            return f"/static/{rel.as_posix()}"
        except Exception:
            return str(p)

