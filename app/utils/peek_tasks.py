# encoding: utf-8
# @File  : peek_tasks.py
# @Author: Xinghui
# @Date  : 2025/09/18/14:24
import argparse
# 用于查询后端服务器上传了什么的小脚本

import os
import requests
import json

NODE_BASE = os.getenv("NODE_BASE", "http://192.168.9.60:8080")
TOKEN = os.getenv("NODE_TOKEN", "")
URL = f"{NODE_BASE}/task/alg_task_fetch"
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pageNum", "-p", type=int,  default=int(os.getenv("PAGE_NUM", 1)))
    p.add_argument("--pageSize", "-s", type=int, default=int(os.getenv("PAGE_SIZE", 10)))
    p.add_argument("--keyword", "-k",            default=os.getenv("KEYWORD", ""))
    return p.parse_args()

def main():
    args = parse_args()
    params = {"pageSize": args.pageSize, "pageNum": args.pageNum, "keyword": args.keyword}

    try:
        print(f" -> GET {URL}  params = {params}")
        r = requests.get(URL, headers=HEADERS, params=params, timeout=8)
        print("status:", r.status_code)
        print("final url:", r.url)  # 实际请求的完整URL，便于确认
        r.raise_for_status()
        data = r.json()

        print("原始返回")
        print(json.dumps(data, ensure_ascii=False, indent=2))

        tasks = data.get("data") or data
        if isinstance(tasks, dict):
            tasks = [tasks]

        print("某个展开任务的 UserData")
        for t in tasks:
            sid = t.get("AlgTaskSession")
            media = t.get("MediaName") or t.get("MediaUrl")
            ctrl = t.get("ControlCommand")
            print(f"\n[Session {sid}] Media={media}, ControlCommand={ctrl}")

            raw_ud = t.get("UserData", [])
            if isinstance(raw_ud, str):
                try:
                    user_data = json.load(raw_ud)
                except json.JSONDecodeError:
                    print(" !! 警告：UserData 是字符串但无法解析", raw_ud[:120], "...")

            else:
                user_data = raw_ud

            for u in user_data:
                name = u.get("name")
                enabled = u.get("enabled")
                rng = (u.get("normalRange") or {})
                # rng_min, rng_max = rng.get("min"), rng.get("max")
                print(f"  - name={name!r}, enabled={enabled}, normalRange(min={rng.get('min')}, max={rng.get('max')})")
                # 看完整原始条目

    except Exception as e:
        print("请求失败: ", e)
        # 添加打印原始响应文本的代码
        if 'r' in locals():
            print(f"原始响应文本: {r.text}")
        else:
            print("未收到响应，可能是网络连接问题或超时")

if __name__ == "__main__":
    main()


