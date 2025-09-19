# encoding: utf-8
# @File  : logger.py
# @Author: Xinghui
# @Date  : 2025/09/17/10:52

# Node 的算法名 → 你的内部名（方便以后扩展）

def norm_alg(name:str) -> str | None:
    n = (name or "").lower()
    if n in ("csrnet", "p2pnetconfig","p2pnetconfig2"):
        return "csrnet"

    if n in ("leftover", "leftover_detect"):
        return "leftover"

    return None



