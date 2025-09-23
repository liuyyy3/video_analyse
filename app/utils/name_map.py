# encoding: utf-8
# @File  : name_map.py
# @Author: Xinghui
# @Date  : 2025/09/17/10:52

# Node 的算法名 → 你的内部名（方便以后扩展）

from typing import Optional
from typing import Any
from typing import Mapping


# 精确别名（全部小写）
_EXACT_ALIAS = {
    "csrnet": "csrnet",
    "p2pnetconfig": "csrnet",
    "crowd": "csrnet",

    "left": "leftover",
    "leftover": "leftover",
    "leftover_detect": "leftover",
}

# 关键词（全部小写；出现即视为该算法）
_KEYWORDS = {
    "csrnet": "csrnet",
    "p2pnetconfig": "csrnet",
    "crowd": "csrnet",
    "人员": "csrnet",
    "密度": "csrnet",
    "拥挤度": "csrnet",
    "人员拥挤": "csrnet",

    "left": "leftover",
    "leftover": "leftover",
    "leftover_detect": "leftover",
    "遗留": "leftover",
    "遗留物": "leftover",
}


def norm_alg(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    t = str(name).strip().lower()
    if not t:
        return None

    if t in _EXACT_ALIAS:
        return _EXACT_ALIAS[t]

    for k, v in _KEYWORDS.items():
        if k in t:
            return v
    return None


def norm_alg_item(item: Mapping[str, Any]) -> Optional[str]:
    if not isinstance(item, Mapping):
        return None
    for key in ("name", "baseAlgname"):
        if key in item:
            result = norm_alg(item.get(key))
            if result:
                return result
    return None





