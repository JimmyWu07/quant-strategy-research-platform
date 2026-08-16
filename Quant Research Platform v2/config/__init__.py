"""
v2 多因子选股系统 - 配置包

统一导出 settings 和 utils 中的所有配置与工具函数,
使其他模块可通过 `from config import XXX` 直接引用.
"""

from config.settings import *  # noqa: F401,F403
from config.utils import *      # noqa: F401,F403
