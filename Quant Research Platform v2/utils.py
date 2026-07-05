"""
v2 多因子选股系统 - 工具函数

日志,重试,缓存,百分位计算等通用工具.
"""

import logging
import pickle
import time
import functools
import hashlib
import json
from pathlib import Path
from typing import Callable, Optional, Any

import numpy as np
import pandas as pd

from config import (
    LOG_LEVEL, LOG_FORMAT, DATA_CACHE_DIR,
    REQUEST_DELAY, REQUEST_RETRIES, REQUEST_TIMEOUT,
)

# ============================================================
# 日志
# ============================================================

def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger,统一输出格式"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        logger.propagate = False
    return logger


logger = get_logger("utils")


# ============================================================
# API 重试装饰器
# ============================================================

def retry_on_error(
    max_retries: int = REQUEST_RETRIES,
    delay: float = REQUEST_DELAY,
    exceptions: tuple = (Exception,),
):
    """API 调用重试装饰器,处理网络波动"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    time.sleep(delay)  # 调用间隔
                    return result
                except exceptions as e:
                    last_error = e
                    logger.warning(
                        "%s 第 %d/%d 次失败: %s",
                        func.__name__, attempt, max_retries, str(e)[:120]
                    )
                    if attempt < max_retries:
                        time.sleep(delay * attempt)  # 递增等待
            logger.error("%s 全部 %d 次重试失败", func.__name__, max_retries)
            return None
        return wrapper
    return decorator


# ============================================================
# 缓存
# ============================================================

def _cache_key(*args, **kwargs) -> str:
    """生成缓存 key"""
    raw = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def save_cache(name: str, data: Any) -> None:
    """保存数据到缓存文件"""
    filepath = DATA_CACHE_DIR / f"{name}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    logger.debug("缓存已保存: %s", filepath)


def load_cache(name: str) -> Optional[Any]:
    """从缓存文件读取数据,不存在返回 None"""
    filepath = DATA_CACHE_DIR / f"{name}.pkl"
    if filepath.exists():
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        logger.debug("缓存命中: %s", filepath)
        return data
    return None


def clear_cache(name: Optional[str] = None) -> None:
    """清除缓存.name=None 清除全部"""
    if name:
        filepath = DATA_CACHE_DIR / f"{name}.pkl"
        filepath.unlink(missing_ok=True)
    else:
        for f in DATA_CACHE_DIR.glob("*.pkl"):
            f.unlink()


# ============================================================
# 数据预处理
# ============================================================

def clean_symbol(symbol: str) -> str:
    """统一股票代码格式: 600519.XSHG / 300750.XSHE"""
    symbol = symbol.strip().upper()
    if "." not in symbol:
        if symbol.startswith(("60", "68")):
            symbol = f"{symbol}.XSHG"
        else:
            symbol = f"{symbol}.XSHE"
    return symbol


def is_st_stock(name: str) -> bool:
    """判断是否为 ST / *ST 股票"""
    if not isinstance(name, str):
        return False
    return "ST" in name.upper()


def is_beijing_stock(symbol: str) -> bool:
    """判断是否为北交所股票(8开头: 83, 87, 88...)"""
    code = symbol.replace(".XSHG", "").replace(".XSHE", "").replace(".BJ", "")
    return code.startswith("8")


def is_star_stock(symbol: str) -> bool:
    """判断是否为科创板股票(688开头)"""
    code = symbol.replace(".XSHG", "").replace(".XSHE", "")
    return code.startswith("688")


def is_chinext_stock(symbol: str) -> bool:
    """判断是否为创业板股票(300/301开头)"""
    code = symbol.replace(".XSHG", "").replace(".XSHE", "")
    return code.startswith(("300", "301"))


# ============================================================
# 统计函数
# ============================================================

def safe_percentile_rank(series: pd.Series) -> pd.Series:
    """
    安全计算百分位排名(0-100),缺失值返回 NaN.
    返回: 100 = 最高值, 0 = 最低值
    """
    result = series.rank(pct=True) * 100.0
    result[series.isna()] = np.nan
    return result


def winsorize(series: pd.Series, lower_pct: float = 1.0, upper_pct: float = 99.0) -> pd.Series:
    """Winsorize 缩尾处理,减少极端值影响"""
    lower = series.quantile(lower_pct / 100.0)
    upper = series.quantile(upper_pct / 100.0)
    return series.clip(lower=lower, upper=upper)


def robust_zscore(series: pd.Series) -> pd.Series:
    """Robust Z-Score(用中位数和 MAD 替代均值和标准差)"""
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=series.index)
    return 0.6745 * (series - median) / mad


def weighted_score(
    scores: dict,
    weights: dict,
    fillna: float = 50.0,
) -> pd.Series:
    """
    加权合并多因子得分.

    参数:
        scores: {因子名: pd.Series(index=stock, value=0-100)}
        weights: {因子名: 权重}
        fillna: 缺失值填充

    返回:
        pd.Series: 加权总分 (0-100)
    """
    result = pd.Series(0.0, index=next(iter(scores.values())).index)
    weight_sum = 0.0

    for name, series in scores.items():
        w = weights.get(name, 0)
        if w == 0:
            continue
        filled = series.fillna(fillna)
        result += filled * w
        weight_sum += w

    if weight_sum > 0:
        result /= weight_sum

    return result


def normalize_to_range(series: pd.Series, lo: float = 0.0, hi: float = 100.0) -> pd.Series:
    """线性映射到 [lo, hi]"""
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series((lo + hi) / 2, index=series.index)
    return lo + (series - s_min) / (s_max - s_min) * (hi - lo)


# ============================================================
# 月度调仓日期生成
# ============================================================

def generate_rebalance_dates(
    start: str,
    end: str,
    freq: str = "monthly",
) -> list:
    """
    生成回测调仓日期列表(月末交易日).

    参数:
        start: 开始日期 "YYYY-MM-DD"
        end: 结束日期 "YYYY-MM-DD"
        freq: "monthly" | "weekly"

    返回:
        list of str
    """
    dates = pd.date_range(start, end, freq="ME" if freq == "monthly" else "W")
    return [d.strftime("%Y-%m-%d") for d in dates]


# ============================================================
# 回测收益计算
# ============================================================

def calc_portfolio_return(
    selected_stocks: list,
    price_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    weights: Optional[pd.Series] = None,
) -> float:
    """
    计算组合在持有期的收益率(等权或自定义权重).

    参数:
        selected_stocks: 选中的股票代码列表
        price_data: DataFrame, columns=股票代码, index=日期, values=收盘价
        start_date: 持有期起始
        end_date: 持有期结束
        weights: 权重 Series(等权为 None)

    返回:
        组合收益率
    """
    if weights is None:
        weights = pd.Series(1.0 / len(selected_stocks), index=selected_stocks)

    returns = []
    for stock in selected_stocks:
        if stock not in price_data.columns:
            continue
        prices = price_data[stock].loc[start_date:end_date]
        if len(prices) < 2:
            continue
        r = (prices.iloc[-1] / prices.iloc[0]) - 1
        returns.append(r * weights.get(stock, 0))

    return sum(returns) if returns else 0.0


# ============================================================
# 进度条
# ============================================================

def progress_bar(current: int, total: int, label: str = "", width: int = 40) -> str:
    """简单的文本进度条"""
    pct = current / total if total > 0 else 1.0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"\r{label} [{bar}] {current}/{total} ({pct:.1%})"
