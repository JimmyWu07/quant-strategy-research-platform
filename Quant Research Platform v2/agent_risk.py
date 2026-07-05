"""
v2 多因子选股系统 - AgentRisk 风控模块

职责:
  - 行业过热检测: PE 分位 > 80%,近1月涨幅 > 15%
  - 个股估值风险: PE/PB 分位 > 85%,PEG > 2.0
  - 扣分制: 每次触发扣 5 分,封顶 25 分

设计理念:
  风控不直接剔除股票,而是扣分.过热/高估的票需要
  更强的因子得分来弥补,保持了组合的灵活性.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    RISK_INDUSTRY_PE_PCT, RISK_INDUSTRY_1M_RETURN,
    RISK_STOCK_PE_PCT, RISK_STOCK_PB_PCT, RISK_PEG_THRESHOLD,
    RISK_PENALTY_PER_TRIGGER, RISK_PENALTY_MAX,
)
from utils import get_logger, safe_percentile_rank

logger = get_logger("AgentRisk")


class AgentRisk:
    """
    风控 Agent.

    用法:
        agent = AgentRisk(industry_map)
        risk_deductions = agent.assess(
            universe_df,
            price_df,
            pe_pb_pct_df,
            financial_df,
        )
        # risk_deductions: pd.Series(index=symbol, value=扣分值, 负值)
    """

    def __init__(self, industry_map: Optional[Dict[str, str]] = None):
        self.industry_map = industry_map or {}

    # ============================================================
    # 主入口
    # ============================================================

    def assess(
        self,
        universe: pd.DataFrame,
        price_df: pd.DataFrame,
        pe_pb_pct_df: Optional[pd.DataFrame] = None,
        financial_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        评估所有风控指标,计算扣分.

        返回:
            DataFrame, columns=[
                symbol,
                risk_industry_pe,      # 行业PE过热扣分
                risk_industry_return,   # 行业涨幅过热扣分
                risk_stock_pe,          # 个股PE过高扣分
                risk_stock_pb,          # 个股PB过高扣分
                risk_peg,               # PEG过高扣分
                risk_total,             # 总扣分(负值)
            ]
        """
        symbols = universe["symbol"].tolist() if "symbol" in universe.columns else list(universe.index)

        # 转换为以 symbol 为索引
        if "symbol" in universe.columns:
            universe_idx = universe.set_index("symbol")
        else:
            universe_idx = universe

        # 初始化扣分
        penalties = pd.DataFrame(0, index=pd.Index(symbols, name="symbol"), columns=[
            "risk_industry_pe", "risk_industry_return",
            "risk_stock_pe", "risk_stock_pb", "risk_peg",
        ])

        # === 1. 行业 PE 过热 ===
        penalties["risk_industry_pe"] = self._check_industry_pe_overheat(
            universe_idx, pe_pb_pct_df
        )

        # === 2. 行业近1月涨幅过热 ===
        penalties["risk_industry_return"] = self._check_industry_return_overheat(
            universe_idx, price_df
        )

        # === 3. 个股 PE 分位过高 ===
        penalties["risk_stock_pe"] = self._check_stock_pe_overheat(
            universe_idx, pe_pb_pct_df
        )

        # === 4. 个股 PB 分位过高 ===
        penalties["risk_stock_pb"] = self._check_stock_pb_overheat(
            universe_idx, pe_pb_pct_df
        )

        # === 5. PEG 过高 ===
        penalties["risk_peg"] = self._check_peg_overheat(
            universe_idx, financial_df
        )

        # === 汇总 ===
        penalties["risk_total"] = penalties.sum(axis=1).clip(upper=RISK_PENALTY_MAX)

        if verbose:
            triggered = (penalties["risk_total"] > 0).sum()
            avg_penalty = penalties["risk_total"][penalties["risk_total"] > 0].mean()
            logger.info(
                "风控评估完成: %d/%d 触发扣分, 平均扣分 %.1f",
                triggered, len(symbols), avg_penalty if not np.isnan(avg_penalty) else 0
            )

        return penalties

    # ============================================================
    # 行业级风控
    # ============================================================

    def _check_industry_pe_overheat(
        self,
        universe: pd.DataFrame,
        pe_pb_pct_df: Optional[pd.DataFrame],
    ) -> pd.Series:
        """
        行业 PE 在近5年历史中 > 80% 分位 → 行业过热.
        对该行业内所有股票扣 RISK_PENALTY_PER_TRIGGER 分.
        """
        result = pd.Series(0, index=universe.index, dtype=float)

        if pe_pb_pct_df is None or "pe_pct" not in pe_pb_pct_df.columns:
            return result

        # 计算每个行业的平均 PE 分位
        if not self.industry_map:
            return result

        industry_series = universe.index.map(self.industry_map)
        industry_pe_pct = pd.DataFrame({
            "industry": industry_series,
            "pe_pct": pe_pb_pct_df["pe_pct"].reindex(universe.index),
        }).dropna()

        if industry_pe_pct.empty:
            return result

        # 每个行业的 PE 分位中位数
        industry_median_pe = industry_pe_pct.groupby("industry")["pe_pct"].median()

        # 过热的行业
        overheated_industries = industry_median_pe[industry_median_pe > RISK_INDUSTRY_PE_PCT].index

        for ind in overheated_industries:
            mask = industry_series == ind
            result[mask] = RISK_PENALTY_PER_TRIGGER

        if len(overheated_industries) > 0:
            logger.debug("行业PE过热 (%d个): %s", len(overheated_industries),
                         ", ".join(overheated_industries[:5]))

        return result

    def _check_industry_return_overheat(
        self,
        universe: pd.DataFrame,
        price_df: pd.DataFrame,
    ) -> pd.Series:
        """
        行业近1月涨幅 > 15% → 短期过热.
        """
        result = pd.Series(0, index=universe.index, dtype=float)

        if price_df is None or price_df.empty or not self.industry_map:
            return result

        # 计算每只股票的近1月收益率
        if len(price_df) < 21:
            return result

        available = [s for s in universe.index if s in price_df.columns]
        if not available:
            return result

        ret_1m = (price_df[available].iloc[-1] / price_df[available].iloc[-21] - 1) * 100

        # 按行业汇总平均涨幅
        industry_series = pd.Series(
            universe.index.map(self.industry_map),
            index=universe.index,
        )
        ret_with_ind = pd.DataFrame({
            "industry": industry_series.reindex(ret_1m.index),
            "ret_1m": ret_1m.values,
        }, index=ret_1m.index).dropna()

        if ret_with_ind.empty:
            return result

        industry_avg_ret = ret_with_ind.groupby("industry")["ret_1m"].mean()
        overheated = industry_avg_ret[industry_avg_ret > RISK_INDUSTRY_1M_RETURN].index

        for ind in overheated:
            mask = industry_series == ind
            result[mask] = RISK_PENALTY_PER_TRIGGER

        return result

    # ============================================================
    # 个股级风控
    # ============================================================

    def _check_stock_pe_overheat(
        self,
        universe: pd.DataFrame,
        pe_pb_pct_df: Optional[pd.DataFrame],
    ) -> pd.Series:
        """
        个股 PE 在自身 5 年历史 > 85% 分位 → 高估风险.
        """
        result = pd.Series(0, index=universe.index, dtype=float)

        if pe_pb_pct_df is None or "pe_pct" not in pe_pb_pct_df.columns:
            return result

        pe_pct = pe_pb_pct_df["pe_pct"].reindex(universe.index)
        result[pe_pct > RISK_STOCK_PE_PCT] = RISK_PENALTY_PER_TRIGGER

        return result

    def _check_stock_pb_overheat(
        self,
        universe: pd.DataFrame,
        pe_pb_pct_df: Optional[pd.DataFrame],
    ) -> pd.Series:
        """
        个股 PB 在自身 5 年历史 > 85% 分位 → 高估风险.
        """
        result = pd.Series(0, index=universe.index, dtype=float)

        if pe_pb_pct_df is None or "pb_pct" not in pe_pb_pct_df.columns:
            return result

        pb_pct = pe_pb_pct_df["pb_pct"].reindex(universe.index)
        result[pb_pct > RISK_STOCK_PB_PCT] = RISK_PENALTY_PER_TRIGGER

        return result

    def _check_peg_overheat(
        self,
        universe: pd.DataFrame,
        financial_df: Optional[pd.DataFrame],
    ) -> pd.Series:
        """
        PEG > 2.0 → 估值相对成长性过高.
        PEG = PE / 净利润增速(%).增速为负的票 PEG 无意义,不扣 PEG 分.
        """
        result = pd.Series(0, index=universe.index, dtype=float)

        if financial_df is None or "peg" not in financial_df.columns:
            return result

        peg = financial_df["peg"].reindex(universe.index)
        # PEG 为负(亏损或增速为负)的不扣 PEG 分,但 PE 分位可能已经扣了
        result[(peg > RISK_PEG_THRESHOLD) & (peg > 0)] = RISK_PENALTY_PER_TRIGGER

        return result

    # ============================================================
    # 诊断工具
    # ============================================================

    def get_risk_summary(self, penalties: pd.DataFrame) -> dict:
        """风控诊断摘要"""
        total = len(penalties)
        if total == 0:
            return {}

        return {
            "stocks_assessed": total,
            "industry_pe_triggered": int((penalties["risk_industry_pe"] > 0).sum()),
            "industry_return_triggered": int((penalties["risk_industry_return"] > 0).sum()),
            "stock_pe_triggered": int((penalties["risk_stock_pe"] > 0).sum()),
            "stock_pb_triggered": int((penalties["risk_stock_pb"] > 0).sum()),
            "peg_triggered": int((penalties["risk_peg"] > 0).sum()),
            "avg_penalty": round(float(penalties["risk_total"].mean()), 2),
            "max_penalty": float(penalties["risk_total"].max()),
            "zero_penalty_pct": round(float((penalties["risk_total"] == 0).mean()) * 100, 1),
        }


# ============================================================
# 模块自检
# ============================================================

if __name__ == "__main__":
    print("AgentRisk 模块加载成功")

    # 模拟测试
    symbols = ["600519.XSHG", "000858.XSHE", "300750.XSHE",
               "000001.XSHE", "600036.XSHG"]
    fake_universe = pd.DataFrame({
        "symbol": symbols, "pe": [30, 25, 50, 8, 6], "pb": [8, 5, 12, 0.8, 0.7],
    }).set_index("symbol")
    fake_pe_pb = pd.DataFrame({
        "pe_pct": [70, 88, 55, 30, 90],
        "pb_pct": [75, 90, 60, 25, 92],
    }, index=symbols)
    fake_industry = {
        "600519.XSHG": "白酒", "000858.XSHE": "白酒",
        "300750.XSHE": "电池", "000001.XSHE": "银行", "600036.XSHG": "银行",
    }

    agent = AgentRisk(fake_industry)
    penalties = agent.assess(fake_universe, None, fake_pe_pb, None)
    print(penalties.to_string())
    print("\n风控摘要:", agent.get_risk_summary(penalties))
