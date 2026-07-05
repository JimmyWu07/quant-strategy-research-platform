"""
v2 多因子选股系统 - AgentMacro 宏观仓位模块

职责:
  - 计算宏观景气度评分: PMI(40%) + M2增速(30%) + 社融增速(30%)
  - 映射到仓位系数: 30% ~ 80%

逻辑:
  景气度分位 < 30%  → 仓位 = 30%(防御)
  景气度分位 30~70% → 仓位 = 线性插值 30%~80%
  景气度分位 > 70%  → 仓位 = 80%(积极,但不过满)

更新频率: 每月调仓时重新评估
"""

from typing import Optional

import numpy as np
import pandas as pd

from config import (
    MACRO_POSITION_MIN, MACRO_POSITION_MAX,
    MACRO_WEIGHT_PMI, MACRO_WEIGHT_M2, MACRO_WEIGHT_SOCIAL_FIN,
    MACRO_PCT_LOW, MACRO_PCT_HIGH,
)
from utils import get_logger

logger = get_logger("AgentMacro")


class AgentMacro:
    """
    宏观仓位 Agent.

    用法:
        agent = AgentMacro()
        macro_data = agent.evaluate(macro_raw_dict)
        position_pct = agent.get_position_size(macro_data)
    """

    def __init__(self):
        pass

    # ============================================================
    # 主入口
    # ============================================================

    def evaluate(self, macro_raw: dict) -> dict:
        """
        评估宏观景气度.

        参数:
            macro_raw: AgentData.get_macro_data() 返回的字典
                       {pmi, pmi_pct, m2_yoy, m2_yoy_pct, social_fin_yoy, social_fin_yoy_pct}

        返回:
            {
                "pmi": float,
                "m2_yoy": float,
                "social_fin_yoy": float,
                "macro_score": float,      # 综合景气度 (0-100)
                "position_ratio": float,   # 建议仓位比例 (0.30~0.80)
                "regime": str,             # 宏观状态: "防御"/"中性"/"积极"
            }
        """
        if not macro_raw:
            logger.warning("宏观数据为空,使用默认中性仓位 50%")
            return {
                "macro_score": 50.0,
                "position_ratio": 0.50,
                "regime": "中性(数据缺失)",
            }

        # === 计算各分项得分 (0-100) ===
        scores = {}
        weights = {
            "pmi": MACRO_WEIGHT_PMI,
            "m2": MACRO_WEIGHT_M2,
            "social_fin": MACRO_WEIGHT_SOCIAL_FIN,
        }

        # PMI 分位(直接使用历史分位值)
        pmi_pct = macro_raw.get("pmi_pct", 50.0)
        if not np.isnan(pmi_pct):
            scores["pmi"] = float(pmi_pct)
        else:
            scores["pmi"] = 50.0

        # M2 增速分位
        m2_pct = macro_raw.get("m2_yoy_pct", 50.0)
        if not np.isnan(m2_pct):
            scores["m2"] = float(m2_pct)
        else:
            scores["m2"] = 50.0

        # 社融增速分位
        sf_pct = macro_raw.get("social_fin_yoy_pct", 50.0)
        if not np.isnan(sf_pct):
            scores["social_fin"] = float(sf_pct)
        else:
            scores["social_fin"] = 50.0

        # === 加权综合 ===
        macro_score = sum(scores[k] * weights[k] for k in weights)
        macro_score = round(macro_score, 2)

        # === 映射仓位 ===
        position_ratio = self._score_to_position(macro_score)
        regime = self._classify_regime(macro_score)

        logger.info(
            "宏观评估: PMI分位=%.0f, M2分位=%.0f, 社融分位=%.0f → "
            "景气度=%.1f, 仓位=%.0f%%, 状态=%s",
            scores["pmi"], scores["m2"], scores["social_fin"],
            macro_score, position_ratio * 100, regime,
        )

        return {
            "pmi": macro_raw.get("pmi"),
            "m2_yoy": macro_raw.get("m2_yoy"),
            "social_fin_yoy": macro_raw.get("social_fin_yoy"),
            "macro_score": macro_score,
            "position_ratio": position_ratio,
            "regime": regime,
            "detail_scores": scores,
        }

    # ============================================================
    # 仓位映射
    # ============================================================

    def _score_to_position(self, macro_score: float) -> float:
        """
        景气度分位 → 仓位系数.

        分位 < 30%: 仓位 = 30%(硬地板)
        分位 30% ~ 70%: 仓位 = 30% + (分位 - 30) / 40 * 50%
        分位 > 70%: 仓位 = 80%(天花板)
        """
        if macro_score <= MACRO_PCT_LOW * 100:
            return MACRO_POSITION_MIN
        elif macro_score >= MACRO_PCT_HIGH * 100:
            return MACRO_POSITION_MAX
        else:
            # 线性插值
            ratio = (macro_score / 100 - MACRO_PCT_LOW) / (MACRO_PCT_HIGH - MACRO_PCT_LOW)
            return round(MACRO_POSITION_MIN + ratio * (MACRO_POSITION_MAX - MACRO_POSITION_MIN), 4)

    def _classify_regime(self, macro_score: float) -> str:
        """宏观状态分类"""
        if macro_score < MACRO_PCT_LOW * 100:
            return "防御"
        elif macro_score < MACRO_PCT_HIGH * 100:
            return "中性"
        else:
            return "积极"

    # ============================================================
    # 仓位调整
    # ============================================================

    def get_position_size(self, macro_result: dict) -> float:
        """从评估结果中提取仓位比例"""
        return macro_result.get("position_ratio", 0.50)

    def apply_position_to_portfolio(
        self,
        portfolio_weights: pd.Series,
        position_ratio: float,
    ) -> pd.Series:
        """
        将宏观仓位应用于组合权重.

        参数:
            portfolio_weights: 原始等权权重(总和=1.0)
            position_ratio: 宏观建议仓位(0.30~0.80)

        返回:
            调整后权重(总和=position_ratio),剩余为现金
        """
        return portfolio_weights * position_ratio


# ============================================================
# 模块自检
# ============================================================

if __name__ == "__main__":
    print("AgentMacro 模块加载成功")

    agent = AgentMacro()

    # 模拟宏观数据
    fake_data = {
        "pmi": 50.5,
        "pmi_pct": 65.0,       # PMI 处于近5年 65% 分位
        "m2_yoy": 10.5,
        "m2_yoy_pct": 55.0,    # M2 增速处于 55% 分位
        "social_fin_yoy": 8.0,
        "social_fin_yoy_pct": 45.0,  # 社融增速处于 45% 分位
    }

    result = agent.evaluate(fake_data)
    print(f"\n宏观评估结果:")
    for k, v in result.items():
        if k != "detail_scores":
            print(f"  {k}: {v}")

    # 仓位映射演示
    print("\n仓位映射曲线:")
    for score in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        pos = agent._score_to_position(score)
        bar = "█" * int(pos * 50)
        print(f"  景气度 {score:2d} → 仓位 {pos:.0%} {bar}")
