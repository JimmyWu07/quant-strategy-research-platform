"""
v2 多因子选股系统 - AgentPortfolio 组合构建模块

职责:
  - 合并因子得分与风控扣分: 最终得分 = 因子总分 - 风控扣分
  - 精选 Top 50 股票
  - 应用组合约束: 单票 ≤ 5%,单行业 ≤ 20%
  - 等权或因子加权

输出:
  - 精选股票池(含各因子明细)
  - 组合权重向量
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    TOP_N_STOCKS, MAX_WEIGHT_PER_STOCK, MAX_INDUSTRY_WEIGHT,
)
from utils import get_logger

logger = get_logger("AgentPortfolio")


class AgentPortfolio:
    """
    组合构建 Agent.

    用法:
        agent = AgentPortfolio(industry_map)
        portfolio = agent.build(
            factor_scores,    # AgentFactor 输出的综合得分
            risk_penalties,   # AgentRisk 输出的风控扣分
        )
        # portfolio["selected"]: Top 50 股票列表
        # portfolio["weights"]: 组合权重
    """

    def __init__(self, industry_map: Optional[Dict[str, str]] = None):
        self.industry_map = industry_map or {}

    # ============================================================
    # 主入口
    # ============================================================

    def build(
        self,
        factor_scores: pd.DataFrame,
        risk_penalties: pd.DataFrame,
        top_n: int = TOP_N_STOCKS,
        verbose: bool = True,
    ) -> dict:
        """
        构建精选组合.

        参数:
            factor_scores: AgentFactor.score_all() 输出,index=股票代码
                           columns: [score_val, score_qual, score_mom, score_vol, total_score]
            risk_penalties: AgentRisk.assess() 输出,index=股票代码
                            columns: [risk_total, ...]
            top_n: 精选数量
            verbose: 是否输出详细日志

        返回:
            {
                "selected": DataFrame,    # Top N 股票及其得分明细
                "all_ranked": DataFrame,  # 所有候选股票排名
                "weights": pd.Series,     # Top N 权重
                "stats": dict,            # 组合统计
            }
        """
        # === 1. 合并得分 ===
        combined = self._merge_scores(factor_scores, risk_penalties)

        if combined.empty:
            logger.error("无候选股票可构建组合")
            return {"selected": pd.DataFrame(), "all_ranked": pd.DataFrame(),
                    "weights": pd.Series(dtype=float), "stats": {}}

        # === 2. 排名 ===
        ranked = combined.sort_values("final_score", ascending=False)

        # === 3. 应用行业约束选取 Top N ===
        selected = self._select_with_constraints(ranked, top_n)

        # === 4. 计算权重 ===
        weights = self._calculate_weights(selected)

        # === 5. 统计 ===
        stats = self._compute_stats(selected, weights, ranked)

        if verbose:
            logger.info(
                "组合构建完成: Top %d/%d, 覆盖 %d 个行业",
                len(selected), len(ranked),
                selected["industry"].nunique() if "industry" in selected.columns else 0,
            )
            logger.info("组合统计: %s", stats)

        return {
            "selected": selected,
            "all_ranked": ranked,
            "weights": weights,
            "stats": stats,
        }

    # ============================================================
    # 得分合并
    # ============================================================

    def _merge_scores(
        self,
        factor_scores: pd.DataFrame,
        risk_penalties: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        最终得分 = 因子总分 - 风控扣分.

        注意:权重已在 config 中分配,这里直接做减法.
        """
        if factor_scores.empty:
            return pd.DataFrame()

        # 确保索引对齐
        common_idx = factor_scores.index

        result = factor_scores.copy()

        # 归一化因子总分到 0-100
        if "total_score" in result.columns:
            raw_total = result["total_score"]
            if raw_total.max() > raw_total.min():
                result["total_score_norm"] = (
                    (raw_total - raw_total.min()) /
                    (raw_total.max() - raw_total.min()) * 100
                )
            else:
                result["total_score_norm"] = 50.0
        else:
            result["total_score_norm"] = 50.0

        # 减去风控扣分
        if risk_penalties is not None and not risk_penalties.empty and "risk_total" in risk_penalties.columns:
            risk = risk_penalties["risk_total"].reindex(common_idx).fillna(0)
            result["risk_deduction"] = risk.values
            result["final_score"] = result["total_score_norm"] - risk.values
        else:
            result["risk_deduction"] = 0
            result["final_score"] = result["total_score_norm"]

        # 确保分数不小于 0
        result["final_score"] = result["final_score"].clip(lower=0)

        # 排名
        result["rank"] = result["final_score"].rank(ascending=False, method="first").astype(int)

        # 添加行业
        if self.industry_map:
            result["industry"] = result.index.map(self.industry_map)
        else:
            result["industry"] = "未知"

        return result

    # ============================================================
    # 行业约束选取
    # ============================================================

    def _select_with_constraints(
        self,
        ranked: pd.DataFrame,
        top_n: int,
    ) -> pd.DataFrame:
        """
        按得分降序选取 Top N.

        若有行业数据,控制单行业权重不超过 MAX_INDUSTRY_WEIGHT;
        若无行业数据(全为"未知"),直接取 Top N.
        """
        # 检查是否有有效的行业数据
        has_valid_industry = (
            "industry" in ranked.columns and
            ranked["industry"].nunique() > 1 and
            not (ranked["industry"] == "未知").all()
        )

        if not has_valid_industry:
            # 无行业数据:直接取 Top N
            result = ranked.head(top_n).copy()
            result["rank"] = range(1, len(result) + 1)
            return result

        # 有行业数据:按行业约束选取
        max_per_industry = max(1, int(top_n * MAX_INDUSTRY_WEIGHT))

        selected = []
        industry_count: Dict[str, int] = {}

        for sym, row in ranked.iterrows():
            if len(selected) >= top_n:
                break

            ind = row.get("industry", "未知")
            current_count = industry_count.get(ind, 0)

            if current_count < max_per_industry:
                selected.append(sym)
                industry_count[ind] = current_count + 1

        result = ranked.loc[selected].copy()
        result["rank"] = range(1, len(result) + 1)
        return result

    # ============================================================
    # 权重计算
    # ============================================================

    def _calculate_weights(self, selected: pd.DataFrame) -> pd.Series:
        """
        计算组合权重.

        默认等权,单票上限 5%.
        若某票超限,超额部分按比例分配给其余股票.
        """
        n = len(selected)
        if n == 0:
            return pd.Series(dtype=float)

        weights = pd.Series(1.0 / n, index=selected.index, name="weight")

        # 单票上限约束
        for _ in range(5):  # 迭代调整
            over = weights[weights > MAX_WEIGHT_PER_STOCK]
            if over.empty:
                break
            excess = (over - MAX_WEIGHT_PER_STOCK).sum()
            weights[over.index] = MAX_WEIGHT_PER_STOCK
            under_idx = weights[weights < MAX_WEIGHT_PER_STOCK].index
            if len(under_idx) > 0:
                weights[under_idx] += excess / len(under_idx)

        # 归一化
        weights = weights / weights.sum()
        return weights.round(6)

    # ============================================================
    # 统计
    # ============================================================

    def _compute_stats(
        self,
        selected: pd.DataFrame,
        weights: pd.Series,
        all_ranked: pd.DataFrame,
    ) -> dict:
        """计算组合统计指标"""
        if selected.empty:
            return {}

        industry_counts = selected["industry"].value_counts() if "industry" in selected.columns else pd.Series()
        industry_weights = industry_counts / len(selected)

        return {
            "n_stocks": len(selected),
            "n_industries": selected["industry"].nunique() if "industry" in selected.columns else 0,
            "avg_final_score": round(float(selected["final_score"].mean()), 2),
            "median_final_score": round(float(selected["final_score"].median()), 2),
            "avg_factor_score": round(float(selected["total_score_norm"].mean()), 2) if "total_score_norm" in selected.columns else None,
            "avg_risk_deduction": round(float(selected["risk_deduction"].mean()), 2) if "risk_deduction" in selected.columns else 0,
            "max_industry_weight": round(float(industry_weights.max()), 3) if not industry_weights.empty else 0,
            "top_industry": industry_weights.index[0] if not industry_weights.empty else None,
            "concentration_hhi": round(float((weights ** 2).sum()), 4) if len(weights) > 0 else 0,
        }

    # ============================================================
    # 诊断工具
    # ============================================================

    def get_portfolio_summary(self, build_result: dict) -> str:
        """生成组合摘要文本"""
        stats = build_result.get("stats", {})
        selected = build_result.get("selected", pd.DataFrame())

        lines = [
            "=" * 60,
            "  精选组合摘要",
            "=" * 60,
            f"  股票数量: {stats.get('n_stocks', 0)}",
            f"  覆盖行业: {stats.get('n_industries', 0)}",
            f"  平均得分: {stats.get('avg_final_score', 0)}",
            f"  平均风控扣分: {stats.get('avg_risk_deduction', 0)}",
            f"  最大行业占比: {stats.get('max_industry_weight', 0):.1%}",
            f"  HHI 集中度: {stats.get('concentration_hhi', 0)}",
            "",
            "  Top 10:",
        ]

        if not selected.empty:
            top10 = selected.head(10)
            for sym, row in top10.iterrows():
                ind = row.get("industry", "?")
                score = row.get("final_score", 0)
                lines.append(f"    {sym} | {ind:12s} | 得分: {score:.1f}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# 模块自检
# ============================================================

if __name__ == "__main__":
    print("AgentPortfolio 模块加载成功")

    # 模拟数据
    np.random.seed(42)
    symbols = [f"stock_{i}" for i in range(100)]
    fake_scores = pd.DataFrame({
        "score_val": np.random.uniform(0, 100, 100),
        "score_qual": np.random.uniform(0, 100, 100),
        "score_mom": np.random.uniform(0, 100, 100),
        "score_vol": np.random.uniform(0, 100, 100),
        "total_score": np.random.uniform(30, 80, 100),
    }, index=symbols)

    fake_risk = pd.DataFrame({
        "risk_total": np.random.choice([0, 5, 10, 15], 100, p=[0.5, 0.2, 0.2, 0.1]),
    }, index=symbols)

    industries = ["半导体", "白酒", "电池", "银行", "医药", "软件", "机械", "军工"]
    fake_industry = {s: np.random.choice(industries) for s in symbols}

    agent = AgentPortfolio(fake_industry)
    result = agent.build(fake_scores, fake_risk)
    print(agent.get_portfolio_summary(result))
