"""
v2 多因子选股系统 - AgentFactor 因子打分模块

职责:
  - 估值因子: PE 历史分位,PB 历史分位(越低越好 → 得分越高)
  - 质量因子: ROE_TTM,毛利率_TTM,经营现金流/净利润(越高越好)
  - 动量因子: 1月收益率,3月收益率(越高越好)
  - 波动反转: 20日波动率,5日反转(波动正向,反转负向)

打分方法:
  1. 每个因子在申万二级行业内做百分位排名 (0-100)
  2. 对负向因子(PE/PB分位,反转)取反: 100 - 排名
  3. 按权重加总各因子得分
  4. 返回每只股票的综合得分 (0-100)
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    WEIGHT_VALUATION, WEIGHT_QUALITY, WEIGHT_MOMENTUM, WEIGHT_VOLATILITY,
    PE_PERCENTILE_HIGH_CAP, PB_PERCENTILE_HIGH_CAP,
    MOMENTUM_1M, MOMENTUM_3M,
    VOLATILITY_20D, REVERSAL_5D,
)
from utils import (
    get_logger, safe_percentile_rank, winsorize, weighted_score,
)

logger = get_logger("AgentFactor")


class AgentFactor:
    """
    因子打分 Agent.

    用法:
        agent = AgentFactor(industry_map)
        scores = agent.score_all(
            universe_df,      # 股票池(含 close, pe, pb 等)
            price_df,         # 历史价格 DataFrame
            financial_df,     # 财务数据 DataFrame
            pe_pb_pct_df,     # PE/PB 历史分位
        )
        # scores: pd.Series(index=symbol, value=综合得分 0-100)
    """

    def __init__(self, industry_map: Optional[Dict[str, str]] = None):
        """
        参数:
            industry_map: {symbol: industry_name},行业分类映射.
                          提供后,因子排名在行业内进行(行业中性化);
                          不提供则全市场排名.
        """
        self.industry_map = industry_map or {}

    # ============================================================
    # 主入口
    # ============================================================

    def score_all(
        self,
        universe: pd.DataFrame,
        price_df: pd.DataFrame,
        financial_df: pd.DataFrame,
        pe_pb_pct_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        计算所有因子得分并加权汇总.

        参数:
            universe: 候选股票池(含 symbol, pe, pb 等字段)
            price_df: 历史收盘价,index=日期, columns=股票代码
            financial_df: 财务数据,index=股票代码, columns=[roe, gross_margin, cfo_to_np, np_yoy]
            pe_pb_pct_df: PE/PB 历史分位,index=股票代码, columns=[pe_pct, pb_pct]
            verbose: 是否输出详细日志

        返回:
            DataFrame, columns = [
                symbol,
                score_val, score_qual, score_mom, score_vol,  # 各模块得分
                total_score,                                    # 综合得分
            ]
        """
        symbols = universe["symbol"].tolist() if "symbol" in universe.columns else list(universe.index)

        # === 1. 估值因子 ===
        val_scores = self._score_valuation(universe, pe_pb_pct_df)
        if verbose:
            logger.info("估值因子完成: %d 只股票", val_scores["score_val"].notna().sum())

        # === 2. 质量因子 ===
        qual_scores = self._score_quality(financial_df)
        if verbose:
            logger.info("质量因子完成: %d 只股票", qual_scores["score_qual"].notna().sum())

        # === 3. 动量因子 ===
        mom_scores = self._score_momentum(price_df, symbols)
        if verbose:
            logger.info("动量因子完成: %d 只股票", mom_scores["score_mom"].notna().sum())

        # === 4. 波动反转因子 ===
        vol_scores = self._score_volatility_reversal(price_df, symbols)
        if verbose:
            logger.info("波动反转因子完成: %d 只股票", vol_scores["score_vol"].notna().sum())

        # === 5. 加权汇总 ===
        all_scores = {
            "score_val": val_scores.get("score_val", pd.Series(dtype=float)),
            "score_qual": qual_scores.get("score_qual", pd.Series(dtype=float)),
            "score_mom": mom_scores.get("score_mom", pd.Series(dtype=float)),
            "score_vol": vol_scores.get("score_vol", pd.Series(dtype=float)),
        }

        module_weights = {
            "score_val": WEIGHT_VALUATION,
            "score_qual": WEIGHT_QUALITY,
            "score_mom": WEIGHT_MOMENTUM,
            "score_vol": WEIGHT_VOLATILITY,
        }

        # 为确保权重按模块分配,每个模块得分为 0-100,再加权
        total = pd.Series(0.0, index=pd.Index(symbols, name="symbol"))

        result_parts = {"symbol": symbols}
        for key, series in all_scores.items():
            w = module_weights[key]
            filled = series.reindex(symbols).fillna(50.0)  # 缺失值给中性分 50
            result_parts[key] = filled.values
            total += filled * w

        # 归一化到 0-100
        result_parts["total_score"] = total.values

        result = pd.DataFrame(result_parts)
        result = result.set_index("symbol")

        if verbose:
            top5 = result.nlargest(5, "total_score")
            logger.info("综合打分完成.Top 5:\n%s", top5[["total_score"]].to_string())

        return result

    # ============================================================
    # 估值因子(越低越好 → 反向排名)
    # ============================================================

    def _score_valuation(
        self,
        universe: pd.DataFrame,
        pe_pb_pct_df: Optional[pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """
        PE/PB 历史分位因子.
        当前分位越低 → 得分越高(100 - 分位).
        """
        symbols = universe["symbol"] if "symbol" in universe.columns else universe.index
        symbols = pd.Index(symbols, name="symbol")

        # --- PE 分位 ---
        if pe_pb_pct_df is not None and "pe_pct" in pe_pb_pct_df.columns:
            pe_pct = pe_pb_pct_df["pe_pct"].reindex(symbols)
        else:
            # 回退:用当前 PE 在全市场中的排名近似
            if "pe" in universe.columns:
                pe_series = universe.set_index("symbol")["pe"] if "symbol" in universe.columns else universe["pe"]
                pe_pct = safe_percentile_rank(pe_series)
            else:
                pe_pct = pd.Series(50.0, index=symbols)

        # 截断极端值
        pe_pct = pe_pct.clip(upper=PE_PERCENTILE_HIGH_CAP)
        # 反向:低分位 = 好
        pe_score = 100.0 - pe_pct

        # --- PB 分位 ---
        if pe_pb_pct_df is not None and "pb_pct" in pe_pb_pct_df.columns:
            pb_pct = pe_pb_pct_df["pb_pct"].reindex(symbols)
        else:
            if "pb" in universe.columns:
                pb_series = universe.set_index("symbol")["pb"] if "symbol" in universe.columns else universe["pb"]
                pb_pct = safe_percentile_rank(pb_series)
            else:
                pb_pct = pd.Series(50.0, index=symbols)

        pb_pct = pb_pct.clip(upper=PB_PERCENTILE_HIGH_CAP)
        pb_score = 100.0 - pb_pct

        # --- 行业内排名修正 ---
        pe_score = self._within_industry_rank(pe_score, reverse=False)
        pb_score = self._within_industry_rank(pb_score, reverse=False)

        # --- 合并:PE 和 PB 等权 ---
        score_val = (pe_score.fillna(50.0) + pb_score.fillna(50.0)) / 2.0

        return {
            "score_val": score_val,
            "pe_pct": pe_pct,
            "pb_pct": pb_pct,
        }

    # ============================================================
    # 质量因子(越高越好 → 正向排名)
    # ============================================================

    def _score_quality(self, financial_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        ROE_TTM,毛利率_TTM,经营现金流/净利润.
        """
        symbols = financial_df.index

        # --- ROE ---
        if "roe" in financial_df.columns:
            roe = winsorize(financial_df["roe"].astype(float))
            roe_score = safe_percentile_rank(roe)
        else:
            roe_score = pd.Series(50.0, index=symbols)

        # --- 毛利率 ---
        if "gross_margin" in financial_df.columns:
            gm = winsorize(financial_df["gross_margin"].astype(float))
            gm_score = safe_percentile_rank(gm)
        else:
            gm_score = pd.Series(50.0, index=symbols)

        # --- 经营现金流/净利润 ---
        if "cfo_to_np" in financial_df.columns:
            cfo = financial_df["cfo_to_np"].astype(float).clip(lower=-5, upper=5)
            # > 1 最好,< 0 最差
            cfo_score = safe_percentile_rank(cfo)
        else:
            cfo_score = pd.Series(50.0, index=symbols)

        # --- 行业内排名修正 ---
        roe_score = self._within_industry_rank(roe_score, reverse=False)
        gm_score = self._within_industry_rank(gm_score, reverse=False)
        cfo_score = self._within_industry_rank(cfo_score, reverse=False)

        # --- 等权合并:ROE + 毛利率 + CFO/NP ---
        score_qual = (
            roe_score.fillna(50.0) * 0.4 +
            gm_score.fillna(50.0) * 0.35 +
            cfo_score.fillna(50.0) * 0.25
        )

        return {
            "score_qual": score_qual,
            "roe_raw": financial_df.get("roe", pd.Series(dtype=float)),
            "gm_raw": financial_df.get("gross_margin", pd.Series(dtype=float)),
            "cfo_raw": financial_df.get("cfo_to_np", pd.Series(dtype=float)),
        }

    # ============================================================
    # 动量因子(越高越好 → 正向排名)
    # ============================================================

    def _score_momentum(
        self,
        price_df: pd.DataFrame,
        symbols: List[str],
    ) -> Dict[str, pd.Series]:
        """
        1月收益率,3月收益率.
        """
        available = [s for s in symbols if s in price_df.columns]
        if not available:
            return {"score_mom": pd.Series(dtype=float)}

        prices = price_df[available]
        idx = pd.Index(available, name="symbol")

        # --- 1月收益率 ---
        if len(prices) >= MOMENTUM_1M:
            ret_1m = (prices.iloc[-1] / prices.iloc[-MOMENTUM_1M] - 1) * 100
        else:
            ret_1m = pd.Series(50.0, index=idx)
        ret_1m = winsorize(ret_1m, 1, 99)
        score_1m = safe_percentile_rank(ret_1m)

        # --- 3月收益率 ---
        if len(prices) >= MOMENTUM_3M:
            ret_3m = (prices.iloc[-1] / prices.iloc[-MOMENTUM_3M] - 1) * 100
        else:
            ret_3m = pd.Series(50.0, index=idx)
        ret_3m = winsorize(ret_3m, 1, 99)
        score_3m = safe_percentile_rank(ret_3m)

        # --- 行业内排名修正 ---
        score_1m = self._within_industry_rank(score_1m, reverse=False)
        score_3m = self._within_industry_rank(score_3m, reverse=False)

        # --- 合并:1M(60%) + 3M(40%) ---
        score_mom = (
            score_1m.fillna(50.0) * 0.6 +
            score_3m.fillna(50.0) * 0.4
        )

        return {
            "score_mom": score_mom,
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
        }

    # ============================================================
    # 波动/反转因子
    # ============================================================

    def _score_volatility_reversal(
        self,
        price_df: pd.DataFrame,
        symbols: List[str],
    ) -> Dict[str, pd.Series]:
        """
        20日波动率(正向:要有弹性),5日反转(负向:短期超跌看反弹).
        """
        available = [s for s in symbols if s in price_df.columns]
        if not available:
            return {"score_vol": pd.Series(dtype=float)}

        prices = price_df[available]
        idx = pd.Index(available, name="symbol")

        # --- 20日波动率 ---
        if len(prices) >= VOLATILITY_20D:
            daily_ret = prices.pct_change().iloc[-VOLATILITY_20D:]
            vol_20d = daily_ret.std() * np.sqrt(252) * 100  # 年化波动率
        else:
            vol_20d = pd.Series(50.0, index=idx)
        vol_20d = winsorize(vol_20d, 1, 99)
        # 正向:波动率越高越好(趋势策略需要弹性),但过高也不好
        score_vol = safe_percentile_rank(vol_20d)
        # 过高的波动率给轻微扣分(超过 99 分位 = 赌博票)
        vol_pct = safe_percentile_rank(vol_20d)
        score_vol = score_vol.where(vol_pct < 99, score_vol * 0.5)

        # --- 5日反转 ---
        if len(prices) >= REVERSAL_5D:
            ret_5d = (prices.iloc[-1] / prices.iloc[-REVERSAL_5D] - 1) * 100
        else:
            ret_5d = pd.Series(0.0, index=idx)
        ret_5d = winsorize(ret_5d, 1, 99)
        # 负向:跌得越多 → 反转潜力越大 → 得分越高
        score_rev = 100.0 - safe_percentile_rank(ret_5d)

        # --- 行业内排名修正 ---
        score_vol = self._within_industry_rank(score_vol, reverse=False)
        score_rev = self._within_industry_rank(score_rev, reverse=False)

        # --- 合并:波动(60%) + 反转(40%) ---
        score_vol_combined = (
            score_vol.fillna(50.0) * 0.6 +
            score_rev.fillna(50.0) * 0.4
        )

        return {
            "score_vol": score_vol_combined,
            "vol_20d": vol_20d,
            "ret_5d": ret_5d,
        }

    # ============================================================
    # 行业中性化
    # ============================================================

    def _within_industry_rank(
        self,
        scores: pd.Series,
        reverse: bool = False,
    ) -> pd.Series:
        """
        在申万二级行业内重新排名,实现行业中性化.

        参数:
            scores: 全市场因子得分
            reverse: True = 反向排名(值越小分越高)

        返回:
            行业中性化后的得分 (0-100)
        """
        if not self.industry_map:
            return scores

        # 映射行业
        industry_series = scores.index.map(self.industry_map)
        industry_series = pd.Series(industry_series, index=scores.index)

        result = scores.copy()
        industries = industry_series.dropna().unique()

        for ind in industries:
            mask = industry_series == ind
            if mask.sum() < 5:
                continue  # 行业内股票太少,不修正

            sub_scores = scores[mask]
            ranked = safe_percentile_rank(sub_scores)
            if reverse:
                ranked = 100.0 - ranked
            result[mask] = ranked

        return result

    def get_factor_detail(self, scores_df: pd.DataFrame, symbol: str) -> dict:
        """获取单只股票的因子明细(用于诊断)"""
        if symbol not in scores_df.index:
            return {}
        row = scores_df.loc[symbol]
        return {k: round(float(v), 2) for k, v in row.items() if not pd.isna(v)}


# ============================================================
# 模块自检
# ============================================================

if __name__ == "__main__":
    print("AgentFactor 模块加载成功")

    # 构造模拟数据
    symbols = ["600519.XSHG", "000858.XSHE", "300750.XSHE"]
    fake_prices = pd.DataFrame({
        s: np.cumprod(1 + np.random.randn(100) * 0.02)
        for s in symbols
    })
    fake_fin = pd.DataFrame({
        "roe": [30.0, 25.0, 15.0],
        "gross_margin": [90.0, 75.0, 30.0],
        "cfo_to_np": [1.2, 0.8, 1.5],
    }, index=symbols)
    fake_universe = pd.DataFrame({
        "symbol": symbols, "pe": [30, 25, 50], "pb": [8, 5, 12],
    })

    agent = AgentFactor()
    scores = agent.score_all(fake_universe, fake_prices, fake_fin, verbose=True)
    print(scores.to_string())
