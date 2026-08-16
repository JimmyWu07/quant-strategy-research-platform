"""
v2 多因子选股系统 - Coordinator 调度器

职责:
  - 串联所有 Agent 模块
  - 单次筛选: run_screening(date) → 精选 50 只股票
  - 逐月回测: run_backtest() → 每月调仓,跟踪收益

数据流:
  AgentData → AgentFactor + AgentRisk + AgentMacro → AgentPortfolio
       │              │              │              │
       │   估值/质量/动量/波动      行业/个股风控    宏观仓位
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                              │
                    最终得分 → Top 50 → 权重
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from config import (
    BACKTEST_START, BACKTEST_END,
    TOP_N_STOCKS, OUTPUT_DIR,
    TURNOVER_COST_RATE, BENCHMARK_SYMBOL, BENCHMARK_NAME,
    RISK_FREE_RATE_MONTHLY,
)
from config.utils import (
    get_logger, save_cache, load_cache,
    generate_rebalance_dates, progress_bar,
)
from agents.agent_data import AgentData
from agents.agent_factor import AgentFactor
from agents.agent_risk import AgentRisk
from agents.agent_macro import AgentMacro
from agents.agent_portfolio import AgentPortfolio

logger = get_logger("Coordinator")


class Coordinator:
    """
    总调度器.

    用法:
        coord = Coordinator()
        result = coord.run_screening("2025-06-30")
        backtest_results = coord.run_backtest()
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.agent_data = AgentData(use_cache=use_cache)
        self.agent_macro = AgentMacro()

        # AgentFactor / AgentRisk / AgentPortfolio 依赖行业分类,
        # 在首次 run 时延迟初始化
        self._industry_map = None
        self._agent_factor = None
        self._agent_risk = None
        self._agent_portfolio = None

    @property
    def industry_map(self) -> dict:
        if self._industry_map is None:
            self._industry_map = self.agent_data.get_industry_map()
        return self._industry_map

    @property
    def factor_agent(self) -> AgentFactor:
        if self._agent_factor is None:
            self._agent_factor = AgentFactor(self.industry_map)
        return self._agent_factor

    @property
    def risk_agent(self) -> AgentRisk:
        if self._agent_risk is None:
            self._agent_risk = AgentRisk(self.industry_map)
        return self._agent_risk

    @property
    def portfolio_agent(self) -> AgentPortfolio:
        if self._agent_portfolio is None:
            self._agent_portfolio = AgentPortfolio(self.industry_map)
        return self._agent_portfolio

    # ============================================================
    # 单次筛选
    # ============================================================

    def run_screening(
        self,
        date: str,
        verbose: bool = True,
    ) -> dict:
        """
        在指定日期执行完整的选股流程.

        参数:
            date: 调仓日期 "YYYY-MM-DD"
            verbose: 是否输出详细日志

        返回:
            {
                "date": str,
                "macro": dict,           # 宏观评估结果
                "portfolio": dict,        # 组合构建结果
                "top_50": pd.DataFrame,   # 精选股票明细
                "weights": pd.Series,     # 组合权重
            }
        """
        t_start = time.time()
        logger.info("=" * 60)
        logger.info("开始选股流程: %s", date)
        logger.info("=" * 60)

        dt = pd.Timestamp(date)
        lookback_start = (dt - pd.DateOffset(years=2)).strftime("%Y-%m-%d")

        # === Step 1: 获取股票池 ===
        logger.info("[Step 1/6] 获取股票池...")
        universe = self.agent_data.get_stock_universe(date)
        if universe.empty:
            logger.error("股票池为空,无法继续")
            return {"date": date, "error": "empty universe"}

        symbols = universe["symbol"].tolist() if "symbol" in universe.columns else list(universe.index)
        logger.info("候选池: %d 只股票", len(symbols))

        # === Step 2: 拉取历史K线 ===
        logger.info("[Step 2/6] 获取历史K线...")
        price_df = self.agent_data.get_daily_prices(symbols, lookback_start, date)
        if price_df is None or price_df.empty:
            logger.error("K线数据为空")
            return {"date": date, "error": "no price data"}

        # === Step 3: 财务数据 ===
        logger.info("[Step 3/6] 获取财务数据...")
        financial_df = self.agent_data.get_financials(symbols, date, universe_df=universe)

        # === Step 4: PE/PB 历史分位 ===
        logger.info("[Step 4/6] 计算 PE/PB 历史分位...")
        pe_pb_pct_df = self.agent_data.get_pe_pb_history(symbols)

        # === Step 5: 因子打分 + 风控 ===
        logger.info("[Step 5/6] 因子打分 & 风控评估...")
        factor_scores = self.factor_agent.score_all(
            universe, price_df, financial_df, pe_pb_pct_df, verbose=verbose,
        )
        risk_penalties = self.risk_agent.assess(
            universe, price_df, pe_pb_pct_df, financial_df, verbose=verbose,
        )

        # === Step 6: 组合构建 ===
        logger.info("[Step 6/6] 组合构建...")
        portfolio = self.portfolio_agent.build(factor_scores, risk_penalties)

        # === 宏观仓位 ===
        macro_raw = self.agent_data.get_macro_data(date)
        macro_result = self.agent_macro.evaluate(macro_raw)

        # 应用宏观仓位
        if not portfolio["weights"].empty:
            adjusted_weights = self.agent_macro.apply_position_to_portfolio(
                portfolio["weights"], macro_result["position_ratio"]
            )
        else:
            adjusted_weights = pd.Series(dtype=float)

        elapsed = time.time() - t_start
        logger.info("选股完成,耗时 %.1f 秒", elapsed)

        # 输出摘要
        if verbose and not portfolio["selected"].empty:
            print(self.portfolio_agent.get_portfolio_summary(portfolio))

        # === 保存结果到 output/ ===
        if not portfolio["selected"].empty:
            self._save_screening_result(date, portfolio, macro_result, adjusted_weights)

        return {
            "date": date,
            "macro": macro_result,
            "portfolio": portfolio,
            "top_50": portfolio["selected"],
            "weights": adjusted_weights,
            "elapsed_seconds": round(elapsed, 1),
        }

    def _save_screening_result(
        self, date: str, portfolio: dict, macro_result: dict, weights: pd.Series
    ) -> None:
        """保存单次筛选结果到 CSV"""
        selected = portfolio["selected"].copy()
        if selected.empty:
            return

        # 添加权重列
        selected["weight"] = selected.index.map(weights) if not weights.empty else 0

        # 文件名
        date_str = date.replace("-", "")
        csv_path = OUTPUT_DIR / f"screening_{date_str}.csv"
        selected.to_csv(csv_path, encoding="utf-8-sig", float_format="%.4f")
        logger.info("结果已保存: %s", csv_path)

        # 也保存一个摘要 JSON
        import json
        summary = {
            "date": date,
            "macro_regime": macro_result.get("regime", "?"),
            "position_ratio": macro_result.get("position_ratio", 0),
            "n_stocks": len(selected),
            "avg_score": round(float(selected["final_score"].mean()), 2),
            "top_3": selected.head(3).index.tolist(),
        }
        json_path = OUTPUT_DIR / f"screening_{date_str}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("摘要已保存: %s", json_path)

    # ============================================================
    # 逐月回测
    # ============================================================

    def run_backtest(
        self,
        start: str = BACKTEST_START,
        end: str = BACKTEST_END,
        verbose: bool = True,
    ) -> dict:
        """
        逐月回测:每月末运行筛选,跟踪下一个月的组合收益.

        返回:
            {
                "rebalance_results": [dict, ...],   # 每期筛选结果
                "monthly_returns": pd.Series,       # 每月组合收益
                "cumulative_return": float,         # 累计收益
                "summary": dict,                    # 回测摘要
            }
        """
        logger.info("=" * 60)
        logger.info("  开始逐月回测: %s → %s", start, end)
        logger.info("=" * 60)

        dates = generate_rebalance_dates(start, end, freq="monthly")
        logger.info("共 %d 个调仓日", len(dates))

        results = []
        monthly_returns = {}
        benchmark_returns = {}

        for i, date in enumerate(dates):
            logger.info("\n>>> 调仓 %d/%d: %s", i + 1, len(dates), date)

            try:
                result = self.run_screening(date, verbose=verbose)
                results.append(result)

                # 计算下月收益(如果有下一期数据)
                if i < len(dates) - 1:
                    next_date = dates[i + 1]
                    stock_ret = self._calc_holding_return(
                        result.get("top_50", pd.DataFrame()),
                        date, next_date,
                    )
                    # 宏观仓位: 股票部分 × position_ratio + 现金部分 × 无风险月利率
                    position_ratio = result.get("macro", {}).get("position_ratio", 1.0)
                    cash_ratio = max(0.0, 1.0 - position_ratio)
                    if not np.isnan(stock_ret):
                        ret = stock_ret * position_ratio + RISK_FREE_RATE_MONTHLY * cash_ratio
                    else:
                        ret = np.nan

                    monthly_returns[date] = ret

                    # 基准收益
                    bench_ret = self._calc_benchmark_return(date, next_date)
                    benchmark_returns[date] = bench_ret

                    excess = ret - bench_ret if not np.isnan(bench_ret) else np.nan
                    logger.info(
                        "持有期 %s → %s 收益: %.2f%% (股票×%.0f%% + 现金×%.0f%%)"
                        " | %s: %.2f%% | 超额: %.2f%%",
                        date, next_date, ret * 100 if not np.isnan(ret) else 0,
                        position_ratio * 100, cash_ratio * 100,
                        BENCHMARK_NAME,
                        bench_ret * 100 if not np.isnan(bench_ret) else 0,
                        excess * 100 if not np.isnan(excess) else 0
                    )

                    # 进度汇总
                    done_count = i + 1
                    valid_returns = [v for v in monthly_returns.values() if not np.isnan(v)]
                    cum_ret = 1.0
                    for v in valid_returns:
                        cum_ret *= (1 + v)
                    cum_ret -= 1
                    avg_ret = sum(valid_returns) / len(valid_returns) if valid_returns else 0
                    win_count = sum(1 for v in valid_returns if v > 0)
                    logger.info(
                        "========== 回测进度: %d/%d 期 (%.0f%%) ==========\n"
                        "  累计收益: %.2f%% | 月均: %.2f%% | 胜率: %d/%d (%.0f%%)\n"
                        "  下一期: %s%s",
                        done_count, len(dates), done_count / len(dates) * 100,
                        cum_ret * 100, avg_ret * 100,
                        win_count, len(valid_returns),
                        win_count / len(valid_returns) * 100 if valid_returns else 0,
                        dates[i + 1] if i + 1 < len(dates) else "完成",
                        " (剩余 %d 期)" % (len(dates) - done_count) if done_count < len(dates) else ""
                    )

            except Exception as e:
                logger.error("调仓 %s 失败: %s", date, e, exc_info=True)
                results.append({"date": date, "error": str(e)})

        # 汇总回测收益
        returns_series = pd.Series(monthly_returns).sort_index()
        benchmark_series = pd.Series(benchmark_returns).sort_index()
        cumulative = (1 + returns_series).prod() - 1 if not returns_series.empty else 0

        summary = self._backtest_summary(returns_series, results, benchmark_series)

        logger.info("\n" + "=" * 60)
        logger.info("回测完成:")
        logger.info("  策略累计收益: %.2f%%", cumulative * 100)
        logger.info("  策略年化收益: %.2f%%", summary.get("annual_return", 0) * 100)
        logger.info("  策略最大回撤: %.2f%%", summary.get("max_drawdown", 0) * 100)
        logger.info("  策略夏普比率: %.2f", summary.get("sharpe_ratio", 0))
        if "benchmark_cumulative" in summary:
            logger.info("  ---")
            logger.info("  %s累计收益: %.2f%%", BENCHMARK_NAME, summary.get("benchmark_cumulative", 0) * 100)
            logger.info("  %s年化收益: %.2f%%", BENCHMARK_NAME, summary.get("benchmark_annual", 0) * 100)
            logger.info("  %s最大回撤: %.2f%%", BENCHMARK_NAME, summary.get("benchmark_max_drawdown", 0) * 100)
            logger.info("  ---")
            logger.info("  超额累计收益: %.2f%%", summary.get("excess_cumulative", 0) * 100)
            logger.info("  超额年化收益: %.2f%%", summary.get("excess_annual", 0) * 100)
            logger.info("  信息比率: %.2f", summary.get("information_ratio", 0))
        logger.info("  ---")
        logger.info("  交易成本(单次): %.4f%%", TURNOVER_COST_RATE * 100)
        logger.info("=" * 60)

        # 保存结果
        self._save_backtest_results(
            returns_series, cumulative, summary, results, benchmark_series
        )

        return {
            "rebalance_results": results,
            "monthly_returns": returns_series,
            "benchmark_returns": benchmark_series,
            "cumulative_return": cumulative,
            "summary": summary,
        }

    def _calc_holding_return(
        self,
        selected: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> float:
        """
        计算等权组合在持有期的收益率(扣除交易成本).

        交易成本: 佣金(万2.5×2) + 印花税(0.05%) + 滑点(0.1%×2) ≈ 0.085%
        """
        if selected.empty:
            return 0.0

        symbols = selected.index.tolist()
        try:
            price_df = self.agent_data.get_daily_prices(symbols, start_date, end_date)
            if price_df is None or price_df.empty or len(price_df) < 2:
                return np.nan

            returns = []
            for sym in symbols:
                if sym in price_df.columns:
                    col = price_df[sym].dropna()
                    if len(col) >= 2:
                        r = col.iloc[-1] / col.iloc[0] - 1
                        returns.append(r)

            if returns:
                gross_return = float(np.mean(returns))
                # 扣除交易成本(买入+卖出)
                net_return = gross_return - TURNOVER_COST_RATE
                return net_return
            return np.nan
        except Exception:
            return np.nan

    def _calc_benchmark_return(self, start_date: str, end_date: str) -> float:
        """计算沪深300基准在持有期的收益率(用新浪指数接口)"""
        try:
            import akshare as ak
            # 新浪指数日线: 一次拉全历史,本地按日期过滤
            cache_name = "benchmark_hs300_daily"
            cached = load_cache(cache_name) if self.agent_data.use_cache else None
            if cached is not None:
                df = cached
            else:
                df = ak.stock_zh_index_daily(symbol="sh000300")
                if df is not None and not df.empty:
                    save_cache(cache_name, df)

            if df is None or df.empty:
                return np.nan

            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            subset = df.loc[mask].sort_values("date")
            if len(subset) < 2:
                return np.nan
            return float(subset["close"].iloc[-1] / subset["close"].iloc[0] - 1)
        except Exception as e:
            logger.debug("基准收益计算失败: %s", str(e)[:80])
            return np.nan

    def _backtest_summary(
        self,
        returns: pd.Series,
        results: List[dict],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> dict:
        """计算回测绩效指标(含基准对比)"""
        valid = returns.dropna()
        if len(valid) < 2:
            return {
                "n_periods": len(valid),
                "cumulative_return": 0,
                "annual_return": 0,
                "volatility": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "error_count": sum(1 for r in results if "error" in r),
            }

        cumulative = (1 + valid).prod() - 1
        annual_return = (1 + cumulative) ** (12 / len(valid)) - 1
        annual_vol = valid.std() * np.sqrt(12)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cum_series = (1 + valid).cumprod()
        running_max = cum_series.cummax()
        drawdown = (cum_series - running_max) / running_max
        max_dd = float(drawdown.min())

        win_rate = (valid > 0).mean()

        summary = {
            "n_periods": len(valid),
            "cumulative_return": round(float(cumulative), 4),
            "annual_return": round(float(annual_return), 4),
            "volatility": round(float(annual_vol), 4),
            "sharpe_ratio": round(float(sharpe), 2),
            "max_drawdown": round(float(max_dd), 4),
            "win_rate": round(float(win_rate), 3),
            "error_count": sum(1 for r in results if "error" in r),
        }

        # 基准对比
        if benchmark_returns is not None and not benchmark_returns.empty:
            bench_valid = benchmark_returns.reindex(valid.index).dropna()
            if len(bench_valid) >= 2:
                bench_cum = (1 + bench_valid).prod() - 1
                bench_annual = (1 + bench_cum) ** (12 / len(bench_valid)) - 1
                bench_vol = bench_valid.std() * np.sqrt(12)
                bench_sharpe = bench_annual / bench_vol if bench_vol > 0 else 0

                # 超额收益
                excess = valid - bench_valid.reindex(valid.index).fillna(0)
                excess_cum = (1 + excess).prod() - 1
                excess_annual = (1 + excess_cum) ** (12 / len(excess)) - 1
                tracking_error = excess.std() * np.sqrt(12) if excess.std() > 0 else 0
                info_ratio = excess_annual / tracking_error if tracking_error > 0 else 0

                # 基准最大回撤
                bench_cum_series = (1 + bench_valid).cumprod()
                bench_running_max = bench_cum_series.cummax()
                bench_dd = (bench_cum_series - bench_running_max) / bench_running_max
                bench_max_dd = float(bench_dd.min())

                summary.update({
                    "benchmark_name": BENCHMARK_NAME,
                    "benchmark_cumulative": round(float(bench_cum), 4),
                    "benchmark_annual": round(float(bench_annual), 4),
                    "benchmark_volatility": round(float(bench_vol), 4),
                    "benchmark_sharpe": round(float(bench_sharpe), 2),
                    "benchmark_max_drawdown": round(float(bench_max_dd), 4),
                    "excess_cumulative": round(float(excess_cum), 4),
                    "excess_annual": round(float(excess_annual), 4),
                    "tracking_error": round(float(tracking_error), 4),
                    "information_ratio": round(float(info_ratio), 2),
                })

        return summary

    def _save_backtest_results(
        self,
        returns: pd.Series,
        cumulative: float,
        summary: dict,
        results: List[dict],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> None:
        """保存回测结果到 output 目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存收益序列(策略+基准)
        returns.to_csv(OUTPUT_DIR / f"monthly_returns_{timestamp}.csv", header=["return"])
        if benchmark_returns is not None and not benchmark_returns.empty:
            combined = pd.DataFrame({"strategy": returns, "benchmark": benchmark_returns})
            combined.to_csv(OUTPUT_DIR / f"returns_vs_benchmark_{timestamp}.csv")

        # 保存摘要
        with open(OUTPUT_DIR / f"backtest_summary_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 保存各期 Top 10(精简版)
        top_records = []
        for r in results:
            if "top_50" in r and not r["top_50"].empty:
                for sym, row in r["top_50"].head(10).iterrows():
                    top_records.append({
                        "date": r["date"],
                        "symbol": sym,
                        "industry": row.get("industry", ""),
                        "final_score": round(float(row.get("final_score", 0)), 2),
                        "rank": int(row.get("rank", 0)),
                    })

        if top_records:
            pd.DataFrame(top_records).to_csv(
                OUTPUT_DIR / f"top10_history_{timestamp}.csv", index=False
            )

        logger.info("回测结果已保存到 %s", OUTPUT_DIR)


# ============================================================
# 模块自检
# ============================================================

if __name__ == "__main__":
    print("Coordinator 模块加载成功")

    coord = Coordinator(use_cache=False)

    # 测试单次筛选(仅拉取当前数据)
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n运行单次筛选 (date={today})...")
    result = coord.run_screening(today, verbose=True)

    if "error" not in result:
        print(f"\n宏观仓位: {result['macro'].get('regime')}, "
              f"仓位={result['macro'].get('position_ratio', 0):.0%}")
        print(f"精选 {len(result.get('top_50', pd.DataFrame()))} 只股票")
