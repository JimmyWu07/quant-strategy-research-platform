"""
v2 多因子选股系统 - AgentData 数据获取模块

数据源架构(网络环境: 东方财富API被墙, 使用以下替代方案):
  - 股票列表: akshare stock_info_a_code_name()
  - 实时行情(PE/PB/市值/成交额): 腾讯行情 HTTP API (qt.gtimg.cn)
  - 历史K线: akshare stock_zh_a_daily() [新浪源]
  - 财务数据: akshare stock_financial_analysis_indicator()
  - 行业分类: akshare stock_board_industry_name_ths() + _cons_ths()
  - 宏观数据: akshare macro_china_*()

所有数据获取函数都具备重试和容错能力,单只股票失败不影响整体流程.
"""

import re
import time
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import requests

from config import (
    MIN_LISTING_DAYS, MIN_AVG_TURNOVER,
    EXCLUDE_ST, EXCLUDE_BEIJING, PRE_FILTER_TOP_N,
    FINANCIAL_LAG_MONTHS, HISTORICAL_THRESHOLD_DAYS,
    INDUSTRY_STANDARD, DATA_CACHE_DIR,
)
from config.utils import (
    get_logger, retry_on_error, save_cache, load_cache,
    clean_symbol, is_st_stock, is_beijing_stock,
    progress_bar,
)

logger = get_logger("AgentData")

# ============================================================
# 缓存键名
# ============================================================
CACHE_UNIVERSE = "stock_universe"
CACHE_PRICES = "daily_prices"
CACHE_FINANCIALS = "financial_data"
CACHE_INDUSTRY = "industry_map"
CACHE_PE_HISTORY = "pe_pb_pct"
CACHE_MACRO = "macro_data"
CACHE_STOCK_LIST = "stock_list"

# 腾讯行情 API 批量上限
TENCENT_BATCH_SIZE = 80


class AgentData:
    """
    数据获取 Agent.

    用法:
        agent = AgentData()
        universe = agent.get_stock_universe("2025-06-30")
        prices = agent.get_daily_prices(symbols, "2024-01-01", "2025-06-30")
        financials = agent.get_financials(symbols, "2025-06-30")
        industry = agent.get_industry_map()
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._industry_map: Optional[Dict[str, str]] = None
        self._stock_name_map: Optional[Dict[str, str]] = None

    # ============================================================
    # 0. 基础股票列表
    # ============================================================

    def _get_stock_list(self) -> pd.DataFrame:
        """获取全 A 股代码+名称列表"""
        cached = load_cache(CACHE_STOCK_LIST) if self.use_cache else None
        if cached is not None:
            return cached

        import akshare as ak
        df = ak.stock_info_a_code_name()
        df.columns = ["symbol_raw", "name"]
        df["symbol"] = df["symbol_raw"].apply(clean_symbol)
        df = df.drop(columns=["symbol_raw"])
        save_cache(CACHE_STOCK_LIST, df)
        return df

    # ============================================================
    # 1. 股票池获取 + 硬门槛过滤
    # ============================================================

    def get_stock_universe(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        获取经硬门槛过滤后的候选股票池.

        数据模式:
          - 实时筛选 (ref_date 距今日 ≤ HISTORICAL_THRESHOLD_DAYS 天): 使用腾讯行情,数据完整(PE/PB/成交额/市值)
          - 历史回测 (超过阈值): 避免前视偏差,用 K 线数据近似过滤(上市天数 + 20日日均成交额)
        """
        cache_name = f"{CACHE_UNIVERSE}_{date}" if date else CACHE_UNIVERSE
        cached = load_cache(cache_name) if self.use_cache else None
        if cached is not None:
            logger.info("命中缓存: %s,共 %d 只股票", cache_name, len(cached))
            return cached

        today = pd.Timestamp.today().normalize()
        ref_ts = pd.Timestamp(date) if date else today
        days_diff = (today - ref_ts).days
        is_historical = (date is not None) and (days_diff > HISTORICAL_THRESHOLD_DAYS)

        if is_historical:
            logger.info("[历史回测模式] ref=%s (距今日 %d 天),改用 K 线估算候选池", date, days_diff)
            df = self._get_stock_universe_historical(stock_list=self._get_stock_list(), ref_date=date)
        else:
            logger.info("[实时筛选模式] ref=%s (距今日 %d 天),使用腾讯行情", date, days_diff)
            df = self._get_stock_universe_realtime(stock_list=self._get_stock_list())

        # 只缓存非空结果,避免空缓存阻断后续重试
        if df is not None and not df.empty:
            save_cache(cache_name, df)
        return df

    def _get_stock_universe_realtime(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        """实时筛选模式: 腾讯行情获取 PE/PB/市值/成交额 等完整数据"""
        all_symbols = stock_list["symbol"].tolist()
        logger.info("全市场: %d 只股票", len(all_symbols))

        quotes = self._fetch_tencent_quotes_batch(all_symbols)
        if quotes.empty:
            logger.error("腾讯行情数据获取失败")
            return pd.DataFrame()

        name_lookup = stock_list.set_index("symbol")["name"]
        if "name" in quotes.columns:
            quotes["name"] = quotes["name"].fillna(quotes["symbol"].map(name_lookup))
        else:
            quotes["name"] = quotes["symbol"].map(name_lookup)

        df = self._apply_hard_filters(quotes, mode="realtime")
        logger.info("股票池过滤完成: 全市场 %d → 候选池 %d 只", len(quotes), len(df))
        return df

    def _get_stock_universe_historical(self, stock_list: pd.DataFrame, ref_date: str) -> pd.DataFrame:
        """
        历史回测模式: 先用腾讯批量行情初筛,再对 Top N 拉 K 线验证上市天数.

        策略: 用当前行情数据做粗筛(成交额/PE/PB),只对最终 Top N 拉历史K线检查上市天数.
        这样 K 线拉取从 ~5000 只降到 ~300 只.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_symbols = stock_list["symbol"].tolist()
        logger.info("全市场: %d 只股票", len(all_symbols))

        # Step1: 腾讯批量行情初筛 (一次API拿全市场)
        quotes = self._fetch_tencent_quotes_batch(all_symbols)
        if quotes.empty:
            logger.error("腾讯行情数据获取失败")
            return pd.DataFrame()

        name_lookup = stock_list.set_index("symbol")["name"]
        if "name" in quotes.columns:
            quotes["name"] = quotes["name"].fillna(quotes["symbol"].map(name_lookup))
        else:
            quotes["name"] = quotes["symbol"].map(name_lookup)

        # Step2: 硬门槛过滤 (ST/北交所/PE/PB/成交额/Top N)
        df = self._apply_hard_filters(quotes, mode="realtime")
        logger.info("腾讯行情初筛: %d → %d 只", len(quotes), len(df))

        if df.empty:
            return df

        # Step3: 只对 Top N 拉历史K线, 验证上市天数
        symbols = df["symbol"].tolist()
        logger.info("拉取 %d 只 Top 候选的 K线 (验证上市天数)...", len(symbols))

        lookback_start = (pd.Timestamp(ref_date) - pd.DateOffset(days=450)).strftime("%Y-%m-%d")
        listing_map = {}
        MAX_WORKERS = min(10, len(symbols))
        completed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(self._fetch_full_kline_with_amount,
                                sym, lookback_start, ref_date): sym
                for sym in symbols
            }
            for future in as_completed(future_map):
                sym = future_map[future]
                completed += 1
                if completed % 50 == 0:
                    logger.info(progress_bar(completed, len(symbols), "上市天数验证"))
                try:
                    info = future.result()
                    if info is not None:
                        listing_map[sym] = info
                except Exception:
                    continue

        # Step4: 上市天数过滤
        if MIN_LISTING_DAYS and MIN_LISTING_DAYS > 0:
            before = len(df)
            valid_syms = [s for s, info in listing_map.items()
                          if info["listing_days"] >= MIN_LISTING_DAYS]
            df = df[df["symbol"].isin(valid_syms)]
            logger.info("上市天数过滤: %d → %d 只 (门槛 %d天, K线有效 %d 只)",
                        before, len(df), MIN_LISTING_DAYS, len(listing_map))

        # Step5: 用历史20日均成交额替换当日成交额 (更准确)
        for idx, row in df.iterrows():
            sym = row["symbol"]
            if sym in listing_map and not np.isnan(listing_map[sym]["avg_turnover_20d"]):
                df.at[idx, "turnover"] = listing_map[sym]["avg_turnover_20d"]

        logger.warning(
            "[历史回测模式] PE/PB/成交额来自腾讯当前行情(非ref_date当日),"
            "但财务+宏观数据已做前视偏差修复,因子排名用截面相对值,影响可控。"
        )
        logger.info("(历史回测) 候选池过滤完成: 候选池 %d 只", len(df))
        return df.reset_index(drop=True)

    def _fetch_full_kline_with_amount(
        self, symbol: str, start: str, end: str
    ) -> Optional[dict]:
        """
        获取单只股票完整历史K线(含成交额列).
        返回 dict: {symbol, listing_days, avg_turnover_20d, close_on_ref}
        """
        hist = self._fetch_single_stock_history_full(symbol, start, end)
        if hist is None or hist.empty:
            return None

        listing_days = len(hist)
        # 近 20 日日均成交额
        if "volume" in hist.columns and "close" in hist.columns:
            # 新浪 ak.stock_zh_a_daily 的列: date, open, high, low, close, volume, outstanding_share, turnover
            # 成交额 ≈ 成交量 * 收盘价 近似
            turnover_series = hist["volume"] * hist["close"]
            if "turnover" in hist.columns:
                # 如果有真实 turnover 列优先使用
                turnover_series = pd.to_numeric(hist["turnover"], errors="coerce").fillna(turnover_series)
            avg_turnover_20d = float(turnover_series.tail(20).mean()) if len(turnover_series) >= 20 else float(turnover_series.mean())
        else:
            avg_turnover_20d = np.nan

        close_on_ref = float(hist["close"].iloc[-1]) if not hist.empty else np.nan

        return {
            "symbol": symbol,
            "listing_days": listing_days,
            "avg_turnover_20d": avg_turnover_20d,
            "close_on_ref": close_on_ref,
        }

    @retry_on_error(max_retries=1, delay=0.3)
    def _fetch_single_stock_history_full(
        self, symbol: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """拉取完整K线(含成交量列), 供历史回测模式做候选池估算"""
        import akshare as ak
        code = symbol.replace(".XSHG", "").replace(".XSHE", "")
        sina_code = f"sh{code}" if symbol.endswith(".XSHG") or code.startswith(("60", "68")) else f"sz{code}"

        try:
            raw = ak.stock_zh_a_daily(symbol=sina_code, adjust="qfq")
        except Exception:
            return None

        if raw is None or raw.empty:
            return None
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.set_index("date").sort_index()
        mask = (raw.index >= start) & (raw.index <= end)
        return raw.loc[mask] if mask.any() else None

    def _apply_hard_filters(self, df: pd.DataFrame, mode: str = "realtime") -> pd.DataFrame:
        """
        应用硬门槛过滤.

        参数 mode:
          - "realtime": 腾讯行情数据,有 PE/PB/成交额(当日)
          - "historical": 已在 _get_stock_universe_historical 内联实现
        """
        initial = len(df)

        # 1. 排除 ST
        if EXCLUDE_ST and "name" in df.columns:
            df = df[~df["name"].apply(is_st_stock)]

        # 2. 排除北交所(8开头)
        if EXCLUDE_BEIJING and "symbol" in df.columns:
            df = df[~df["symbol"].apply(is_beijing_stock)]

        # 3. 排除 PE/PB <= 0(亏损公司,估值因子不适用)
        for col in ["pe", "pb"]:
            if col in df.columns:
                df = df[df[col] > 0]

        # 4. 排除 PE > 500(极端值)
        if "pe" in df.columns:
            df = df[df["pe"] <= 500]

        # 5. 流动性门槛:成交额 >= 5000万
        if "turnover" in df.columns:
            df = df[df["turnover"] >= MIN_AVG_TURNOVER]

        # 6. 按成交额降序取 Top N(流动性最好的票优先)
        if "turnover" in df.columns and len(df) > PRE_FILTER_TOP_N:
            df = df.nlargest(PRE_FILTER_TOP_N, "turnover")

        # 7. 移除 NaN 关键列
        key_cols = [c for c in ["close", "pe", "pb", "symbol"] if c in df.columns]
        df = df.dropna(subset=key_cols)

        return df.reset_index(drop=True)

    # ============================================================
    # 腾讯行情 API(实时行情: PE/PB/市值/价格/成交额)
    # ============================================================

    def _fetch_tencent_quotes_batch(self, symbols: List[str]) -> pd.DataFrame:
        """批量拉取腾讯行情数据"""
        logger.info("批量获取腾讯行情 (%d 只)...", len(symbols))
        results = []

        for i in range(0, len(symbols), TENCENT_BATCH_SIZE):
            batch = symbols[i:i + TENCENT_BATCH_SIZE]
            try:
                batch_df = self._fetch_tencent_batch(batch)
                if not batch_df.empty:
                    results.append(batch_df)
            except Exception as e:
                logger.warning("腾讯行情批次 %d 失败: %s", i // TENCENT_BATCH_SIZE, str(e)[:80])

            if i % 500 == 0 and i > 0:
                logger.info(progress_bar(min(i, len(symbols)), len(symbols), "腾讯行情"))

        if not results:
            return pd.DataFrame()

        df = pd.concat(results, ignore_index=True)
        logger.info("腾讯行情获取完成: %d 只", len(df))
        return df

    @retry_on_error(max_retries=2, delay=1.0)
    def _fetch_tencent_batch(self, symbols: List[str]) -> pd.DataFrame:
        """请求腾讯行情 API 一个批次"""
        # 转换为腾讯格式: 600519.XSHG → sh600519, 300750.XSHE → sz300750
        tx_codes = []
        for s in symbols:
            code = s.replace(".XSHG", "").replace(".XSHE", "").replace(".BJ", "")
            if s.endswith(".XSHG") or code.startswith(("60", "68")):
                tx_codes.append(f"sh{code}")
            else:
                tx_codes.append(f"sz{code}")

        url = f"http://qt.gtimg.cn/q={','.join(tx_codes)}"
        resp = requests.get(url, timeout=15)
        resp.encoding = "gbk"

        if not resp.text.strip():
            return pd.DataFrame()

        rows = []
        for line in resp.text.strip().split("\n"):
            match = re.search(r'="([^"]*)"', line)
            if not match:
                continue
            fields = match.group(1).split("~")
            if len(fields) < 50:
                continue

            try:
                name = fields[1]
                code = fields[2]
                price = float(fields[3]) if fields[3] else None
                turnover_val = float(fields[37]) if fields[37] else None  # 成交额(万)
                turnover_rate = float(fields[38]) if fields[38] else None  # 换手率(%)
                pe = float(fields[39]) if fields[39] else None             # PE(TTM)
                total_mv = float(fields[45]) if fields[45] else None       # 总市值(亿)
                pb = float(fields[46]) if fields[46] else None             # PB
                pe_dynamic = float(fields[52]) if len(fields) > 52 and fields[52] else None

                # 用动态PE优先,否则用TTM PE
                pe_final = pe_dynamic if pe_dynamic and pe_dynamic > 0 else pe

                sym = clean_symbol(code)
                turnover_yuan = turnover_val * 10000 if turnover_val else None  # 万→元

                rows.append({
                    "symbol": sym,
                    "name": name,
                    "close": price,
                    "pe": pe_final,
                    "pb": pb,
                    "market_cap": total_mv * 1e8 if total_mv else None,  # 亿→元
                    "turnover": turnover_yuan,
                    "turnover_rate": turnover_rate,
                })
            except (ValueError, IndexError):
                continue

        return pd.DataFrame(rows)

    # ============================================================
    # 2. 历史K线数据(新浪源: stock_zh_a_daily)
    # ============================================================

    def get_daily_prices(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        批量获取股票历史日线数据(多线程并行).

        返回:
            DataFrame, index=日期, columns=股票代码, values=收盘价
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        cache_name = f"{CACHE_PRICES}_{start_date}_{end_date}"
        cached = load_cache(cache_name) if self.use_cache else None
        if cached is not None:
            logger.info("命中缓存: %s", cache_name)
            return cached

        logger.info("获取 %d 只股票历史K线 (%s → %s)...", len(symbols), start_date, end_date)

        MAX_WORKERS = min(10, len(symbols))
        price_dict = {}
        failed = 0
        completed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_sym = {
                executor.submit(self._fetch_single_stock_history, sym, start_date, end_date): sym
                for sym in symbols
            }

            for future in as_completed(future_to_sym):
                sym = future_to_sym[future]
                completed += 1
                if completed % 50 == 0:
                    logger.info(progress_bar(completed, len(symbols), "拉取K线"))

                try:
                    hist = future.result()
                    if hist is not None and not hist.empty:
                        price_dict[sym] = hist["close"]
                    else:
                        failed += 1
                except Exception as e:
                    logger.debug("%s K线失败: %s", sym, str(e)[:80])
                    failed += 1

        logger.info("K线获取完成: 成功 %d, 失败 %d", len(price_dict), failed)

        if not price_dict:
            return pd.DataFrame()

        result = pd.DataFrame(price_dict)
        result.index = pd.to_datetime(result.index)
        result = result.sort_index()

        save_cache(cache_name, result)
        return result

    @retry_on_error(max_retries=1, delay=0.3)
    def _fetch_single_stock_history(
        self, symbol: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """拉取单只股票历史K线(新浪源),含超时保护"""
        import akshare as ak
        import signal

        code = symbol.replace(".XSHG", "").replace(".XSHE", "")
        sina_code = f"sh{code}" if symbol.endswith(".XSHG") or code.startswith(("60", "68")) else f"sz{code}"

        try:
            raw = ak.stock_zh_a_daily(symbol=sina_code, adjust="qfq")
        except Exception:
            return None

        if raw is None or raw.empty:
            return None
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.set_index("date").sort_index()
        mask = (raw.index >= start) & (raw.index <= end)
        return raw.loc[mask] if mask.any() else None

    # ============================================================
    # 3. 财务数据
    # ============================================================

    def get_financials(
        self,
        symbols: List[str],
        ref_date: str,
        universe_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        获取财务指标(ROE,毛利率,经营现金流,净利润增速,PEG).

        参数:
            symbols: 股票代码列表
            ref_date: 参考日期(用于防前视偏差,只取该日期前可用的财报)
            universe_df: 股票池(含PE列,用于计算PEG)

        返回:
            DataFrame, index=股票代码, columns=[roe, gross_margin, cfo_to_np, np_yoy, peg]
        """
        cache_name = f"{CACHE_FINANCIALS}_{ref_date}"
        cached = load_cache(cache_name) if self.use_cache else None
        if cached is not None:
            logger.info("命中缓存: %s", cache_name)
            return cached

        logger.info("获取 %d 只股票财务数据 (ref=%s)...", len(symbols), ref_date)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}
        failed = 0
        completed = 0
        MAX_WORKERS = min(5, len(symbols))
        TIMEOUT_PER_STOCK = 15  # 秒

        executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        future_map = {
            executor.submit(self._fetch_single_financial, sym, ref_date): sym
            for sym in symbols
        }
        try:
            for future in as_completed(future_map, timeout=len(symbols) * TIMEOUT_PER_STOCK):
                sym = future_map[future]
                completed += 1
                if completed % 50 == 0:
                    logger.info(progress_bar(completed, len(symbols), "拉取财务"))
                try:
                    fin = future.result(timeout=TIMEOUT_PER_STOCK)
                    if fin:
                        results[sym] = fin
                    else:
                        failed += 1
                except Exception:
                    failed += 1
        except Exception:
            logger.warning("财务数据拉取超时,已完成 %d/%d", completed, len(symbols))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        logger.info("财务数据获取完成: 成功 %d, 失败 %d", len(results), failed)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results).T
        df.index.name = "symbol"

        # 计算 PEG = PE / 净利润增速(%)
        if universe_df is not None and "np_yoy" in df.columns:
            if "symbol" in universe_df.columns:
                pe_lookup = universe_df.set_index("symbol")["pe"]
            else:
                pe_lookup = universe_df["pe"]
            pe_aligned = pe_lookup.reindex(df.index)
            np_yoy_aligned = df["np_yoy"]
            # PEG 仅在 PE>0 且 np_yoy>0 时有效
            mask = (pe_aligned > 0) & (np_yoy_aligned > 0)
            df["peg"] = np.nan
            df.loc[mask, "peg"] = pe_aligned[mask] / np_yoy_aligned[mask]
            logger.info("PEG 计算: %d/%d 有效", mask.sum(), len(df))

        save_cache(cache_name, df)
        return df

    @staticmethod
    def _get_available_report_date(ref_date: str) -> pd.Timestamp:
        """
        根据 ref_date 和 FINANCIAL_LAG_MONTHS 计算最新可用财报日期.

        财报披露规则:
          一季报(3/31) → 4/30前公布,5月起可用
          中报(6/30) → 8/31前公布,9月起可用
          三季报(9/30) → 10/31前公布,11月起可用
          年报(12/31) → 次年4/30前公布,次年5月起可用
        """
        ref = pd.Timestamp(ref_date)
        ref_year = ref.year

        # 候选报告期:从近到远
        candidates = []
        for year_offset in [0, -1, -2]:
            y = ref_year + year_offset
            for q_month, available_month in FINANCIAL_LAG_MONTHS.items():
                # 报告期日期(季末)
                report_date = pd.Timestamp(year=y, month=q_month, day=1) + pd.offsets.MonthEnd(0)
                # 可用日期
                if q_month == 12:
                    # 年报:次年X月可用
                    available_date = pd.Timestamp(year=y + 1, month=available_month, day=1)
                else:
                    available_date = pd.Timestamp(year=y, month=available_month, day=1)
                if ref >= available_date:
                    candidates.append(report_date)

        return max(candidates) if candidates else pd.Timestamp(year=ref_year - 3, month=12, day=31)

    @retry_on_error(max_retries=2, delay=0.5)
    def _fetch_single_financial(self, symbol: str, ref_date: str = "") -> Optional[dict]:
        """
        获取单只股票核心财务指标.

        防前视偏差: 只取 ref_date 之前已披露的最近一期财报.
        """
        import akshare as ak
        code = symbol.replace(".XSHG", "").replace(".XSHE", "")

        try:
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2020")
            if df is None or df.empty:
                return None

            # 防前视偏差: 只取 ref_date 前可用的财报
            if ref_date:
                available_date = self._get_available_report_date(ref_date)

                # 找日期列(akshare 通常用"日期"列)
                date_col = None
                for c in df.columns:
                    if "日期" in str(c) or "date" in str(c).lower():
                        date_col = c
                        break

                if date_col is not None:
                    df["_report_date"] = pd.to_datetime(df[date_col], errors="coerce")
                    df = df[df["_report_date"] <= available_date].drop(columns=["_report_date"])
                    if df.empty:
                        logger.debug("%s: 无 %s 之前的财报数据", symbol, available_date.date())
                        return None
                # 若找不到日期列,退回取 iloc[0](降级处理)

            latest = df.iloc[0]
            result = {}

            # ROE: 净资产收益率(%)
            for c in ["净资产收益率(%)", "净资产收益率", "加权净资产收益率(%)"]:
                val = latest.get(c)
                if val is not None and pd.notna(val):
                    result["roe"] = float(val)
                    break

            # 毛利率: 销售毛利率(%)
            for c in ["销售毛利率(%)", "销售毛利率", "毛利率(%)"]:
                val = latest.get(c)
                if val is not None and pd.notna(val):
                    result["gross_margin"] = float(val)
                    break

            # 净利润同比增长率
            for c in ["净利润同比增长率(%)", "净利润同比增长率", "归属母公司净利润同比增长率(%)"]:
                val = latest.get(c)
                if val is not None and pd.notna(val):
                    result["np_yoy"] = float(val)
                    break

            # 经营现金流/净利润
            cfo, np_val = None, None
            for c in ["经营现金流量净额(元)", "经营活动产生的现金流量净额(元)", "每股经营性现金流(元)"]:
                val = latest.get(c)
                if val is not None and pd.notna(val):
                    cfo = float(val)
                    break
            for c in ["净利润(元)", "归属母公司净利润(元)", "净利润"]:
                val = latest.get(c)
                if val is not None and pd.notna(val):
                    np_val = float(val)
                    break
            if cfo and np_val and np_val != 0:
                result["cfo_to_np"] = round(cfo / np_val, 4)

            return result if result else None
        except Exception:
            return None

    # ============================================================
    # 4. 行业分类(同花顺行业板块)
    # ============================================================

    def get_industry_map(self) -> Dict[str, str]:
        """
        获取股票 → 行业映射(同花顺行业分类,近似申万二级粒度).
        """
        if self._industry_map is not None:
            return self._industry_map

        cached = load_cache(CACHE_INDUSTRY) if self.use_cache else None
        if cached is not None:
            self._industry_map = cached
            logger.info("命中缓存: %s (%d 条)", CACHE_INDUSTRY, len(cached))
            return cached

        logger.info("正在获取同花顺行业分类...")
        industry_map = self._fetch_industry_from_ths()

        self._industry_map = industry_map
        save_cache(CACHE_INDUSTRY, industry_map)
        logger.info("行业分类完成: %d 只股票, %d 个行业",
                    len(industry_map),
                    len(set(industry_map.values())))
        return industry_map

    @retry_on_error(max_retries=2, delay=0.5)
    def _fetch_industry_from_ths(self) -> Dict[str, str]:
        """通过同花顺行业板块获取股票-行业映射"""
        import akshare as ak

        # 获取所有行业板块
        boards = ak.stock_board_industry_name_ths()
        if boards is None or boards.empty:
            logger.warning("同花顺行业板块列表为空")
            return {}

        industry_map = {}
        total_boards = len(boards)

        for idx, (_, row) in enumerate(boards.iterrows()):
            board_name = row.get("name", "")
            board_code = row.get("code", "")
            if not board_code or not board_name:
                continue

            try:
                cons = ak.stock_board_industry_cons_ths(symbol=board_code)
                if cons is None or cons.empty:
                    continue
                code_col = "代码" if "代码" in cons.columns else cons.columns[0]
                for _, sr in cons.iterrows():
                    sym = clean_symbol(str(sr[code_col]))
                    industry_map[sym] = board_name
            except Exception:
                continue

        return industry_map

    # ============================================================
    # 5. PE/PB 历史分位(简化版:用腾讯当前PE/PB在全市场排名近似)
    # ============================================================

    def get_pe_pb_history(self, symbols: List[str]) -> pd.DataFrame:
        """
        PE/PB 分位数据.

        注: 历史分位需要5年日频PE/PB序列数据,免费API难以获取.
        当前方案: 在候选池内做截面排名作为近似分位.
        这等价于「当前PE在全市场处于什么分位」,虽然不是完美的时间序列
        分位,但在截面上提供了相对估值信息.

        返回:
            DataFrame, index=股票代码, columns=[pe_pct, pb_pct]
        """
        cache_name = f"{CACHE_PE_HISTORY}_{len(symbols)}"
        cached = load_cache(cache_name) if self.use_cache else None
        if cached is not None:
            return cached

        logger.info("计算 PE/PB 截面分位 (%d 只)...", len(symbols))

        # 获取腾讯行情(含PE/PB)
        quotes = self._fetch_tencent_quotes_batch(symbols)
        if quotes.empty:
            return pd.DataFrame(columns=["pe_pct", "pb_pct"])

        quotes = quotes.set_index("symbol")

        result = pd.DataFrame(index=pd.Index(symbols, name="symbol"))
        result["pe_pct"] = quotes["pe"].rank(pct=True) * 100
        result["pb_pct"] = quotes["pb"].rank(pct=True) * 100

        save_cache(cache_name, result)
        return result

    # ============================================================
    # 6. 宏观数据
    # ============================================================

    def get_macro_data(self, ref_date: Optional[str] = None) -> dict:
        """
        获取宏观指标:PMI,M2增速,社融增速.

        防前视偏差: 根据 ref_date 截断历史序列,只取该日期之前的数据.
        """
        cache_name = f"{CACHE_MACRO}_{ref_date}" if ref_date else CACHE_MACRO
        cached = load_cache(cache_name) if self.use_cache else None
        if cached is not None:
            return cached

        logger.info("获取宏观数据 (ref=%s)...", ref_date or "latest")
        result = {}

        pmi = self._fetch_pmi(ref_date)
        if pmi is not None and not pmi.empty:
            result["pmi"] = float(pmi.iloc[-1])
            pmi_series = pmi.astype(float)
            result["pmi_pct"] = round((pmi_series < result["pmi"]).mean() * 100, 2)

        m2 = self._fetch_m2(ref_date)
        if m2 is not None and not m2.empty:
            result["m2_yoy"] = float(m2.iloc[-1])
            m2_series = m2.astype(float)
            result["m2_yoy_pct"] = round((m2_series < result["m2_yoy"]).mean() * 100, 2)

        sf = self._fetch_social_financing(ref_date)
        if sf is not None and not sf.empty:
            result["social_fin_yoy"] = float(sf.iloc[-1])
            sf_series = sf.astype(float)
            result["social_fin_yoy_pct"] = round(
                (sf_series < result["social_fin_yoy"]).mean() * 100, 2
            )

        save_cache(cache_name, result)
        return result

    @staticmethod
    def _truncate_by_date(df: pd.DataFrame, ref_date: str) -> pd.DataFrame:
        """根据 ref_date 截断 DataFrame,只保留该日期之前的行"""
        if not ref_date or df is None or df.empty:
            return df

        ref_ts = pd.Timestamp(ref_date)
        # 找日期列
        date_col = None
        for c in df.columns:
            if "月" in str(c) or "日期" in str(c) or "date" in str(c).lower():
                date_col = c
                break

        if date_col is not None:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            df = df[dates <= ref_ts]
            if df.empty:
                logger.warning("宏观数据: 截断后为空 (ref=%s)", ref_date)

        return df

    @retry_on_error(max_retries=2, delay=0.5)
    def _fetch_pmi(self, ref_date: Optional[str] = None) -> Optional[pd.Series]:
        import akshare as ak
        df = ak.macro_china_pmi()
        if df is None or df.empty:
            return None
        df = self._truncate_by_date(df, ref_date)
        if df is None or df.empty:
            return None
        for c in ["制造业PMI", "PMI", "数值"]:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce").dropna()
        return None

    @retry_on_error(max_retries=2, delay=0.5)
    def _fetch_m2(self, ref_date: Optional[str] = None) -> Optional[pd.Series]:
        import akshare as ak
        df = ak.macro_china_money_supply()
        if df is None or df.empty:
            return None
        df = self._truncate_by_date(df, ref_date)
        if df is None or df.empty:
            return None
        for c in ["M2同比", "M2同比增长", "M2供应量同比增长"]:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce").dropna()
        if "M2" in df.columns or "货币和准货币(M2)" in df.columns:
            col = "M2" if "M2" in df.columns else "货币和准货币(M2)"
            m2_abs = pd.to_numeric(df[col], errors="coerce").dropna()
            return m2_abs.pct_change(12).dropna() * 100
        return None

    @retry_on_error(max_retries=2, delay=0.5)
    def _fetch_social_financing(self, ref_date: Optional[str] = None) -> Optional[pd.Series]:
        import akshare as ak
        try:
            df = ak.macro_china_shrzgm()
        except Exception:
            return None
        if df is None or df.empty:
            return None
        df = self._truncate_by_date(df, ref_date)
        if df is None or df.empty:
            return None
        for c in ["社会融资规模增量", "社融增量"]:
            if c in df.columns:
                series = pd.to_numeric(df[c], errors="coerce").dropna()
                return series.pct_change(12).dropna() * 100
        return None


# ============================================================
# 模块自检
# ============================================================

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("AgentData 模块加载成功")
    agent = AgentData(use_cache=False)

    print("\n--- 测试1: 获取股票池 ---")
    universe = agent.get_stock_universe()
    print(f"候选池数量: {len(universe)}")
    if not universe.empty:
        print(universe.head(5)[["symbol", "name", "close", "pe", "pb", "turnover"]].to_string())

    print("\n--- 测试2: 行业分类(前10条)---")
    industry = agent.get_industry_map()
    for i, (sym, ind) in enumerate(industry.items()):
        if i >= 10:
            break
        print(f"  {sym} → {ind}")

    print("\n--- 测试3: 单只股票历史K线 ---")
    prices = agent.get_daily_prices(["600519.XSHG"], "2026-06-01", "2026-07-05")
    if not prices.empty:
        print(prices.tail(3).to_string())

    print("\n--- 测试4: 财务数据 ---")
    fin = agent.get_financials(["600519.XSHG"], "2026-07-05")
    if not fin.empty:
        print(fin.to_string())

    print("\n--- 测试5: 宏观数据 ---")
    macro = agent.get_macro_data()
    for k, v in macro.items():
        if not k.endswith("_pct"):
            print(f"  {k}: {v}")
