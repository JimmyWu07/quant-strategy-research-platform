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
    FINANCIAL_LAG_MONTHS,
    INDUSTRY_STANDARD, DATA_CACHE_DIR,
)
from utils import (
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
        """
        cache_name = f"{CACHE_UNIVERSE}_{date}" if date else CACHE_UNIVERSE
        cached = load_cache(cache_name) if self.use_cache else None
        if cached is not None:
            logger.info("命中缓存: %s,共 %d 只股票", cache_name, len(cached))
            return cached

        logger.info("正在获取 A 股全市场数据...")

        # 1a. 获取股票列表
        stock_list = self._get_stock_list()
        all_symbols = stock_list["symbol"].tolist()
        logger.info("全市场: %d 只股票", len(all_symbols))

        # 1b. 批量获取腾讯行情数据(PE/PB/市值/成交额/价格)
        quotes = self._fetch_tencent_quotes_batch(all_symbols)
        if quotes.empty:
            logger.error("腾讯行情数据获取失败")
            return pd.DataFrame()

        # 1c. 合并名称(quotes已有name,stock_list的name作为补充)
        name_lookup = stock_list.set_index("symbol")["name"]
        if "name" in quotes.columns:
            quotes["name"] = quotes["name"].fillna(quotes["symbol"].map(name_lookup))
        else:
            quotes["name"] = quotes["symbol"].map(name_lookup)

        # 1d. 硬门槛过滤
        df = self._apply_hard_filters(quotes)
        logger.info("股票池过滤完成: 全市场 %d → 候选池 %d 只", len(quotes), len(df))
        save_cache(cache_name, df)
        return df

    def _apply_hard_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用硬门槛过滤"""
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
        批量获取股票历史日线数据(带线程超时保护).

        返回:
            DataFrame, index=日期, columns=股票代码, values=收盘价
        """
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        cache_name = f"{CACHE_PRICES}_{start_date}_{end_date}"
        cached = load_cache(cache_name) if self.use_cache else None
        if cached is not None:
            logger.info("命中缓存: %s", cache_name)
            return cached

        logger.info("获取 %d 只股票历史K线 (%s → %s)...", len(symbols), start_date, end_date)

        price_dict = {}
        failed = 0
        PER_STOCK_TIMEOUT = 15  # 单只15秒超时

        for i, sym in enumerate(symbols):
            if i % 50 == 0:
                logger.info(progress_bar(i, len(symbols), "拉取K线"))

            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self._fetch_single_stock_history, sym, start_date, end_date)
                    try:
                        hist = future.result(timeout=PER_STOCK_TIMEOUT)
                    except FuturesTimeout:
                        logger.warning("%s K线超时 (>%ds), 跳过", sym, PER_STOCK_TIMEOUT)
                        failed += 1
                        continue

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
    ) -> pd.DataFrame:
        """
        获取财务指标(ROE,毛利率,经营现金流,净利润增速).

        返回:
            DataFrame, index=股票代码, columns=[roe, gross_margin, cfo_to_np, np_yoy]
        """
        cache_name = f"{CACHE_FINANCIALS}_{ref_date}"
        cached = load_cache(cache_name) if self.use_cache else None
        if cached is not None:
            logger.info("命中缓存: %s", cache_name)
            return cached

        logger.info("获取 %d 只股票财务数据 (ref=%s)...", len(symbols), ref_date)

        results = {}
        failed = 0

        for i, sym in enumerate(symbols):
            if i % 200 == 0:
                logger.info(progress_bar(i, len(symbols), "拉取财务"))

            try:
                fin = self._fetch_single_financial(sym)
                if fin:
                    results[sym] = fin
                else:
                    failed += 1
            except Exception:
                failed += 1

        logger.info("财务数据获取完成: 成功 %d, 失败 %d", len(results), failed)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results).T
        df.index.name = "symbol"
        save_cache(cache_name, df)
        return df

    @retry_on_error(max_retries=2, delay=0.5)
    def _fetch_single_financial(self, symbol: str) -> Optional[dict]:
        """获取单只股票核心财务指标(最新一期)"""
        import akshare as ak
        code = symbol.replace(".XSHG", "").replace(".XSHE", "")

        try:
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2020")
            if df is None or df.empty:
                return None

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
        """获取宏观指标:PMI,M2增速,社融增速."""
        cached = load_cache(CACHE_MACRO) if self.use_cache else None
        if cached is not None:
            return cached

        logger.info("获取宏观数据...")
        result = {}

        pmi = self._fetch_pmi()
        if pmi is not None and not pmi.empty:
            result["pmi"] = float(pmi.iloc[-1])
            pmi_series = pmi.astype(float)
            result["pmi_pct"] = round((pmi_series < result["pmi"]).mean() * 100, 2)

        m2 = self._fetch_m2()
        if m2 is not None and not m2.empty:
            result["m2_yoy"] = float(m2.iloc[-1])
            m2_series = m2.astype(float)
            result["m2_yoy_pct"] = round((m2_series < result["m2_yoy"]).mean() * 100, 2)

        sf = self._fetch_social_financing()
        if sf is not None and not sf.empty:
            result["social_fin_yoy"] = float(sf.iloc[-1])
            sf_series = sf.astype(float)
            result["social_fin_yoy_pct"] = round(
                (sf_series < result["social_fin_yoy"]).mean() * 100, 2
            )

        save_cache(CACHE_MACRO, result)
        return result

    @retry_on_error(max_retries=2, delay=0.5)
    def _fetch_pmi(self) -> Optional[pd.Series]:
        import akshare as ak
        df = ak.macro_china_pmi()
        if df is None or df.empty:
            return None
        for c in ["制造业PMI", "PMI", "数值"]:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce").dropna()
        return None

    @retry_on_error(max_retries=2, delay=0.5)
    def _fetch_m2(self) -> Optional[pd.Series]:
        import akshare as ak
        df = ak.macro_china_money_supply()
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
    def _fetch_social_financing(self) -> Optional[pd.Series]:
        import akshare as ak
        try:
            df = ak.macro_china_shrzgm()
        except Exception:
            return None
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
