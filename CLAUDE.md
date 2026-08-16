# 量化策略研究平台 - CLAUDE.md

## 项目概述

基于 JoinQuant 的量化交易策略研究,回测与可视化平台.当前处于 **v1(已完成)→ v2(已完成)→ v3(规划中)** 阶段.

## 环境

- Python 3.12 项目
- Windows 11,终端用 bash(Git Bash)
- 依赖:numpy, pandas, matplotlib, scipy, statsmodels, scikit-learn, ta, jupyter, akshare, jqdatasdk
- 平台:JoinQuant(聚宽)用于回测/模拟交易

## 项目结构

```
my_strategy_code/
├── README.md
├── requirements.txt
├── CLAUDE.md                      ← 本文件
├── .gitignore
├── Quant Research Platform v1/    ← 原型(已完成)
│   ├── joinquant_strategy.py      # 单策略文件(MA+ADX)
│   ├── 300308.XSHE.png
│   └── 300502.XSHE.png
└── Quant Research Platform v2/    ← 多因子选股系统(已完成)
    ├── main.py                    # CLI 入口
    ├── config/                    # 配置部分
    │   ├── __init__.py             # 统一导出 settings + utils
    │   ├── settings.py             # 全局配置(路径/因子权重/风控阈值/组合约束)
    │   └── utils.py                # 工具函数(日志/重试/缓存/统计)
    ├── agents/                    # 核心代码部分
    │   ├── __init__.py             # 统一导出所有 Agent + Coordinator
    │   ├── agent_data.py           # 数据获取 Agent
    │   ├── agent_factor.py         # 因子打分 Agent
    │   ├── agent_risk.py           # 风控 Agent
    │   ├── agent_macro.py          # 宏观仓位 Agent
    │   ├── agent_portfolio.py      # 组合构建 Agent
    │   └── coordinator.py          # 总调度器
    ├── output/                    # 数据部分 - 输出结果(.gitignore)
    └── data_cache/                # 数据部分 - 数据缓存(.gitignore)
```

## ⚠️ API 可用性(关键)

网络环境限制——东方财富系 API 全部被墙,必须使用替代数据源:

| 数据类型 | ❌ 不可用 | ✅ 可用替代 |
|----------|-----------|------------|
| 实时行情(PE/PB/市值) | `stock_zh_a_spot_em()` | 腾讯 HTTP API (`qt.gtimg.cn`) |
| 历史K线 | `stock_zh_a_hist()` | `stock_zh_a_daily()` [新浪源] |
| 财务数据 | East Money 财务接口 | `stock_financial_analysis_indicator()` |
| 股票列表 | - | `stock_info_a_code_name()` |
| 行业分类 | - | `stock_board_industry_name_ths()` + `_cons_ths()` |
| 宏观数据 | - | `macro_china_*()` |

## v2 多因子选股系统

### 因子框架
```
总分 = 估值(15%) + 质量(25%) + 动量(20%) + 波动反转(15%) - 风控扣分(25%)
```

| 模块 | 因子 | 方向 |
|------|------|------|
| 估值 | PE分位,PB分位 | 越低越好 |
| 质量 | ROE,毛利率,经营现金流/净利润 | 越高越好 |
| 动量 | 1月收益率(60%),3月收益率(40%) | 越高越好 |
| 波动反转 | 20日波动率(60%),5日反转(40%) | 波动正向/反转负向 |
| 风控扣分 | 行业PE>80%分位,行业涨幅>15%,个股PE>85%分位,PB>85%分位,PEG>2 | 每次触发扣5分,封顶25分 |

### 硬门槛过滤
全A股(5528) → 排除ST/北交所 → PE/PB>0 → 成交额≥5000万 → Top 1000

### 宏观仓位
PMI(40%) + M2(30%) + 社融(30%) → 景气度分位 → 仓位映射 30%~80%

### 组合约束
- Top 50 精选
- 单票 ≤ 5%
- 单行业 ≤ 20%
- 月度调仓

### 运行方式
```bash
cd "Quant Research Platform v2"
python main.py screen     # 当前选股(首次5-30分钟,之后缓存加速)
python main.py backtest   # 两年逐月回测
python main.py check      # 模块检查
```

### 首次运行耗时参考
- 腾讯行情(全市场PE/PB): ~1分钟
- 历史K线(1000只×2年): ~30分钟
- 财务数据(1000只): ~10分钟
- 因子计算+风控+组合: <1分钟
- 缓存后秒级加载

## v1 策略概要

- **策略**: 双均线(MA5/MA10)+ ADX(14, 阈值30)趋势跟踪
- **标的**: 300502.XSHE(新易盛)
- **基准**: 000300.XSHG(沪深300)
- **执行**: 每天 14:50

## ✅ 已修复问题 (2026-07-14)

以下问题已全部修复，回测结果可信。

### ✅ P0: 财务数据前视偏差 — 已修复

**修复**：`_fetch_single_financial()` 新增 `ref_date` 参数，通过 `_get_available_report_date()` 根据 `FINANCIAL_LAG_MONTHS` 计算最新可用财报日期，过滤掉未来数据。

**涉及文件**：
- `agents/agent_data.py` — `_get_available_report_date()` + `_fetch_single_financial(symbol, ref_date)`

### ✅ P0: 宏观数据前视偏差 — 已修复

**修复**：`get_macro_data(ref_date)` 新增 `_truncate_by_date()` 方法截断历史序列，缓存 key 改为 `f"{CACHE_MACRO}_{ref_date}"`。三个宏观 fetch 函数均接收 `ref_date` 参数。

**涉及文件**：
- `agents/agent_data.py` — `get_macro_data()` + `_fetch_pmi(ref_date)` + `_fetch_m2(ref_date)` + `_fetch_social_financing(ref_date)`

### ✅ P1: PEG 风控检查失效 — 已修复

**修复**：`get_financials()` 新增 `universe_df` 参数，用 universe 中的 PE 除以财务数据中的 `np_yoy` 计算 PEG（仅当 PE>0 且 np_yoy>0 时有效）。Coordinator 已传入 universe。

**涉及文件**：
- `agents/agent_data.py` — `get_financials(symbols, ref_date, universe_df=universe)`
- `agents/coordinator.py` — `run_screening()` 传 `universe_df=universe`

### ✅ P1: 回测缺少基准对比和交易成本 — 已修复

**修复**：
- 新增 `_calc_benchmark_return()` 计算沪深300基准收益
- `_calc_holding_return()` 扣除交易成本（佣金万2.5×2 + 印花税0.05% + 滑点0.1%×2 ≈ 0.085%）
- `_backtest_summary()` 新增基准对比指标（超额收益、信息比率、跟踪误差）
- 配置项: `config/settings.py` 新增 `COMMISSION_RATE`, `STAMP_DUTY_RATE`, `SLIPPAGE_RATE`, `TURNOVER_COST_RATE`, `BENCHMARK_SYMBOL`

### ✅ P2: K线拉取并行化 — 已修复

**修复**：`get_daily_prices()` 改用 `ThreadPoolExecutor(max_workers=10)` + `as_completed` 并行拉取，300 只约 3-5 分钟（原 30 分钟）。

## ✅ 已修复问题 (2026-07-15)

### ✅ P0: 候选池历史回测前视偏差（腾讯实时行情污染历史）— 已修复

**修复**：`get_stock_universe(date)` 新增「实时 vs 历史」双模式：
- 距今日 ≤ `HISTORICAL_THRESHOLD_DAYS=14` 天 → 实时模式：腾讯行情（PE/PB/成交额完整）
- 超过阈值 → 历史回测模式：`_get_stock_universe_historical()` 并行拉 450 天 K 线，用 `listing_days`（上市天数）+ `avg_turnover_20d`（20 日日均成交额）做过滤

**已知限制**：无免费历史 PE/PB 数据源，跳过 PE>0/PB>0/PE≤500 硬门槛；此缺口由 PE/PB 截面百分位因子得分间接补偿（日志有警告）。

**涉及文件**：
- `agents/agent_data.py` — `_get_stock_universe_realtime()` + `_get_stock_universe_historical()` + `_fetch_full_kline_with_amount()` + `_fetch_single_stock_history_full()`
- `config/settings.py` — `HISTORICAL_THRESHOLD_DAYS = 14`

### ✅ P1: `MIN_LISTING_DAYS` 配置无效 — 已修复

**修复**：历史回测模式内联实现「`listing_days >= MIN_LISTING_DAYS=250`」过滤逻辑（K 线行数近似上市天数）。

### ✅ P1: 宏观仓位现金部分零收益 — 已修复

**修复**：`run_backtest()` 循环中根据每期 `position_ratio` 拆分组合收益：
```
组合月收益 = 股票收益 × position_ratio + 无风险月利率 × (1 - position_ratio)
```
配置：`RISK_FREE_RATE_ANNUAL = 0.025`（年化 2.5%，月化 ≈ 0.208%）

### ✅ P2: 流动性门槛用「当日成交额」非「日均」— 已修复

**修复**：历史回测模式下成交额用近 20 个交易日均值（`tail(20).mean()`），优先使用 akshare 原生 `turnover` 列，缺省用 `volume × close` 近似。

### ✅ P0: 旧 data_cache 含前视数据 — 已清空

手动删除整个 `Quant Research Platform v2/data_cache/` 目录，下次回测从干净缓存开始。

---

## 聚宽平台约束

| 约束 | 说明 |
|------|------|
| import 限制 | 不支持自定义模块 import,需内联所有依赖函数 |
| 未来函数 | `attribute_history()` 不含当日,`iloc[-1]` = 昨日数据 |
| g 全局变量 | 使用聚宽内置 `g` 对象存储策略状态 |
| 手续费 | 回测默认万2.5+滑点 |
