# 量化策略研究平台

基于 JoinQuant 的量化交易策略研究、回测与可视化平台。当前包含 **V1（趋势跟踪）** 和 **V2（多因子选股）** 两套策略系统。

## 环境

- Python 3.12
- Windows 11
- 依赖：numpy, pandas, matplotlib, scipy, statsmodels, scikit-learn, ta, jupyter, akshare, jqdatasdk, python-docx

```bash
pip install -r requirements.txt
```

## 项目结构

```
my_strategy_code/
├── README.md
├── requirements.txt
├── CLAUDE.md                          # AI 项目说明（给 Claude 看的记忆文件）
├── v2模块拆解分析.docx                 # V2 架构详细拆解文档
├── .gitignore
│
├── Quant Research Platform v1/         ← 原型：单策略趋势跟踪
│   ├── joinquant_strategy.py          # 双均线+ADX 策略（聚宽平台用）
│   ├── 300308.XSHE.png                # 回测结果图（示例）
│   └── 300502.XSHE.png                # 回测结果图（新易盛）
│
└── Quant Research Platform v2/        ← 多因子选股系统
    ├── main.py                        # CLI 入口
    ├── config/                        # 配置层
    │   ├── __init__.py                # 统一导出
    │   ├── settings.py                # 全局配置（阈值、权重、成本等）
    │   └── utils.py                   # 工具函数（重试、缓存、打分）
    ├── agents/                        # 核心 Agent 层
    │   ├── __init__.py                # 统一导出
    │   ├── agent_data.py              # 数据获取 Agent
    │   ├── agent_factor.py            # 因子打分 Agent
    │   ├── agent_risk.py              # 风控 Agent
    │   ├── agent_macro.py             # 宏观仓位 Agent
    │   ├── agent_portfolio.py         # 组合构建 Agent
    │   └── coordinator.py             # 总调度器（串联所有 Agent）
    ├── output/                        # 输出结果（.gitignore）
    └── data_cache/                    # 数据缓存（.gitignore）
```

---

## V1 — 双均线+ADX 趋势跟踪

单策略原型，关注趋势跟随信号，部署于聚宽平台。

| 项目 | 说明 |
|------|------|
| 策略 | 双均线（MA5/MA10）+ ADX（14，阈值 30）趋势跟踪 |
| 标的 | 300502.XSHE（新易盛） |
| 基准 | 000300.XSHG（沪深 300） |
| 执行 | 每天 14:50 |

### 使用方式

将 `Quant Research Platform v1/joinquant_strategy.py` 全部代码复制到 [JoinQuant](https://www.joinquant.com) 策略编辑器：
1. 登录 → 我的策略 → 新建策略
2. 粘贴代码，运行回测或模拟交易

---

## V2 — 多因子选股系统

多 Agent 协作的因子选股框架，支持本地单次选股和逐月回测。已修复所有前视偏差 bug，回测结果具备参考性。

### 因子框架

```
最终得分 = 估值得分×15% + 质量得分×25% + 动量得分×20% + 波动反转×15% − 风控扣分(封顶25%)
组合收益 = 股票收益×仓位比例 + 无风险月利率×(1−仓位比例)
```

| 模块 | 权重 | 子因子 | 方向 |
|------|------|--------|------|
| **估值** | 15% | PE 分位、PB 分位（各 50%） | 越低越好（反向） |
| **质量** | 25% | ROE(40%)、毛利率(35%)、CFO/NP(25%) | 越高越好（正向） |
| **动量** | 20% | 1 月收益率(60%)、3 月收益率(40%) | 越高越好（正向） |
| **波动反转** | 15% | 20 日波动率(60%)、5 日反转(40%) | 波动正向 / 反转负向 |
| **风控扣分** | 封顶 25% | 行业过热×2 + 个股高估×2 + PEG | 每项触发 −5 分 |

### 硬门槛过滤（候选池构建）

```
全 A 股(≈5500)
  → 排除 ST/*ST
  → 排除北交所（8 开头）
  → （实时模式）PE/PB>0、PE≤500、成交≥5000万
  → （历史模式）上市≥250 个交易日、20 日日均成交≥5000万
  → 按成交降序取 Top 300
```

历史回测无法拿到当时 PE/PB，故用流动性+上市天数近似。极端估值股票在行业内排名阶段被自然排除。

### 宏观仓位控制（30% ~ 80%）

```
综合景气度分位 = PMI 分位×40% + M2 分位×30% + 社融分位×30%

  景气度 < 30%   → 仓位 30%（防御，硬地板）
  30% ~ 70%      → 30%→80% 线性插值
  景气度 > 70%   → 仓位 80%（积极，80%天花板，永不满仓留现金）
```

### 风控扣分规则（−5 分/项，封顶 −25 分）

| 检查项 | 触发阈值 |
|--------|----------|
| 行业 PE 过热 | 行业 PE 分位中位数 > 80% |
| 行业涨幅过热 | 行业近 1 月平均涨幅 > 15% |
| 个股 PE 高估 | 个股 PE 分位 > 85% |
| 个股 PB 高估 | 个股 PB 分位 > 85% |
| PEG 过高 | PE / 净利润同比增速 > 2.0（增速为负不扣） |

### 组合约束

- **Top 50** 精选（按 final_score 降序 + 行业约束）
- 单票 ≤ 5%
- 单行业 ≤ 20%（最多 10 只）
- 月度调仓（月末最后一个交易日）
- 权重分配：等权（当前）

### 运行方式

```bash
cd "Quant Research Platform v2"

python main.py check      # 模块依赖检查（必跑，确认环境）
python main.py screen     # 当前选股（首次 3-5 分钟，缓存后秒级）
python main.py backtest   # 逐月回测（默认 2024-07 → 2026-07，24 期）
```

回测输出：
- `output/backtest_monthly_returns.csv` — 月收益序列（策略 vs 沪深300）
- `output/backtest_performance_summary.json` — 绩效指标（年化/夏普/最大回撤/超额/信息比率等）
- `output/screening_YYYYMMDD.csv` — 每期 Top 50 精选明细
- `output/screening_history_top10.csv` — 每期 Top 10 历史汇总

### 首次运行耗时参考（已并行优化）

| 步骤 | 耗时 | 说明 |
|------|------|------|
| 腾讯行情（全市场批量） | ~1 分钟 | 80 只/批，~70 次请求 |
| 历史 K 线（Top 300 × 2 年） | ~3-5 分钟 | 10 线程并行，之前 30 分钟 |
| 财务数据（300 只） | ~6-10 分钟 | 5 线程并行 + 15 秒超时容错 |
| 因子计算 + 风控 + 组合构建 | <1 分钟 | 纯内存运算 |
| 行业分类 / 宏观数据 / 沪深300 | <30 秒 | 一次性拉取，缓存永久复用 |
| **缓存后二次运行** | **秒级** | 所有数据 pickle 落盘 |

### 回测精度说明（已修复的 P0 bug）

| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| 财务数据前视 | 永远拿最新财报（含未来数据） | `_get_available_report_date()` 按 `FINANCIAL_LAG_MONTHS` 过滤可用财报 |
| 宏观数据前视 | 永远拿最新值（含未来数据） | `_truncate_by_date()` 按调仓日截断序列，缓存 key 含日期 |
| 候选池过滤前视 | 拿实时 PE/PB 过滤历史 | 历史模式：腾讯行情初筛→Top 300→K 线验证 250 天上市+20 日均成交 |
| PEG 风控 | 始终失效（无 peg 列） | PE / np_yoy 实时计算 |
| 基准对比 | 无 | 沪深300 指数一次性拉取并缓存 |
| 交易成本 | 0 | 佣金万 2.5 + 印花税 0.05% + 滑点 0.1% ≈ 单次 0.085% |
| 无风险利率 | 现金算 0% | 现金部分年化 2.5%（月化 ≈ 0.208%） |

---

## API 可用性

网络环境限制——东方财富系 API 不可用，使用以下替代数据源：

| 数据类型 | 数据源 | 接口 |
|----------|--------|------|
| 全 A 股列表 | akshare | `stock_info_a_code_name()` |
| 实时行情（PE/PB/市值/成交） | 腾讯 HTTP API | `qt.gtimg.cn`（批量 80 只/次） |
| 历史 K 线（前复权） | akshare（新浪源） | `stock_zh_a_daily()` |
| 财务数据（ROE/毛利率/CFO/NP 增速） | akshare | `stock_financial_analysis_indicator()` |
| 行业分类（同花顺） | akshare | `stock_board_industry_name_ths()` + `_cons_ths()` |
| 宏观数据（PMI/M2/社融） | akshare | `macro_china_pmi / money_supply / shrzgm` |
| 沪深 300 指数（前复权） | akshare（新浪源） | `stock_zh_index_daily(symbol="sh000300")` |

所有数据均 pickle 落盘至 `data_cache/`，二次运行秒级加载。

## 聚宽平台约束（V1 适用）

| 约束 | 说明 |
|------|------|
| import 限制 | 不支持自定义模块 import，需内联所有依赖函数 |
| 未来函数 | `attribute_history()` 不含当日，`iloc[-1]` = 昨日数据 |
| g 全局变量 | 使用聚宽内置 `g` 对象存储策略状态 |
| 手续费 | 回测默认万 2.5 + 滑点 |
