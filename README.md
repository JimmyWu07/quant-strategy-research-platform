# 量化策略研究平台

基于 JoinQuant 的量化交易策略研究、回测与可视化平台。当前包含 **V1（趋势跟踪）** 和 **V2（多因子选股）** 两套策略系统。

## 环境

- Python 3.12
- Windows 11，终端使用 bash (Git Bash)
- 依赖：numpy, pandas, matplotlib, scipy, statsmodels, scikit-learn, ta, jupyter, akshare, jqdatasdk

```bash
pip install -r requirements.txt
```

## 项目结构

```
my_strategy_code/
├── README.md
├── requirements.txt
├── CLAUDE.md
├── .gitignore
│
├── Quant Research Platform v1/       ← 原型：单策略趋势跟踪
│   ├── joinquant_strategy.py         # 双均线+ADX 策略（聚宽平台用）
│   ├── 300308.XSHE.png               # 回测结果图（示例）
│   └── 300502.XSHE.png               # 回测结果图（新易盛）
│
└── Quant Research Platform v2/       ← 多因子选股系统
    ├── config.py                     # 全局配置（资金、手续费、滑点等）
    ├── utils.py                      # 工具函数
    ├── agent_data.py                 # 数据获取 Agent
    ├── agent_factor.py               # 因子打分 Agent
    ├── agent_risk.py                 # 风控 Agent
    ├── agent_macro.py                # 宏观仓位 Agent
    ├── agent_portfolio.py            # 组合构建 Agent
    ├── coordinator.py                # 总调度器
    ├── main.py                       # CLI 入口
    ├── output/                       # 输出结果（.gitignore）
    └── data_cache/                   # 数据缓存（.gitignore）
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

多 Agent 协作的因子选股框架，支持本地运行、逐月回测。

### 因子框架

```
总分 = 估值(15%) + 质量(25%) + 动量(20%) + 波动反转(15%) - 风控扣分(25%)
```

| 模块 | 因子 | 方向 |
|------|------|------|
| 估值 | PE 分位、PB 分位 | 越低越好 |
| 质量 | ROE、毛利率、经营现金流/净利润 | 越高越好 |
| 动量 | 1 月收益率(60%)、3 月收益率(40%) | 越高越好 |
| 波动反转 | 20 日波动率(60%)、5 日反转(40%) | 波动正向 / 反转负向 |
| 风控扣分 | 行业 PE>80% 分位、行业涨幅>15%、个股 PE>85% 分位、PB>85% 分位、PEG>2 | 每次 -5 分，封顶 -25 分 |

### 硬门槛过滤

```
全 A 股(5528) → 排除 ST/北交所 → PE/PB > 0 → 成交额 ≥ 5000 万 → Top 1000
```

### 宏观仓位

PMI(40%) + M2(30%) + 社融(30%) → 景气度分位 → 仓位映射 30% ~ 80%

### 组合约束

- Top 50 精选
- 单票 ≤ 5%
- 单行业 ≤ 20%
- 月度调仓

### 运行方式

```bash
cd "Quant Research Platform v2"

python main.py screen     # 当前选股（首次 5-30 分钟，之后缓存加速）
python main.py backtest   # 两年逐月回测
python main.py check      # 模块可用性检查
```

### 首次运行耗时参考

| 步骤 | 耗时 |
|------|------|
| 腾讯行情（全市场 PE/PB） | ~1 分钟 |
| 历史 K 线（1000 只 × 2 年） | ~30 分钟 |
| 财务数据（1000 只） | ~10 分钟 |
| 因子计算 + 风控 + 组合 | <1 分钟 |
| 缓存后 | 秒级加载 |

---

## API 可用性

网络环境限制——东方财富系 API 不可用，使用替代数据源：

| 数据类型 | 替代方案 |
|----------|----------|
| 实时行情（PE/PB/市值） | 腾讯 HTTP API (`qt.gtimg.cn`) |
| 历史 K 线 | `stock_zh_a_daily()` 新浪源 |
| 财务数据 | `stock_financial_analysis_indicator()` |
| 股票列表 | `stock_info_a_code_name()` |
| 行业分类 | `stock_board_industry_name_ths()` + `_cons_ths()` |
| 宏观数据 | `macro_china_*()` |

## 聚宽平台约束

| 约束 | 说明 |
|------|------|
| import 限制 | 不支持自定义模块 import，需内联所有依赖函数 |
| 未来函数 | `attribute_history()` 不含当日，`iloc[-1]` = 昨日数据 |
| g 全局变量 | 使用聚宽内置 `g` 对象存储策略状态 |
| 手续费 | 回测默认万 2.5+滑点 |
