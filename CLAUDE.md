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
    ├── config.py                  # 全局配置
    ├── utils.py                   # 工具函数
    ├── agent_data.py              # 数据获取 Agent
    ├── agent_factor.py            # 因子打分 Agent
    ├── agent_risk.py              # 风控 Agent
    ├── agent_macro.py             # 宏观仓位 Agent
    ├── agent_portfolio.py         # 组合构建 Agent
    ├── coordinator.py             # 总调度器
    ├── main.py                    # CLI 入口
    ├── output/                    # 输出结果(.gitignore)
    └── data_cache/                # 数据缓存(.gitignore)
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

## 聚宽平台约束

| 约束 | 说明 |
|------|------|
| import 限制 | 不支持自定义模块 import,需内联所有依赖函数 |
| 未来函数 | `attribute_history()` 不含当日,`iloc[-1]` = 昨日数据 |
| g 全局变量 | 使用聚宽内置 `g` 对象存储策略状态 |
| 手续费 | 回测默认万2.5+滑点 |
