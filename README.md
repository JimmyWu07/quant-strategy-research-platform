# 量化策略研究平台

基于 JoinQuant 的量化交易策略研究,回测与可视化平台.

## 环境安装

```bash
pip install -r requirements.txt
```

## 项目结构

```
quant-strategy-research-platform/
│
├── README.md
├── requirements.txt
│
├── configs/
│   └── config.py              # 全局参数(资金,手续费,滑点等)
│
├── data/
│   └── stock_pool.csv         # 股票池
│
├── strategies/
│   ├── ma_adx_strategy.py     # 双均线+ADX趋势
│   ├── rsi_strategy.py        # RSI超买超卖
│   └── momentum_strategy.py   # 动量突破
│
├── backtest/
│   ├── performance.py         # 绩效分析(夏普,最大回撤等)
│   └── risk.py                # 风控模块
│
├── research/
│   └── factor_analysis.py     # 因子分析
│
├── joinquant/
│   └── main_strategy.py       # 聚宽平台一体化策略入口
│
└── app/
    └── streamlit_app.py       # 本地可视化面板
```

## 运行方式

### 聚宽平台

将 `joinquant/main_strategy.py` 代码复制到 [JoinQuant](https://www.joinquant.com) 策略编辑器:

1. 登录后进入「我的策略」→「新建策略」
2. 粘贴代码,运行回测或模拟交易

### 本地回测与可视化

```bash
# 启动可视化面板
streamlit run app/streamlit_app.py

# 因子分析
python research/factor_analysis.py

# 回测绩效
python backtest/performance.py
```

## 聚宽注意事项

| 事项 | 说明 |
|------|------|
| import限制 | 聚宽不支持自定义模块import,`joinquant/main_strategy.py` 需内联所有依赖函数 |
| 未来函数 | `attribute_history()` 不含当日,`iloc[-1]` = 昨日数据 |
| g全局变量 | 使用聚宽内置 `g` 对象存储策略状态 |
| 手续费 | 回测默认万2.5+滑点,实盘可用 `set_order_cost()` 覆盖 |
