# 聚宽量化策略

基于 JoinQuant 平台的量化交易策略集合，目前包含双均线+ADX趋势跟踪策略。

## 环境安装

```bash
pip install -r requirements.txt
```

## 项目结构

```
my_strategy_code/
├── strategies/                  # 策略（每个文件独立可运行）
│   ├── double_ma_adx.py         # 双均线+ADX趋势
│   ├── mean_reversion.py        # 均值回归
│   └── momentum_breakout.py     # 动量突破
│
├── config/                      # 参数配置
│   ├── strategy_params.py       # 策略参数
│   ├── universe.py              # 股票池
│   └── risk_control.py          # 风控参数
│
├── core/                        # 本地回测工具
│   ├── indicators.py            # 技术指标
│   ├── risk_manager.py          # 风控逻辑
│   └── data_utils.py            # 数据处理
│
├── research/                    # 因子分析、参数寻优
│   ├── factor_analysis.ipynb
│   └── parameter_optimization.py
│
├── tests/                       # 本地测试
│   ├── test_indicators.py
│   └── test_risk_manager.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

> **说明**：`strategies/` 内每个文件完整独立，可直接复制到聚宽平台运行；`core/` 仅供本地回测使用，聚宽不支持自定义 import。

## 策略说明

### 双均线+ADX趋势（`joinquant_strategy.py`）

- **标的**：新易盛（300502.XSHE）
- **信号**：5/10日双均线金叉/死叉，叠加14日ADX趋势过滤
- **执行**：每天14:50计算信号，以实时价成交
- **基准**：沪深300

## 运行方式

### 聚宽平台运行

策略文件直接在 [JoinQuant](https://www.joinquant.com) 网页平台运行：

1. 登录后进入「我的策略」→「新建策略」
2. 将策略代码粘贴到编辑器
3. 点击「运行回测」或「模拟交易」

### 本地开发

core、tests、research 等模块为标准 Python，本地运行：

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
python -m pytest tests/

# 参数优化
python research/parameter_optimization.py

# Jupyter 研究
jupyter notebook research/
```

## 聚宽注意事项

| 事项 | 说明 |
|------|------|
| import限制 | 聚宽不支持自定义模块import，所有函数必须写在策略文件里 |
| 未来函数 | `attribute_history()` 不含当日，`iloc[-1]` = 昨日数据 |
| g全局变量 | 用聚宽内置 `g` 对象存策略状态，非Python全局变量 |
| 手续费 | 回测默认万2.5+滑点，实盘可用 `set_order_cost()` 覆盖 |

## 多策略管理

- **单仓库多分支**：所有策略放一个仓库，简单直观
- **一策略一仓库**：每个策略独立版本管理，适合多策略并行迭代
