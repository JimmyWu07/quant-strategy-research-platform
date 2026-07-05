"""
v2 多因子选股系统 - 入口

用法:
  # 当前筛查(生成今日精选 50 只)
  python main.py screen

  # 逐月回测(2024.07 → 2026.07)
  python main.py backtest

  # 快速检查所有模块是否正常
  python main.py check
"""

import io
import sys
from datetime import datetime
from pathlib import Path

# 修复 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 确保项目根目录在 Python path 中
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_screen():
    """当前选股"""
    from coordinator import Coordinator

    print("\n" + "█" * 60)
    print("  多因子选股系统 v2 — 当前筛选")
    print("█" * 60)

    coord = Coordinator(use_cache=True)
    date = datetime.now().strftime("%Y-%m-%d")
    result = coord.run_screening(date, verbose=True)

    if "error" in result:
        print(f"\n❌ 选股失败: {result['error']}")
        return

    print(f"\n✅ 选股完成")
    print(f"   日期: {result['date']}")
    print(f"   宏观状态: {result['macro'].get('regime', '?')}")
    print(f"   建议仓位: {result['macro'].get('position_ratio', 0):.0%}")
    print(f"   精选股票: {len(result.get('top_50', []))} 只")
    print(f"   耗时: {result.get('elapsed_seconds', 0):.1f} 秒")


def cmd_backtest():
    """逐月回测"""
    from coordinator import Coordinator
    from config import BACKTEST_START, BACKTEST_END

    print("\n" + "█" * 60)
    print(f"  多因子选股系统 v2 — 逐月回测")
    print(f"  回测区间: {BACKTEST_START} → {BACKTEST_END}")
    print("█" * 60)

    coord = Coordinator(use_cache=True)
    result = coord.run_backtest(verbose=True)

    summary = result.get("summary", {})
    print("\n📊 回测绩效:")
    print(f"   累计收益: {summary.get('cumulative_return', 0):.2%}")
    print(f"   年化收益: {summary.get('annual_return', 0):.2%}")
    print(f"   年化波动: {summary.get('volatility', 0):.2%}")
    print(f"   夏普比率: {summary.get('sharpe_ratio', 0):.2f}")
    print(f"   最大回撤: {summary.get('max_drawdown', 0):.2%}")
    print(f"   胜率: {summary.get('win_rate', 0):.1%}")
    print(f"   调仓次数: {summary.get('n_periods', 0)}")
    print(f"   错误次数: {summary.get('error_count', 0)}")


def cmd_check():
    """快速检查模块依赖和导入"""
    import importlib

    modules = [
        "config",
        "utils",
        "agent_data",
        "agent_factor",
        "agent_risk",
        "agent_macro",
        "agent_portfolio",
        "coordinator",
    ]

    print("\n模块检查:")
    all_ok = True
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
            print(f"  ✅ {mod_name}")
        except Exception as e:
            print(f"  ❌ {mod_name}: {e}")
            all_ok = False

    # 检查第三方依赖
    third_party = ["numpy", "pandas", "akshare"]
    print("\n第三方依赖:")
    for pkg in third_party:
        try:
            importlib.import_module(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ⚠️  {pkg} 未安装")
            all_ok = False

    if all_ok:
        print("\n✅ 所有模块正常")
    else:
        print("\n⚠️  部分模块有问题,请检查依赖")


def print_usage():
    print("""
用法:
  python main.py screen        # 当前选股
  python main.py backtest      # 逐月回测
  python main.py check         # 模块检查
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "screen":
        cmd_screen()
    elif cmd == "backtest":
        cmd_backtest()
    elif cmd == "check":
        cmd_check()
    else:
        print(f"未知命令: {cmd}")
        print_usage()
