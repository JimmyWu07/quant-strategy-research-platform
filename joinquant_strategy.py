"""
聚宽版双均线+ADX趋势策略

核心逻辑：
- 每天14:50执行,用历史K线(不含当日)计算信号,避免未来函数
- 金叉/死叉触发时以当天实时价格成交
- 基准收益:沪深300

本地代码来源:
- strategy.py (DoubleMAStrategy)
- config.py (STRATEGY_PARAMS)
"""

import numpy as np
import pandas as pd


def initialize(context):
    g.stock = '300502.XSHE'         # 新易盛
    g.short_window = 5
    g.long_window = 10
    g.adx_period = 14
    g.adx_threshold = 30
    g.in_position = False

    set_benchmark('000300.XSHG')    # 基准收益 = 沪深300
    run_daily(daily_check, time='14:50')


def daily_check(context):
    stock = g.stock

    # ============ 1. 拉历史K线（不含当日，无未来函数）============
    lookback = max(g.long_window, g.adx_period * 3) + 30
    df = attribute_history(stock, lookback, '1d',
                           fields=['open', 'high', 'low', 'close'],
                           skip_paused=True, df=True)

    if df is None or len(df) < 70:
        return

    close = df['close']
    high = df['high']
    low = df['low']

    # ============ 2. 双均线 ============
    ma_short = close.rolling(g.short_window).mean()
    ma_long = close.rolling(g.long_window).mean()

    # ============ 3. ADX ============
    period = g.adx_period
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    s_plus = plus_dm.rolling(period).mean()
    s_minus = minus_dm.rolling(period).mean()
    plus_di = 100 * s_plus / atr.replace(0, np.nan)
    minus_di = 100 * s_minus / atr.replace(0, np.nan)
    di_sum = plus_di + minus_di
    dx = 100 * abs(plus_di - minus_di) / di_sum.replace(0, np.nan)
    adx = dx.rolling(period).mean()

    # ============ 4. 取昨日值和前日值（attribute_history不含当日，iloc[-1]=昨天）============
    ma_s_yest = ma_short.iloc[-1]     # 昨日MA短
    ma_s_prev = ma_short.iloc[-2]     # 前日MA短
    ma_l_yest = ma_long.iloc[-1]      # 昨日MA长
    ma_l_prev = ma_long.iloc[-2]      # 前日MA长
    adx_yest = adx.iloc[-1]           # 昨日ADX

    if pd.isna(ma_s_yest) or pd.isna(ma_l_yest) or pd.isna(adx_yest):
        return

    # ============ 5. 信号判断（用昨日确认数据判断交叉，无未来函数）============
    has_trend = adx_yest > g.adx_threshold

    # 金叉：昨日 MA短>MA长 且 前日 MA短<=MA长，即昨日发生上穿
    golden = (
        ma_s_yest > ma_l_yest and
        ma_s_prev <= ma_l_prev and
        has_trend
    )
    # 死叉：昨日发生下穿
    death = (
        ma_s_yest < ma_l_yest and
        ma_s_prev >= ma_l_prev and
        has_trend
    )

    # ============ 6. 下单 ============
    cash = context.portfolio.available_cash

    log.info('%s | MA短:%.2f/%.2f MA长:%.2f/%.2f ADX:%.2f 趋势:%s 持仓:%s 金叉:%s 死叉:%s' %
             (context.current_dt.strftime('%Y-%m-%d'),
              ma_s_prev, ma_s_yest, ma_l_prev, ma_l_yest, adx_yest,
              str(has_trend), str(g.in_position),
              str(golden), str(death)))

    if golden and not g.in_position:
        order_value(stock, cash)
        g.in_position = True
        log.info('>>> 买入.ADX=%.2f' % adx_yest)

    elif death and g.in_position:
        order_target(stock, 0)
        g.in_position = False
        log.info('>>> 卖出.ADX=%.2f' % adx_yest)
