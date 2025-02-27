import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端为非交互式
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import platform
from matplotlib.font_manager import FontManager

def setup_chinese_font():
    """
    设置中文字体，优先使用系统已安装的中文字体
    """
    fm = FontManager()
    font_names = [f.name for f in fm.ttflist]
    
    # 按优先级排序的中文字体列表
    chinese_fonts = [
        'WenQuanYi Micro Hei',  # Linux优先
        'Noto Sans CJK SC',     # Linux备选
        'Microsoft YaHei',      # Windows优先
        'SimHei',              # Windows备选
        'Arial Unicode MS',    # MacOS优先
        'PingFang SC',         # MacOS备选
        'Heiti SC'             # MacOS备选
    ]
    
    # 查找第一个可用的中文字体
    for font in chinese_fonts:
        if font in font_names:
            plt.rcParams['font.sans-serif'] = [font]
            break
    else:
        # 如果没有找到任何中文字体，使用默认字体并打印警告
        print("警告：未找到合适的中文字体，可能会导致中文显示为方块")
        print("建议安装以下字体之一：")
        print("Linux: apt-get install fonts-wqy-microhei")
        print("MacOS: 系统自带中文字体")
        print("Windows: 系统自带中文字体")
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.family'] = 'sans-serif'

# 在程序开始时设置字体
setup_chinese_font()

from mock.mock_xueqiu import generate_mock_xueqiu_data
from service.snowball_service import SnowballService

snowball = SnowballService()


def detect_gradient_decline(df, window=5, threshold=0.005):
    """
    检测股票数据中的连续梯度下跌模式。

    参数:
    - df: 包含股票数据的DataFrame（必须有'close'列）
    - window: 要寻找的连续下跌次数（不一定是连续天数）
    - threshold: 符合条件的最小下跌百分比

    返回:
    - 包含检测到的模式的DataFrame
    """
    # 确保使用的是副本
    data = df.copy()

    # 计算每日价格变化
    data['price_change'] = data['close'].pct_change()

    # 为价格下跌的日子创建信号
    data['decline'] = (data['price_change'] < -threshold).astype(int)

    # 初始化结果DataFrame
    patterns = pd.DataFrame()
    
    # 查找连续下跌模式（不要求严格连续天数）
    decline_count = 0
    decline_start_idx = None
    
    for i in range(1, len(data)):
        current_idx = data.index[i]
        
        # 如果当天下跌
        if data.loc[current_idx, 'decline'] == 1:
            # 如果是新的下跌序列开始
            if decline_count == 0:
                decline_start_idx = data.index[i-1]  # 记录下跌开始前一天的索引
            
            # 增加下跌计数
            decline_count += 1
            
            # 如果达到了指定的连续下跌次数
            if decline_count == window:
                # 计算总跌幅
                start_price = data.loc[decline_start_idx, 'close']
                end_price = data.loc[current_idx, 'close']
                total_decline_pct = (end_price - start_price) / start_price * 100
                
                # 创建新行
                new_row = pd.DataFrame({
                    'close': [end_price],
                    'start_price': [start_price],
                    'start_date': [decline_start_idx],
                    'price_change': [data.loc[current_idx, 'price_change']],
                    'total_decline_pct': [total_decline_pct],
                    'decline_days': [(current_idx - decline_start_idx).days]
                }, index=[current_idx])
                
                # 添加到结果中
                patterns = pd.concat([patterns, new_row])
                
                # 重置计数器，允许检测重叠的模式
                decline_count = 0
        
        # 如果当天不是下跌但下跌次数不足window，继续保持当前下跌序列
        # 这里不重置decline_count，允许中间有非下跌日
    
    return patterns


def detect_trend_decline(df, window=10, threshold=0.05, max_up_days=3, min_down_days=5):
    """
    检测股票数据中的趋势性下跌模式，允许中间有少量上涨日。
    
    参数:
    - df: 包含股票数据的DataFrame（必须有'close'列）
    - window: 要检查的窗口天数
    - threshold: 整体下跌的最小百分比阈值
    - max_up_days: 窗口内允许的最大上涨天数
    - min_down_days: 窗口内要求的最小下跌天数
    
    返回:
    - 包含检测到的模式的DataFrame
    """
    # 确保使用的是副本
    data = df.copy()
    
    # 计算每日价格变化
    data['price_change'] = data['close'].pct_change()
    
    # 标记上涨和下跌日
    data['up_day'] = (data['price_change'] > 0).astype(int)
    data['down_day'] = (data['price_change'] < 0).astype(int)
    
    # 初始化结果DataFrame
    trend_patterns = pd.DataFrame()
    
    # 对每个可能的窗口结束位置进行检查
    for i in range(window, len(data)):
        end_idx = data.index[i]
        start_idx = data.index[i - window]
        
        # 获取窗口数据
        window_data = data.loc[start_idx:end_idx].copy()
        
        # 计算窗口内的上涨和下跌天数
        up_days = window_data['up_day'].sum()
        down_days = window_data['down_day'].sum()
        
        # 计算窗口内的总价格变化百分比
        start_price = window_data['close'].iloc[0]
        end_price = window_data['close'].iloc[-1]
        total_change_pct = (end_price - start_price) / start_price * 100
        
        # 检查是否符合趋势下跌条件:
        # 1. 总跌幅超过阈值
        # 2. 上涨天数不超过允许的最大值
        # 3. 下跌天数不少于要求的最小值
        if (total_change_pct < -threshold and 
            up_days <= max_up_days and 
            down_days >= min_down_days):
            
            # 创建新行
            new_row = pd.DataFrame({
                'close': [end_price],
                'start_price': [start_price],
                'total_decline_pct': [total_change_pct],
                'up_days': [up_days],
                'down_days': [down_days],
                'window_size': [window]
            }, index=[end_idx])
            
            # 添加到结果中
            trend_patterns = pd.concat([trend_patterns, new_row])
    
    return trend_patterns


def fetch_data_from_xueqiu(stock_code, start_date, end_date):
    """
    从雪球获取股票数据

    参数:
    - stock_code: 股票代码，例如 'SH000001'
    - start_date: 开始日期，格式 'YYYY-MM-DD'
    - end_date: 结束日期，格式 'YYYY-MM-DD'

    返回:
    - 包含股票数据的DataFrame
    """
    # 计算需要获取多少天的K线数据
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days + 1

    # 获取K线数据
    # 注意：雪球的K线API返回的是从当前往前推N天的数据
    kline_data = snowball.kline(stock_code, days)

    # 将数据转换为DataFrame
    df = pd.DataFrame(kline_data['data']['item'])

    # 列名映射，根据返回的数据结构调整
    # 实际字段可能与示例代码不同，需要根据实际返回调整
    column_names = {
        0: 'timestamp',  # 时间戳
        1: 'volume',  # 成交量
        2: 'open',  # 开盘价
        3: 'high',  # 最高价
        4: 'low',  # 最低价
        5: 'close',  # 收盘价
        6: 'chg',  # 涨跌额
        7: 'percent',  # 涨跌幅
        8: 'turnoverrate',  # 换手率
        9: 'amount'  # 成交额
    }

    # 重命名列
    df.rename(columns=column_names, inplace=True)

    # 转换时间戳到日期
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)

    # 按日期排序，从早到晚
    df.sort_index(inplace=True)

    # 计算移动平均线
    df['ma5'] = df['close'].rolling(window=5).mean().round(2)
    df['ma10'] = df['close'].rolling(window=10).mean().round(2)
    df['ma20'] = df['close'].rolling(window=20).mean().round(2)
    df['ma30'] = df['close'].rolling(window=30).mean().round(2)
    df['ma60'] = df['close'].rolling(window=60).mean().round(2)

    # 只保留在指定日期范围内的数据
    df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]

    return df


def visualize_pattern(data, pattern_date, window=5):
    """
    可视化检测到的下跌模式。

    参数:
    - data: 完整的股票数据DataFrame
    - pattern_date: 模式完成的日期
    - window: 模式窗口大小
    """
    # 获取模式周围的数据
    pattern_idx = data.index.get_loc(pattern_date)
    
    # 获取起始日期（如果在patterns中有记录）
    if 'start_date' in data.loc[pattern_date]:
        start_date = data.loc[pattern_date, 'start_date']
        start_idx = data.index.get_loc(start_date)
    else:
        # 如果没有记录起始日期，使用默认的窗口大小
        start_idx = max(0, pattern_idx - window * 2)
    
    # 设置显示范围
    display_start_idx = max(0, start_idx - 5)  # 显示起始日期前5个交易日
    end_idx = min(len(data) - 1, pattern_idx + 5)  # 显示结束日期后5个交易日

    plot_data = data.iloc[display_start_idx:end_idx + 1].copy()

    # 获取模式部分
    decline_data = data.iloc[start_idx:pattern_idx + 1].copy()

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))

    # 创建一个数字索引用于绘图
    x_values = np.arange(len(plot_data))
    x_ticks = plot_data.index

    # 绘制OHLC K线
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        color = 'red' if row['close'] < row['open'] else 'green'
        # 使用数字索引i而不是日期索引
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        ax.plot([i - 0.3, i + 0.3], [row['open'], row['open']], color=color, linewidth=2)
        ax.plot([i - 0.3, i + 0.3], [row['close'], row['close']], color=color, linewidth=2)

    # 绘制移动平均线
    if 'ma5' in plot_data.columns:
        ax.plot(x_values, plot_data['ma5'], 'white', linewidth=1, label='MA5')
    if 'ma10' in plot_data.columns:
        ax.plot(x_values, plot_data['ma10'], 'yellow', linewidth=1, label='MA10')
    if 'ma20' in plot_data.columns:
        ax.plot(x_values, plot_data['ma20'], 'magenta', linewidth=1, label='MA20')
    if 'ma30' in plot_data.columns:
        ax.plot(x_values, plot_data['ma30'], 'green', linewidth=1, label='MA30')
    if 'ma60' in plot_data.columns:
        ax.plot(x_values, plot_data['ma60'], 'blue', linewidth=1, label='MA60')

    # 突出显示下跌模式
    # 找出decline_data在plot_data中的索引位置
    decline_indices = [x_values[plot_data.index.get_loc(idx)] for idx in decline_data.index if idx in plot_data.index]
    decline_prices = [decline_data.loc[idx, 'close'] for idx in decline_data.index if idx in plot_data.index]
    
    # 绘制趋势线
    ax.plot(decline_indices, decline_prices, 'r-', linewidth=2, label=f'梯度下跌')
    
    # 标记起点和终点
    if decline_indices:
        ax.scatter(decline_indices[0], decline_prices[0], color='yellow', s=100, zorder=5)
        ax.scatter(decline_indices[-1], decline_prices[-1], color='red', s=100, zorder=5)
        
        # 添加总跌幅标注
        total_decline = (decline_prices[-1] - decline_prices[0]) / decline_prices[0] * 100
        days_count = len(decline_indices)
        ax.annotate(f"总跌幅: {total_decline:.2f}%\n下跌次数: {window}\n持续天数: {days_count}",
                   (decline_indices[-1], decline_prices[-1]),
                   textcoords="offset points",
                   xytext=(10, -20),
                   ha='left',
                   fontsize=10,
                   color='red',
                   bbox=dict(boxstyle="round,pad=0.3", fc='#1E1E1E', ec="red", alpha=0.8))

    # 添加下跌日的标注
    for j, idx in enumerate(decline_data.index):
        if idx in plot_data.index and j > 0:
            plot_idx = x_values[plot_data.index.get_loc(idx)]
            prev_close = decline_data['close'].iloc[j - 1]
            curr_close = decline_data['close'].iloc[j]
            pct_change = (curr_close - prev_close) / prev_close * 100
            if pct_change < 0:  # 只标注下跌日
                ax.annotate(f"{pct_change:.2f}%",
                           (plot_idx, curr_close),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center',
                           fontsize=8,
                           color='red')

    # 设置x轴刻度为日期
    step = max(1, len(plot_data) // 10)  # 最多显示10个刻度
    tick_positions = x_values[::step]
    tick_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in plot_data.index[::step]]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', color='white')
    ax.tick_params(axis='both', colors='white')  # 设置刻度颜色为白色

    ax.set_title(f'梯度下跌模式 {pattern_date.strftime("%Y-%m-%d")}', color='white')
    ax.set_xlabel('日期', color='white')
    ax.set_ylabel('价格', color='white')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', facecolor='#1E1E1E', labelcolor='white')
    ax.set_facecolor('#1E1E1E')  # 深色背景
    fig.set_facecolor('#1E1E1E')

    # 设置所有文本为白色
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_color('white')

    # 调整布局以避免标签被裁剪
    plt.tight_layout()

    return fig


def advanced_decline_analysis(df, window=5, threshold=0.5, trend_params=None):
    """
    执行多种类型的下跌模式检测。

    参数:
    - df: 包含股票数据的DataFrame
    - window: 模式窗口大小
    - threshold: 下跌阈值百分比
    - trend_params: 趋势下跌的参数字典，包含以下键:
        - trend_window: 趋势窗口大小
        - trend_threshold: 趋势下跌阈值
        - max_up_days: 最大允许上涨天数
        - min_down_days: 最小要求下跌天数

    返回:
    - 包含不同模式类型的字典
    """
    # 设置默认的趋势参数
    if trend_params is None:
        trend_params = {
            'trend_window': window * 2,
            'trend_threshold': threshold * 5,  # 默认为基本阈值的5倍
            'max_up_days': (window * 2) // 3,
            'min_down_days': (window * 2) // 2
        }
    
    results = {}

    # 1. 基本连续价格下跌
    results['consecutive_declines'] = detect_gradient_decline(df, window, threshold)
    
    # 2. 趋势性下跌（新增）
    results['trend_declines'] = detect_trend_decline(
        df, 
        window=trend_params['trend_window'], 
        threshold=trend_params['trend_threshold'],
        max_up_days=trend_params['max_up_days'],
        min_down_days=trend_params['min_down_days']
    )

    # 3. 成交量增加的下跌
    if 'volume' in df.columns:
        # 计算成交量增加
        data = df.copy()
        data['volume_change'] = data['volume'].pct_change()

        # 复制检测逻辑但添加成交量条件
        data['price_change'] = data['close'].pct_change()
        data['decline'] = ((data['price_change'] < -0.005) & (data['volume_change'] > 0)).astype(int)
        data['consecutive_vol_declines'] = data['decline'].rolling(window=window).sum()
        results['volume_confirmed_declines'] = data[data['consecutive_vol_declines'] == window].copy()

    # 4. 移动平均线交叉下跌（价格穿越MA线向下）
    if 'ma20' not in df.columns and len(df) >= 20:
        df['ma20'] = df['close'].rolling(window=20).mean()

    if 'ma20' in df.columns:
        data = df.copy()
        data['below_ma'] = (data['close'] < data['ma20']).astype(int)
        data['ma_crossdown'] = (data['below_ma'].diff() == 1).astype(int)
        data['ma_down_trend'] = (data['ma20'].diff() < 0).astype(int)
        # 识别价格穿越MA20向下且MA20呈下降趋势的情况
        results['ma_crossover_declines'] = data[(data['ma_crossdown'] == 1) & (data['ma_down_trend'] == 1)].copy()

    return results


def visualize_advanced_pattern(data, pattern_date, pattern_type, window=5):
    """
    可视化高级分析检测到的模式。

    参数:
    - data: 完整的股票数据DataFrame
    - pattern_date: 模式完成的日期
    - pattern_type: 模式类型
    - window: 模式窗口大小
    """
    # 获取模式周围的数据
    pattern_idx = data.index.get_loc(pattern_date)
    start_idx = max(0, pattern_idx - window * 2)
    end_idx = min(len(data) - 1, pattern_idx + window)

    plot_data = data.iloc[start_idx:end_idx + 1].copy()

    # 获取模式部分
    decline_start_idx = max(0, pattern_idx - window + 1)
    decline_data = data.iloc[decline_start_idx:pattern_idx + 1].copy()

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    fig.suptitle(f'{pattern_type} {pattern_date.strftime("%Y-%m-%d")}', color='white')

    # 创建一个数字索引用于绘图
    x_values = np.arange(len(plot_data))
    x_ticks = plot_data.index

    # 绘制OHLC K线
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        color = 'red' if row['close'] < row['open'] else 'green'
        ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        ax1.plot([i - 0.3, i + 0.3], [row['open'], row['open']], color=color, linewidth=2)
        ax1.plot([i - 0.3, i + 0.3], [row['close'], row['close']], color=color, linewidth=2)

    # 绘制移动平均线
    if 'ma5' in plot_data.columns:
        ax1.plot(x_values, plot_data['ma5'], 'white', linewidth=1, label='MA5')
    if 'ma10' in plot_data.columns:
        ax1.plot(x_values, plot_data['ma10'], 'yellow', linewidth=1, label='MA10')
    if 'ma20' in plot_data.columns:
        ax1.plot(x_values, plot_data['ma20'], 'magenta', linewidth=1, label='MA20')
    if 'ma30' in plot_data.columns:
        ax1.plot(x_values, plot_data['ma30'], 'green', linewidth=1, label='MA30')
    if 'ma60' in plot_data.columns:
        ax1.plot(x_values, plot_data['ma60'], 'blue', linewidth=1, label='MA60')

    # 突出显示下跌模式
    decline_indices = [x_values[plot_data.index.get_loc(idx)] for idx in decline_data.index]
    ax1.plot(decline_indices, decline_data['close'], 'r--', linewidth=2, label=f'{window}天下跌')

    # 添加注释
    for j, idx in enumerate(decline_data.index):
        plot_idx = x_values[plot_data.index.get_loc(idx)]
        if j > 0:
            prev_close = decline_data['close'].iloc[j - 1]
            curr_close = decline_data['close'].iloc[j]
            pct_change = (curr_close - prev_close) / prev_close * 100
            ax1.annotate(f"{pct_change:.2f}%",
                       (plot_idx, curr_close),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center',
                       fontsize=8,
                       color='red')

    # 在第二个子图中绘制成交量或其他指标
    if pattern_type == 'volume_confirmed_declines':
        ax2.bar(x_values, plot_data['volume'], color=['red' if c < o else 'green' for c, o in zip(plot_data['close'], plot_data['open'])])
        ax2.set_ylabel('成交量')
    elif pattern_type == 'ma_crossover_declines':
        ax2.plot(x_values, plot_data['ma20'], 'magenta', label='MA20')
        ax2.plot(x_values, plot_data['close'], 'white', label='价格')
        ax2.set_ylabel('价格与MA20')
        ax2.legend()

    # 设置x轴刻度
    step = max(1, len(plot_data) // 10)
    tick_positions = x_values[::step]
    tick_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in plot_data.index[::step]]

    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([], color='white')
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', color='white')

    # 设置样式
    ax1.set_ylabel('价格', color='white')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', facecolor='#1E1E1E', labelcolor='white')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', facecolor='#1E1E1E', labelcolor='white')
    
    # 设置背景色
    ax1.set_facecolor('#1E1E1E')
    ax2.set_facecolor('#1E1E1E')
    fig.set_facecolor('#1E1E1E')

    # 设置标题颜色
    fig.suptitle(f'{pattern_type} {pattern_date.strftime("%Y-%m-%d")}', color='white')

    # 设置所有文本为白色
    for text in ax1.get_xticklabels() + ax1.get_yticklabels():
        text.set_color('white')
    for text in ax2.get_xticklabels() + ax2.get_yticklabels():
        text.set_color('white')

    # 设置y轴标签颜色
    ax2.set_ylabel('成交量' if pattern_type == 'volume_confirmed_declines' else '价格与MA20', color='white')

    # 调整布局
    plt.tight_layout()

    return fig


def visualize_trend_pattern(data, pattern_date, window=10):
    """
    可视化趋势下跌模式。

    参数:
    - data: 完整的股票数据DataFrame
    - pattern_date: 模式完成的日期
    - window: 模式窗口大小
    """
    # 获取模式周围的数据
    pattern_idx = data.index.get_loc(pattern_date)
    start_idx = max(0, pattern_idx - window)
    end_idx = min(len(data) - 1, pattern_idx + window // 2)

    plot_data = data.iloc[start_idx:end_idx + 1].copy()

    # 获取模式部分
    trend_data = data.iloc[pattern_idx - window + 1:pattern_idx + 1].copy()

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    fig.suptitle(f'趋势下跌模式 {pattern_date.strftime("%Y-%m-%d")}', color='white')

    # 创建一个数字索引用于绘图
    x_values = np.arange(len(plot_data))
    
    # 绘制OHLC K线
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        color = 'red' if row['close'] < row['open'] else 'green'
        ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        ax1.plot([i - 0.3, i + 0.3], [row['open'], row['open']], color=color, linewidth=2)
        ax1.plot([i - 0.3, i + 0.3], [row['close'], row['close']], color=color, linewidth=2)

    # 绘制移动平均线
    if 'ma5' in plot_data.columns:
        ax1.plot(x_values, plot_data['ma5'], 'white', linewidth=1, label='MA5')
    if 'ma10' in plot_data.columns:
        ax1.plot(x_values, plot_data['ma10'], 'yellow', linewidth=1, label='MA10')
    if 'ma20' in plot_data.columns:
        ax1.plot(x_values, plot_data['ma20'], 'magenta', linewidth=1, label='MA20')
    if 'ma30' in plot_data.columns:
        ax1.plot(x_values, plot_data['ma30'], 'green', linewidth=1, label='MA30')
    if 'ma60' in plot_data.columns:
        ax1.plot(x_values, plot_data['ma60'], 'blue', linewidth=1, label='MA60')

    # 突出显示趋势下跌
    trend_indices = [x_values[plot_data.index.get_loc(idx)] for idx in trend_data.index if idx in plot_data.index]
    trend_prices = [trend_data.loc[idx, 'close'] for idx in trend_data.index if idx in plot_data.index]
    
    # 绘制趋势线
    ax1.plot(trend_indices, trend_prices, 'r-', linewidth=2, label=f'趋势下跌')
    
    # 标记起点和终点
    if trend_indices:
        ax1.scatter(trend_indices[0], trend_prices[0], color='yellow', s=100, zorder=5)
        ax1.scatter(trend_indices[-1], trend_prices[-1], color='red', s=100, zorder=5)
        
        # 添加总跌幅标注
        total_decline = (trend_prices[-1] - trend_prices[0]) / trend_prices[0] * 100
        ax1.annotate(f"总跌幅: {total_decline:.2f}%",
                    (trend_indices[-1], trend_prices[-1]),
                    textcoords="offset points",
                    xytext=(10, -20),
                    ha='left',
                    fontsize=10,
                    color='red',
                    bbox=dict(boxstyle="round,pad=0.3", fc='#1E1E1E', ec="red", alpha=0.8))

    # 在第二个子图中绘制价格变化百分比
    price_changes = plot_data['close'].pct_change() * 100
    colors = ['red' if x < 0 else 'green' for x in price_changes]
    ax2.bar(x_values[1:], price_changes[1:], color=colors[1:])
    ax2.axhline(y=0, color='white', linestyle='-', alpha=0.3)
    ax2.set_ylabel('日涨跌幅(%)', color='white')

    # 设置x轴刻度
    step = max(1, len(plot_data) // 10)
    tick_positions = x_values[::step]
    tick_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in plot_data.index[::step]]

    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([], color='white')
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', color='white')

    # 设置样式
    ax1.set_ylabel('价格', color='white')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', facecolor='#1E1E1E', labelcolor='white')
    ax2.grid(True, alpha=0.3)
    
    # 设置背景色
    ax1.set_facecolor('#1E1E1E')
    ax2.set_facecolor('#1E1E1E')
    fig.set_facecolor('#1E1E1E')

    # 设置所有文本为白色
    for text in ax1.get_xticklabels() + ax1.get_yticklabels():
        text.set_color('white')
    for text in ax2.get_xticklabels() + ax2.get_yticklabels():
        text.set_color('white')

    # 调整布局
    plt.tight_layout()

    return fig


def analyze_market_declines(stock_code, start_date, end_date, window=5, threshold=0.005, trend_params=None):
    """
    分析股票数据中的梯度下跌的主函数。

    参数:
    - stock_code: 要分析的股票代码
    - start_date: 分析的开始日期 (YYYY-MM-DD)
    - end_date: 分析的结束日期 (YYYY-MM-DD)
    - window: 要寻找的连续下跌次数
    - threshold: 符合条件的最小下跌百分比
    - trend_params: 趋势下跌的参数字典，包含以下键:
        - trend_window: 趋势窗口大小
        - trend_threshold: 趋势下跌阈值
        - max_up_days: 最大允许上涨天数
        - min_down_days: 最小要求下跌天数
    """
    # 设置默认的趋势参数
    if trend_params is None:
        trend_params = {
            'trend_window': window * 2,
            'trend_threshold': 0.01,  # 1%
            'max_up_days': (window * 2) // 3,
            'min_down_days': (window * 2) // 2
        }
    
    # 创建分析结果目录
    analysis_time = datetime.now().strftime("%Y%m%d")
    static_dir = "static"
    result_dir = f"{static_dir}/analysis_results/{stock_code}_{analysis_time}"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"{result_dir}/charts", exist_ok=True)
    os.makedirs(f"{result_dir}/data", exist_ok=True)

    # 加载数据
    try:
        print(f"正在获取 {stock_code} 的数据...")
        data = fetch_data_from_xueqiu(stock_code, start_date, end_date)
        print(f"获取到 {len(data)} 条数据记录")

        # 确保数据按日期排序
        data.sort_index(inplace=True)

        # 检测模式
        print(f"正在检测梯度下跌模式...")
        patterns = detect_gradient_decline(data, window, threshold)

        # 打印结果
        if patterns.empty:
            print(f"未找到 {stock_code} 的梯度下跌模式")
        else:
            print(f"找到 {len(patterns)} 个 {stock_code} 的梯度下跌模式:")
            print(patterns[['close', 'price_change', 'total_decline_pct']])

            # 保存基本分析结果
            patterns.to_csv(f"{result_dir}/data/basic_patterns.csv")
            print(f"基本分析结果已保存至 {result_dir}/data/basic_patterns.csv")

            # 可视化基本模式
            for pattern_date in patterns.index:
                fig = visualize_pattern(data, pattern_date, window)
                fig.savefig(f"{result_dir}/charts/basic_pattern_{pattern_date.strftime('%Y%m%d')}.png")
                plt.close(fig)

        # 运行高级分析
        print("\n正在执行高级分析...")
        adv_results = advanced_decline_analysis(data, window, threshold, trend_params)

        print("\n高级分析结果:")
        for pattern_type, results in adv_results.items():
            if not results.empty:
                print(f"{pattern_type}: 发现 {len(results)} 个模式")
                
                # 保存高级分析结果
                results.to_csv(f"{result_dir}/data/{pattern_type}.csv")
                print(f"{pattern_type}分析结果已保存至 {result_dir}/data/{pattern_type}.csv")

                # 可视化高级分析模式
                for pattern_date in results.index:
                    if pattern_type == 'trend_declines':
                        fig = visualize_trend_pattern(data, pattern_date, int(results.loc[pattern_date, 'window_size']))
                    else:
                        fig = visualize_advanced_pattern(data, pattern_date, pattern_type, window)
                    fig.savefig(f"{result_dir}/charts/{pattern_type}_{pattern_date.strftime('%Y%m%d')}.png")
                    plt.close(fig)

        print(f"\n分析完成！所有结果已保存至目录: {result_dir}")
        return patterns, result_dir

    except Exception as e:
        print(f"分析过程中出错: {e}")
        return pd.DataFrame(), None


def main():
    print("=" * 50)
    print("股票市场梯度下跌分析系统")
    print("=" * 50)

    # 获取用户输入并处理默认值
    stock_code = input("请输入股票代码 (例如: SH688981): ").strip()
    if not stock_code:
        stock_code = "SH688981"

    start_date = input("请输入开始日期 (YYYY-MM-DD): ").strip()
    if not start_date:
        start_date = "2024-01-01"

    end_date = input("请输入结束日期 (YYYY-MM-DD): ").strip()
    if not end_date:
        end_date = "2025-02-25"

    window_input = input("请输入连续下跌次数 (默认5次): ").strip()
    window = int(window_input) if window_input else 5

    threshold_input = input("请输入单日下跌最小阈值 (默认0.5%): ").strip()
    threshold = float(threshold_input) / 100 if threshold_input else 0.005
    
    # 获取趋势下跌相关参数
    print("\n高级参数设置 (直接回车使用默认值):")
    
    trend_window_input = input(f"趋势窗口大小 (默认{window*2}天): ").strip()
    trend_window = int(trend_window_input) if trend_window_input else window * 2
    
    trend_threshold_input = input("趋势下跌阈值 (默认1.0%): ").strip()
    trend_threshold = float(trend_threshold_input) / 100 if trend_threshold_input else 0.01
    
    max_up_days_input = input(f"最大允许上涨天数 (默认{trend_window//3}天): ").strip()
    max_up_days = int(max_up_days_input) if max_up_days_input else trend_window // 3
    
    min_down_days_input = input(f"最小要求下跌天数 (默认{trend_window//2}天): ").strip()
    min_down_days = int(min_down_days_input) if min_down_days_input else trend_window // 2
    
    # 输入参数打印
    print(f"\n分析参数:")
    print(f"股票代码: {stock_code}")
    print(f"开始日期: {start_date}")
    print(f"结束日期: {end_date}")
    print(f"连续下跌次数: {window}")
    print(f"单日下跌最小阈值: {threshold * 100}%")
    print(f"\n高级参数:")
    print(f"趋势窗口大小: {trend_window}天")
    print(f"趋势下跌阈值: {trend_threshold * 100}%")
    print(f"最大允许上涨天数: {max_up_days}天")
    print(f"最小要求下跌天数: {min_down_days}天")
    

    print(f"\n开始分析 {stock_code} 的数据...")

    # 运行分析
    try:
        patterns, result_dir = analyze_market_declines(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            window=window,
            threshold=threshold,
            trend_params={
                'trend_window': trend_window,
                'trend_threshold': trend_threshold,
                'max_up_days': max_up_days,
                'min_down_days': min_down_days
            }
        )

        if result_dir:
            print(f"\n分析结果已保存至: {result_dir}")
            print("包含以下内容：")
            print(f"1. 数据文件: {result_dir}/data/")
            print(f"2. 图表文件: {result_dir}/charts/")

    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()