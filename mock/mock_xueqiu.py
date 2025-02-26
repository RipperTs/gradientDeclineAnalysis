import pandas as pd
import numpy as np
import datetime
import random


def generate_mock_xueqiu_data(stock_code, start_date, end_date, include_decline=True):
    """
    生成模拟的雪球股票数据

    参数:
    - stock_code: 股票代码
    - start_date: 开始日期 (YYYY-MM-DD)
    - end_date: 结束日期 (YYYY-MM-DD)
    - include_decline: 是否包含一个明显的梯度下跌模式

    返回:
    - 包含模拟股票数据的DataFrame
    """
    # 将日期字符串转换为datetime对象
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    # 创建交易日期列表（排除周末）
    date_list = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # 0-4表示周一至周五
            date_list.append(current)
        current += datetime.timedelta(days=1)

    # 生成数据字典
    data = {
        'date': date_list,
        'symbol': [stock_code] * len(date_list),
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': [],
        'amount': [],  # 成交额
        'turnover_rate': [],  # 换手率
        'pe_ttm': [],  # 市盈率
        'pb': []  # 市净率
    }

    # 设置初始价格
    base_price = 100.0
    current_price = base_price

    # 生成价格数据
    for i in range(len(date_list)):
        # 创建随机波动
        if include_decline and i >= len(date_list) - 10 and i < len(date_list) - 5:
            # 创建一个明显的5天下跌模式
            change_pct = np.random.uniform(-0.03, -0.005)  # -0.5%到-3%之间的下跌
        else:
            change_pct = np.random.normal(0.0005, 0.015)  # 均值略为正，波动较大

        # 计算今日开盘价（基于昨日收盘价加一点随机波动）
        if i == 0:
            open_price = base_price
        else:
            open_price = data['close'][i - 1] * (1 + np.random.normal(0, 0.005))

        # 计算收盘价
        close_price = open_price * (1 + change_pct)

        # 确保价格不会低于1
        close_price = max(1.0, close_price)

        # 设置最高价和最低价
        daily_volatility = abs(close_price - open_price) * np.random.uniform(1.0, 2.0)
        if close_price > open_price:
            high_price = close_price + daily_volatility * np.random.uniform(0.1, 0.5)
            low_price = open_price - daily_volatility * np.random.uniform(0.1, 0.4)
        else:
            high_price = open_price + daily_volatility * np.random.uniform(0.1, 0.4)
            low_price = close_price - daily_volatility * np.random.uniform(0.1, 0.5)

        low_price = max(0.1, low_price)  # 确保最低价格不会太低

        # 生成成交量
        if include_decline and i >= len(date_list) - 10 and i < len(date_list) - 5:
            # 下跌期间成交量增加
            volume = np.random.uniform(5000000, 15000000)
        else:
            volume = np.random.uniform(3000000, 10000000)

        # 计算成交额
        amount = volume * ((open_price + close_price) / 2) * 100  # 假设每手100股

        # 计算换手率 (成交量占总流通股的百分比)
        total_float_shares = 100000000  # 假设流通股为1亿股
        turnover = (volume / total_float_shares) * 100

        # 添加数据
        data['open'].append(round(open_price, 2))
        data['high'].append(round(high_price, 2))
        data['low'].append(round(low_price, 2))
        data['close'].append(round(close_price, 2))
        data['volume'].append(int(volume))
        data['amount'].append(round(amount, 2))
        data['turnover_rate'].append(round(turnover, 2))
        data['pe_ttm'].append(round(np.random.uniform(15, 25), 2))
        data['pb'].append(round(np.random.uniform(1.5, 3.0), 2))

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 设置日期为索引
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 添加移动平均线
    df['ma5'] = df['close'].rolling(window=5).mean().round(2)
    df['ma10'] = df['close'].rolling(window=10).mean().round(2)
    df['ma20'] = df['close'].rolling(window=20).mean().round(2)
    df['ma30'] = df['close'].rolling(window=30).mean().round(2)
    df['ma60'] = df['close'].rolling(window=60).mean().round(2)

    # 添加MACD指标
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2

    # 添加KDJ指标
    low_9 = df['low'].rolling(window=9).min()
    high_9 = df['high'].rolling(window=9).max()
    df['k'] = 50
    df['d'] = 50
    df['j'] = 50

    for i in range(9, len(df)):
        if high_9[i] - low_9[i] != 0:
            rsv = (df['close'].iloc[i] - low_9[i]) / (high_9[i] - low_9[i]) * 100
        else:
            rsv = 50
        df['k'].iloc[i] = 2 / 3 * df['k'].iloc[i - 1] + 1 / 3 * rsv
        df['d'].iloc[i] = 2 / 3 * df['d'].iloc[i - 1] + 1 / 3 * df['k'].iloc[i]
        df['j'].iloc[i] = 3 * df['k'].iloc[i] - 2 * df['d'].iloc[i]

    return df


if __name__ == '__main__':

    # 生成模拟数据示例
    stock_code = "SH000001"  # 上证指数
    start_date = "2024-01-01"
    end_date = "2025-02-25"

    mock_data = generate_mock_xueqiu_data(stock_code, start_date, end_date, include_decline=True)

    # 显示数据的前几行
    print("模拟的雪球股票数据结构：")
    print(mock_data.head())

    # 输出数据中的所有列
    print("\n数据包含的字段：")
    print(mock_data.columns.tolist())

    # 保存到CSV文件以便分析
    csv_filename = f"{stock_code}_mock_data.csv"
    mock_data.to_csv(csv_filename)
    print(f"\n模拟数据已保存至 {csv_filename}")

    # 打印数据形状
    print(f"\n数据形状：{mock_data.shape}")