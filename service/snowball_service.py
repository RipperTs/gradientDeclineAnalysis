from config import SCKEY, U_UID
import pysnowball as ball


class SnowballService:
    """
    雪球服务类
    """

    def __init__(self):
        self.token = SCKEY
        self.uid = U_UID
        ball.set_token(SCKEY)

    def real_time_quotes(self, symbol):
        """
        获取实时行情
        :param symbol: 股票代码
        :return: 返回实时行情
        """
        result = ball.quotec(symbol)
        return result

    def market_data_details(self, symbol):
        """
        获取股票详情
        :param symbol: 股票代码
        :return: 返回股票详情
        """
        result = ball.quote_detail(symbol)
        return result

    def pankou(self, symbol):
        """
        获取实时分笔数据，可以实时取得股票当前报价和成交信息
        :param symbol: 股票代码
        :return: 返回盘口数据
        """
        result = ball.pankou(symbol)
        return result

    def kline(self, symbol, days=100):
        """
        获取K线数据
        :param symbol: 股票代码
        :param period: K线周期, 可制定从现在到N天前，默认100.
        :return: 返回K线数据
        """
        result = ball.kline(symbol, days)
        return result

    def performance_report(self, symbol):
        """
        获取业绩报告
        :param symbol: 股票代码
        :return: 返回业绩报告
        """
        result = ball.earningforecast(symbol)
        return result

    def agency_ratings(self, symbol):
        """
        获取机构评级
        :param symbol: 股票代码
        :return: 返回机构评级
        """
        result = ball.report(symbol)
        return result


    def capital_flow(self, symbol):
        """
        获取当日资金流如流出数据，每分钟数据
        :param symbol: 股票代码
        :return: 返回资金流向
        """
        result = ball.capital_flow(symbol)
        return result


    def capital_history(self, symbol):
        """
        获取历史资金流如流出数据，每日数据
        :param symbol: 股票代码, SZ002027
        :return: 返回资金流向历史数据
        """
        result = ball.capital_history(symbol)
        return result

    def capital_assort(self, symbol):
        """
        获取资金成交分布数据
        :param symbol: 股票代码
        :return: 返回资金分布
        """
        result = ball.capital_assort(symbol)
        return result

    def blocktrans(self, symbol):
        """
        大宗交易数据
        :param symbol: 股票代码
        :return:
        """
        result = ball.blocktrans(symbol)
        return result


    def margin(self, symbol):
        """
        融资融券数据
        :param symbol: 股票代码
        :return:
        """
        result = ball.margin(symbol)
        return result

    def indicator(self, symbol,  is_annals=0, count=10):
        """
        按年度、季度获取业绩报表数据。
        :param symbol: 股票代码
        :param is_annals: 只获取年报,默认为1
        :param count: 获取的条数
        :return:
        """
        result = ball.indicator(symbol, is_annals, count)
        return result

    def suggest_stock(self, keyword):
        """
        根据关键词搜索股票
        :param keyword: 关键词
        :return:
        """
        result = ball.suggest_stock(keyword)
        return result


if __name__ == '__main__':
    snowball = SnowballService()
    print(snowball.suggest_stock('完美世界'))