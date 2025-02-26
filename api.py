from flask import Flask, jsonify, request, send_from_directory
from datetime import datetime
import os
from xueqiu import analyze_market_declines, fetch_data_from_xueqiu
from config import *

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path=STATIC_URL_PATH)

@app.route('/')
def index():
    return send_from_directory(STATIC_FOLDER, 'index.html')

@app.route(f'/api/{API_VERSION}/analyze', methods=['GET'])
def analyze_stock():
    try:
        # 获取请求参数
        stock_code = request.args.get('symbol', DEFAULT_STOCK)
        start_date = request.args.get('start_date', DEFAULT_START_DATE)
        end_date = request.args.get('end_date', DEFAULT_END_DATE)
        window = int(request.args.get('window', str(DEFAULT_WINDOW)))
        threshold = float(request.args.get('threshold', str(DEFAULT_THRESHOLD))) / 100

        # 运行分析
        patterns, result_dir = analyze_market_declines(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            window=window,
            threshold=threshold
        )

        # 获取原始K线数据
        kline_data = fetch_data_from_xueqiu(stock_code, start_date, end_date)
        
        # 构建响应数据
        response = {
            'data': {
                'symbol': stock_code,
                'column': ['timestamp', 'volume', 'open', 'high', 'low', 'close', 'chg', 'percent', 
                          'turnoverrate', 'amount', 'ma5', 'ma10', 'ma20', 'ma30', 'ma60'],
                'item': []
            },
            'patterns': {
                'basic_patterns': [],
                'consecutive_declines': [],
                'volume_confirmed_declines': [],
                'ma_crossover_declines': [],
                'charts': {
                    'basic_patterns': [],
                    'consecutive_declines': [],
                    'volume_confirmed_declines': [],
                    'ma_crossover_declines': []
                }
            },
            'error_code': 0,
            'error_description': ''
        }

        # 添加K线数据
        for idx, row in kline_data.iterrows():
            item = [
                int(idx.timestamp() * 1000),  # timestamp
                int(row['volume']) if 'volume' in row else 0,
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['chg']) if 'chg' in row else 0,
                float(row['percent']) if 'percent' in row else 0,
                float(row['turnoverrate']) if 'turnoverrate' in row else 0,
                float(row['amount']) if 'amount' in row else 0,
                float(row['ma5']) if 'ma5' in row else None,
                float(row['ma10']) if 'ma10' in row else None,
                float(row['ma20']) if 'ma20' in row else None,
                float(row['ma30']) if 'ma30' in row else None,
                float(row['ma60']) if 'ma60' in row else None,
            ]
            response['data']['item'].append(item)

        # 添加模式数据
        if not patterns.empty:
            for idx, row in patterns.iterrows():
                pattern = {
                    'date': idx.strftime('%Y-%m-%d'),
                    'close': float(row['close']),
                    'price_change': float(row['price_change']),
                    'total_decline_pct': float(row['total_decline_pct'])
                }
                response['patterns']['basic_patterns'].append(pattern)

        # 添加图表路径
        if result_dir:
            base_url = request.host_url.rstrip('/')
            charts_dir = f"{result_dir}/{CHARTS_SUBDIR}"
            
            # 遍历charts目录获取所有图表
            for filename in os.listdir(charts_dir):
                file_path = f"{base_url}{STATIC_URL_PATH}/analysis_results/{stock_code}_{datetime.now().strftime('%Y%m%d')}/{CHARTS_SUBDIR}/{filename}"
                if filename.startswith('basic_pattern_'):
                    response['patterns']['charts']['basic_patterns'].append(file_path)
                elif filename.startswith('consecutive_declines_'):
                    response['patterns']['charts']['consecutive_declines'].append(file_path)
                elif filename.startswith('volume_confirmed_declines_'):
                    response['patterns']['charts']['volume_confirmed_declines'].append(file_path)
                elif filename.startswith('ma_crossover_declines_'):
                    response['patterns']['charts']['ma_crossover_declines'].append(file_path)

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error_code': 500,
            'error_description': str(e)
        }), 500

if __name__ == '__main__':
    # 确保静态目录存在
    os.makedirs(STATIC_FOLDER, exist_ok=True)
    os.makedirs(ANALYSIS_RESULTS_DIR, exist_ok=True)
    app.run(debug=RELOAD, port=PORT, host='0.0.0.0') 