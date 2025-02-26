import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

RELOAD = True if os.getenv('RELOAD', 'false').lower() == 'true' else False
PORT = int(os.getenv('PORT', 11000))

XQ_TOKEN = os.getenv('XQ_TOKEN', '')
U_UID = os.getenv('U_UID', '')

SCKEY = "xq_a_token=" + XQ_TOKEN + ";u=" + U_UID

# Flask应用配置
STATIC_FOLDER = 'static'
STATIC_URL_PATH = '/static'

# 分析结果配置
ANALYSIS_RESULTS_DIR = f"{STATIC_FOLDER}/analysis_results"
CHARTS_SUBDIR = "charts"
DATA_SUBDIR = "data"

# API配置
API_VERSION = 'v1'
DEFAULT_STOCK = 'SH688981'
DEFAULT_START_DATE = '2024-01-01'
DEFAULT_END_DATE = '2025-02-25'
DEFAULT_WINDOW = 5
DEFAULT_THRESHOLD = 0.5  # 百分比