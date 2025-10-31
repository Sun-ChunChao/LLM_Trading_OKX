import os
import time
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
import re
from dotenv import load_dotenv
import json
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, List, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


class TradingConfig:
    """
    交易配置类 - 集中管理所有配置参数
    
    功能说明：
    - 管理交易对、杠杆、时间周期等基础配置
    - 管理AI分析相关配置
    - 管理智能仓位管理系统配置
    - 所有配置集中在一处，方便统一管理和修改
    """

    def __init__(self):
        # ==================== 基础交易配置 ====================
        self.symbol = 'BTC/USDT:USDT'  # 交易对：BTC永续合约，USDT结算
        self.leverage = 15              # 杠杆倍数：15倍杠杆
        self.timeframe = '15m'          # K线周期：15分钟周期（用于技术分析）
        self.test_mode = True           # 模拟盘模式：True=模拟盘, False=实盘（生产环境需谨慎）
        
        # ==================== 数据获取配置 ====================
        self.data_points = 96           # 历史数据点数：96个15分钟K线 = 24小时数据
        
        # ==================== AI配置 ====================
        self.ai_provider = 'qwen'       # AI提供商：'qwen'=通义千问, 'deepseek'=DeepSeek
        
        # ==================== 合约规格配置 ====================
        self.contract_size = 0.01       # BTC合约乘数：每张合约代表0.01个BTC（由交易所API动态获取）
        self.min_amount = 0.01          # 最小交易量：0.01张合约（由交易所限制决定）

        # ==================== 分析周期配置 ====================
        # 用于多时间框架技术分析的周期设置
        self.analysis_periods = {
            'short_term': 21,   # 短期均线周期：21个K线点（约5.25小时）
            'medium_term': 55,  # 中期均线周期：55个K线点（约13.75小时）
            'long_term': 89     # 长期均线周期：89个K线点（约22.25小时）
        }

        # ==================== 智能仓位管理配置 ====================
        # 根据信号信心度、趋势强度、市场波动等因素动态调整仓位大小
        self.position_management = {
            # 核心开关
            'enable_intelligent_position': False,  # 启用智能仓位：True=根据信心度调整, False=固定仓位
            
            # 固定仓位配置（当智能仓位关闭时使用）
            'fixed_contracts': 0.5,               # 固定仓位大小：0.5张合约（约等于0.005 BTC）
            
            # 基础仓位配置（当智能仓位开启时使用）
            'base_usdt_amount': 50,               # 基础下单金额：50 USDT作为基础单位（可根据账户资金调整）
            
            # 信心度乘数（根据AI给出的信心等级调整仓位）
            'high_confidence_multiplier': 2.0,    # 高信心乘数：HIGH信心时仓位翻倍（50*2=100 USDT）
            'medium_confidence_multiplier': 1.0,  # 中等信心乘数：MEDIUM信心保持基础仓位（50*1=50 USDT）
            'low_confidence_multiplier': 0.3,     # 低信心乘数：LOW信心减仓到30%（50*0.3=15 USDT）
            
            # 风险控制
            'max_position_ratio': 0.15,           # 最大仓位比例：单次开仓不超过账户余额的15%
            
            # 市场状态乘数（根据市场条件进一步微调）
            'trend_strength_multiplier': 1.5,     # 趋势强度乘数：强势趋势时增加50%仓位
            'volatility_multiplier': 0.8,         # 波动率乘数：高波动时减少20%仓位（降低风险）
            'rsi_extreme_multiplier': 0.5         # RSI极端值乘数：RSI>75或<25时减半仓位（避免追高杀跌）
        }


class AIClientManager:
    """AI客户端管理器"""

    def __init__(self):
        self.clients = {
            'deepseek': OpenAI(
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com"
            ),
            'qwen': OpenAI(
                api_key=os.getenv('QWEN_API_KEY'),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        }

    def get_client(self, provider: str) -> OpenAI:
        """获取指定AI客户端"""
        if provider not in self.clients:
            logger.warning(f"AI提供商 {provider} 不存在，使用 qwen")
            provider = 'qwen'
        return self.clients[provider]

    def get_model_name(self, provider: str) -> str:
        """获取模型名称"""
        return "deepseek-chat" if provider == 'deepseek' else "qwen-max"


class ExchangeManager:
    """交易所管理器"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchange = self._initialize_exchange()

    def _initialize_exchange(self) -> ccxt.Exchange:
        """初始化交易所连接"""
        exchange = ccxt.okx({
            'options': {'defaultType': 'swap'},
            'apiKey': os.getenv('OKX_API_KEY'),
            'secret': os.getenv('OKX_SECRET'),
            'password': os.getenv('OKX_PASSWORD'),
        })

        if self.config.test_mode:
            exchange.set_sandbox_mode(True)
            logger.info("⚠️ 使用 OKX 模拟盘环境")
        else:
            logger.info("⚠️ 使用 OKX 实盘环境")

        return exchange

    def setup_exchange(self) -> bool:
        """设置交易所参数"""
        try:
            logger.info("🔍 获取BTC合约规格...")
            markets = self.exchange.load_markets()
            btc_market = markets[self.config.symbol]

            # 存储合约规格
            self.config.contract_size = float(btc_market['contractSize'])
            self.config.min_amount = btc_market['limits']['amount']['min']

            logger.info(f"✅ 合约规格: 1张 = {self.config.contract_size} BTC")
            logger.info(f"📏 最小交易量: {self.config.min_amount} 张")

            # 检查现有持仓
            if not self._check_position_mode():
                return False

            # 设置交易模式
            self._set_trading_mode()

            # 验证设置
            self._validate_setup()

            logger.info("🎯 程序配置完成：全仓模式 + 单向持仓")
            return True

        except Exception as e:
            logger.error(f"❌ 交易所设置失败: {e}")
            return False

    def _check_position_mode(self) -> bool:
        """检查持仓模式"""
        positions = self.exchange.fetch_positions([self.config.symbol])

        for pos in positions:
            if pos['symbol'] == self.config.symbol:
                contracts = float(pos.get('contracts', 0))
                mode = pos.get('mgnMode')

                if contracts > 0 and mode == 'isolated':
                    logger.error("❌ 检测到逐仓持仓，程序无法继续运行！")
                    logger.info(f"📊 逐仓持仓详情: 方向:{pos.get('side')}, 数量:{contracts}")
                    return False
        return True

    def _set_trading_mode(self):
        """设置交易模式"""
        try:
            # 设置单向持仓模式
            self.exchange.set_position_mode(False, self.config.symbol)
            logger.info("✅ 已设置单向持仓模式")

            # 设置全仓模式和杠杆
            self.exchange.set_leverage(
                self.config.leverage,
                self.config.symbol,
                {'mgnMode': 'cross'}
            )
            logger.info(f"✅ 已设置全仓模式，杠杆倍数: {self.config.leverage}x")

        except Exception as e:
            logger.warning(f"⚠️ 设置交易模式失败 (可能已设置): {e}")

    def _validate_setup(self):
        """验证设置"""
        balance = self.exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        logger.info(f"💰 当前USDT余额: {usdt_balance:.2f}")

        current_pos = self.get_current_position()
        if current_pos:
            logger.info(f"📦 当前持仓: {current_pos['side']}仓 {current_pos['size']}张")
        else:
            logger.info("📦 当前无持仓")

    def get_current_position(self) -> Optional[Dict]:
        """获取当前持仓情况"""
        try:
            positions = self.exchange.fetch_positions([self.config.symbol])

            for pos in positions:
                if pos['symbol'] == self.config.symbol:
                    contracts = float(pos['contracts']) if pos['contracts'] else 0

                    if contracts > 0:
                        return {
                            'side': pos['side'],
                            'size': contracts,
                            'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                            'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                            'leverage': float(pos['leverage']) if pos['leverage'] else self.config.leverage,
                            'symbol': pos['symbol']
                        }
            return None

        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
            return None


class TechnicalAnalyzer:
    """技术分析器"""

    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        try:
            # 移动平均线
            for window in [5, 20, 50, 60]:
                df[f'sma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
            
            # EMA指标
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()

            # MACD
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # RSI - 多个周期
            for period in [7, 14]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

            # 布林带
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            # 成交量分析
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']

            # ATR (平均真实范围)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_3'] = true_range.rolling(3).mean()
            df['atr_14'] = true_range.rolling(14).mean()

            # 波动率计算
            df['volatility'] = df['close'].pct_change().rolling(20).std()

            # 支撑阻力位
            df['resistance'] = df['high'].rolling(20).max()
            df['support'] = df['low'].rolling(20).min()

            return df.bfill().ffill()

        except Exception as e:
            logger.error(f"技术指标计算失败: {e}")
            return df

    @staticmethod
    def get_market_trend(df: pd.DataFrame) -> Dict[str, Any]:
        """判断市场趋势 - 优化版：更积极识别趋势"""
        try:
            current_price = df['close'].iloc[-1]

            # 多时间框架趋势分析
            trend_short = "上涨" if current_price > df['ema_20'].iloc[-1] else "下跌"
            trend_medium = "上涨" if current_price > df['ema_50'].iloc[-1] else "下跌"

            # MACD趋势
            macd_trend = "bullish" if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else "bearish"

            # 计算趋势强度（均线斜率）
            ema_20_slope = (df['ema_20'].iloc[-1] - df['ema_20'].iloc[-5]) / df['ema_20'].iloc[-5]
            ema_50_slope = (df['ema_50'].iloc[-1] - df['ema_50'].iloc[-10]) / df['ema_50'].iloc[-10]
            trend_strength = abs(ema_20_slope) + abs(ema_50_slope)

            # 🔥 优化：更积极的趋势判断
            # 只要主趋势明确或短期趋势配合MACD，就判定为强势
            if trend_short == "上涨" or (trend_medium == "上涨" and macd_trend == "bullish"):
                overall_trend = "强势上涨"
            elif trend_short == "下跌" or (trend_medium == "下跌" and macd_trend == "bearish"):
                overall_trend = "强势下跌"
            else:
                overall_trend = "震荡整理"

            return {
                'short_term': trend_short,
                'medium_term': trend_medium,
                'macd': macd_trend,
                'overall': overall_trend,
                'rsi_7': df['rsi_7'].iloc[-1],
                'rsi_14': df['rsi_14'].iloc[-1],
                'volatility': df['volatility'].iloc[-1] if 'volatility' in df else 0,
                'trend_strength': trend_strength  # 新增：趋势强度指标
            }
        except Exception as e:
            logger.error(f"趋势分析失败: {e}")
            return {}

    @staticmethod
    def get_support_resistance_levels(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
        """计算支撑阻力位"""
        try:
            recent_high = df['high'].tail(lookback).max()
            recent_low = df['low'].tail(lookback).min()
            current_price = df['close'].iloc[-1]

            return {
                'static_resistance': recent_high,
                'static_support': recent_low,
                'dynamic_resistance': df['bb_upper'].iloc[-1],
                'dynamic_support': df['bb_lower'].iloc[-1],
                'price_vs_resistance': ((recent_high - current_price) / current_price) * 100,
                'price_vs_support': ((current_price - recent_low) / recent_low) * 100
            }
        except Exception as e:
            logger.error(f"支撑阻力计算失败: {e}")
            return {}


class SentimentAnalyzer:
    """情绪分析器"""

    def __init__(self):
        self.api_url = "https://service.cryptoracle.network/openapi/v2/endpoint"
        self.api_key = "7ad48a56-8730-4238-a714-eebc30834e3e"

    def get_sentiment_indicators(self) -> Optional[Dict]:
        """获取情绪指标"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=4)

            request_body = {
                "apiKey": self.api_key,
                "endpoints": ["CO-A-02-01", "CO-A-02-02"],
                "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "timeType": "15m",
                "token": ["BTC"]
            }

            headers = {"Content-Type": "application/json", "X-API-KEY": self.api_key}
            response = requests.post(self.api_url, json=request_body, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return self._parse_sentiment_data(data)

            return None

        except Exception as e:
            logger.error(f"情绪指标获取失败: {e}")
            return None

    def _parse_sentiment_data(self, data: Dict) -> Optional[Dict]:
        """解析情绪数据"""
        if data.get("code") != 200 or not data.get("data"):
            return None

        time_periods = data["data"][0]["timePeriods"]

        for period in time_periods:
            period_data = period.get("data", [])
            sentiment = {}

            for item in period_data:
                endpoint = item.get("endpoint")
                value = item.get("value", "").strip()

                if value and endpoint in ["CO-A-02-01", "CO-A-02-02"]:
                    try:
                        sentiment[endpoint] = float(value)
                    except (ValueError, TypeError):
                        continue

            if "CO-A-02-01" in sentiment and "CO-A-02-02" in sentiment:
                positive = sentiment['CO-A-02-01']
                negative = sentiment['CO-A-02-02']
                net_sentiment = positive - negative

                data_delay = int((datetime.now() - datetime.strptime(
                    period['startTime'], '%Y-%m-%d %H:%M:%S')).total_seconds() // 60)

                logger.info(f"✅ 使用情绪数据时间: {period['startTime']} (延迟: {data_delay}分钟)")

                return {
                    'positive_ratio': positive,
                    'negative_ratio': negative,
                    'net_sentiment': net_sentiment,
                    'data_time': period['startTime'],
                    'data_delay_minutes': data_delay
                }

        logger.warning("❌ 所有时间段数据都为空")
        return None


class PositionManager:
    """仓位管理器"""

    def __init__(self, config: TradingConfig, exchange_manager: ExchangeManager):
        self.config = config
        self.exchange_manager = exchange_manager

    def calculate_intelligent_position(self, signal_data: Dict, price_data: Dict,
                                       current_position: Optional[Dict]) -> float:
        """计算智能仓位大小"""
        config = self.config.position_management

        # 如果禁用智能仓位，使用固定仓位（从配置读取）
        if not config.get('enable_intelligent_position', True):
            fixed_contracts = config.get('fixed_contracts', 0.1)
            logger.info(f"🔧 智能仓位已禁用，使用固定仓位: {fixed_contracts} 张")
            logger.info(f"   固定仓位价值约: {fixed_contracts * price_data['price'] * self.config.contract_size:.2f} USDT")
            return fixed_contracts

        try:
            balance = self.exchange_manager.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            base_usdt = config['base_usdt_amount']

            logger.info(f"💰 可用USDT余额: {usdt_balance:.2f}, 下单基数{base_usdt}")

            # 1. 信心程度调整
            confidence_multiplier = {
                'HIGH': config['high_confidence_multiplier'],
                'MEDIUM': config['medium_confidence_multiplier'],
                'LOW': config['low_confidence_multiplier']
            }.get(signal_data['confidence'], 1.0)

            # 2. 趋势强度调整
            trend = price_data['trend_analysis'].get('overall', '震荡整理')
            trend_multiplier = config['trend_strength_multiplier'] if trend in ['强势上涨', '强势下跌'] else 1.0

            # 3. RSI状态调整
            rsi = price_data['technical_data'].get('rsi_7', 50)
            rsi_multiplier = config.get('rsi_extreme_multiplier', 0.5) if rsi > 80 or rsi < 20 else 1.0

            # 4. 波动率调整
            volatility = price_data['trend_analysis'].get('volatility', 0.02)
            volatility_multiplier = max(0.3, min(1.0, 1 - (volatility * 10)))  # 波动率越高，仓位越小

            # 5. 当前持仓调整
            position_multiplier = 1.0
            if current_position:
                # 如果信号方向与当前持仓相同，可以考虑加仓
                if (signal_data['signal'] == 'BUY' and current_position['side'] == 'long') or \
                   (signal_data['signal'] == 'SELL' and current_position['side'] == 'short'):
                    position_multiplier = 1.2  # 同向加仓
                else:
                    position_multiplier = 0.8  # 反向减仓

            # 计算建议投入金额
            suggested_usdt = base_usdt * confidence_multiplier * trend_multiplier * rsi_multiplier * volatility_multiplier * position_multiplier
            
            # 风险控制：最大仓位限制
            max_usdt = usdt_balance * config['max_position_ratio']
            final_usdt = min(suggested_usdt, max_usdt)

            # 计算合约张数
            contract_size = final_usdt / (price_data['price'] * self.config.contract_size)
            contract_size = round(contract_size, 2)  # OKX精度处理

            # 确保最小交易量
            if contract_size < self.config.min_amount:
                contract_size = self.config.min_amount
                logger.warning(f"⚠️ 仓位小于最小值，调整为: {contract_size} 张")

            logger.info(f"🎯 仓位计算详情:")
            logger.info(f"   - 信心乘数: {confidence_multiplier}")
            logger.info(f"   - 趋势乘数: {trend_multiplier}")
            logger.info(f"   - RSI乘数: {rsi_multiplier}")
            logger.info(f"   - 波动率乘数: {volatility_multiplier:.2f}")
            logger.info(f"   - 持仓乘数: {position_multiplier}")
            logger.info(f"   - 最终仓位: {final_usdt:.2f} USDT → {contract_size:.2f} 张合约")

            return contract_size

        except Exception as e:
            logger.error(f"❌ 仓位计算失败，使用基础仓位: {e}")
            # 紧急备用计算
            base_usdt = config['base_usdt_amount']
            contract_size = base_usdt / (price_data['price'] * self.config.contract_size)
            return round(max(contract_size, self.config.min_amount), 2)


class TradingBot:
    """主交易机器人"""

    def __init__(self):
        self.config = TradingConfig()
        self.ai_manager = AIClientManager()
        self.exchange_manager = ExchangeManager(self.config)
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.position_manager = PositionManager(self.config, self.exchange_manager)

        # 状态存储
        self.signal_history = []
        self.price_history = []

        # 性能跟踪
        self.trade_performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'trade_history': []
        }

    def update_trade_performance(self, signal_data: Dict, executed_price: float, pnl: float):
        """更新交易性能统计"""
        self.trade_performance['total_trades'] += 1
        if pnl > 0:
            self.trade_performance['winning_trades'] += 1
        else:
            self.trade_performance['losing_trades'] += 1
        self.trade_performance['total_pnl'] += pnl

        trade_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'signal': signal_data['signal'],
            'price': executed_price,
            'pnl': pnl,
            'confidence': signal_data['confidence'],
            'position_size': signal_data.get('position_size', 0)
        }
        self.trade_performance['trade_history'].append(trade_record)

        # 保持历史记录长度
        if len(self.trade_performance['trade_history']) > 100:
            self.trade_performance['trade_history'].pop(0)

    def print_performance_summary(self):
        """打印性能总结"""
        if self.trade_performance['total_trades'] == 0:
            return

        win_rate = (self.trade_performance['winning_trades'] / self.trade_performance['total_trades']) * 100
        avg_pnl = self.trade_performance['total_pnl'] / self.trade_performance['total_trades']

        logger.info("\n" + "=" * 30)
        logger.info("📈 策略性能总结")
        logger.info("=" * 30)
        logger.info(f"总交易次数: {self.trade_performance['total_trades']}")
        logger.info(f"盈利交易: {self.trade_performance['winning_trades']}")
        logger.info(f"亏损交易: {self.trade_performance['losing_trades']}")
        logger.info(f"胜率: {win_rate:.2f}%")
        logger.info(f"总盈亏: {self.trade_performance['total_pnl']:.2f} USDT")
        logger.info(f"平均盈亏: {avg_pnl:.2f} USDT")
        
        # 添加仓位统计
        if self.trade_performance['trade_history']:
            avg_position_size = sum(t['position_size'] for t in self.trade_performance['trade_history']) / len(self.trade_performance['trade_history'])
            logger.info(f"平均仓位: {avg_position_size:.2f} 张")
        
        logger.info("=" * 30)

    def get_btc_ohlcv_enhanced(self) -> Optional[Dict]:
        """获取增强版K线数据"""
        try:
            ohlcv = self.exchange_manager.exchange.fetch_ohlcv(
                self.config.symbol,
                self.config.timeframe,
                limit=self.config.data_points
            )

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # 计算技术指标
            df = self.technical_analyzer.calculate_technical_indicators(df)

            current_data = df.iloc[-1]
            previous_data = df.iloc[-2]

            # 获取分析数据
            trend_analysis = self.technical_analyzer.get_market_trend(df)
            levels_analysis = self.technical_analyzer.get_support_resistance_levels(df)

            return {
                'price': current_data['close'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'high': current_data['high'],
                'low': current_data['low'],
                'volume': current_data['volume'],
                'timeframe': self.config.timeframe,
                'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
                'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(10).to_dict('records'),
                'technical_data': {
                    'sma_5': current_data.get('sma_5', 0),
                    'sma_20': current_data.get('sma_20', 0),
                    'sma_50': current_data.get('sma_50', 0),
                    'sma_60': current_data.get('sma_60', 0),
                    'ema_20': current_data.get('ema_20', 0),
                    'ema_50': current_data.get('ema_50', 0),
                    'rsi_7': current_data.get('rsi_7', 0),
                    'rsi_14': current_data.get('rsi_14', 0),
                    'macd': current_data.get('macd', 0),
                    'macd_signal': current_data.get('macd_signal', 0),
                    'macd_histogram': current_data.get('macd_histogram', 0),
                    'bb_upper': current_data.get('bb_upper', 0),
                    'bb_lower': current_data.get('bb_lower', 0),
                    'bb_position': current_data.get('bb_position', 0),
                    'volume_ratio': current_data.get('volume_ratio', 0),
                    'atr_3': current_data.get('atr_3', 0),
                    'atr_14': current_data.get('atr_14', 0),
                    'volatility': current_data.get('volatility', 0)
                },
                'trend_analysis': trend_analysis,
                'levels_analysis': levels_analysis,
                'full_data': df
            }
        except Exception as e:
            logger.error(f"获取增强K线数据失败: {e}")
            return None

    def get_btc_4h_ohlcv_enhanced(self) -> Optional[Dict]:
        """获取4小时K线数据用于长期背景分析"""
        try:
            ohlcv = self.exchange_manager.exchange.fetch_ohlcv(
                self.config.symbol,
                '4h',  # 4小时时间框架
                limit=96  # 获取足够的数据点
            )

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # 计算技术指标
            df = self.technical_analyzer.calculate_technical_indicators(df)

            current_data = df.iloc[-1]

            return {
                'price': current_data['close'],
                'volume': current_data['volume'],
                'technical_data': {
                    'ema_20': current_data.get('ema_20', 0),
                    'ema_50': current_data.get('ema_50', 0),
                    'atr_3': current_data.get('atr_3', 0),
                    'atr_14': current_data.get('atr_14', 0),
                    'volume_ma': current_data.get('volume_ma', 0)
                },
                'full_data': df
            }
        except Exception as e:
            logger.error(f"获取4小时K线数据失败: {e}")
            return None

    def generate_technical_analysis_text(self, price_data: Dict) -> str:
        """生成技术分析文本 - 使用真实数据"""
        if 'technical_data' not in price_data:
            return "技术指标数据不可用"

        tech = price_data['technical_data']
        trend = price_data.get('trend_analysis', {})
        levels = price_data.get('levels_analysis', {})

        def safe_float(value, default=0):
            return float(value) if value and pd.notna(value) else default

        # 获取真实序列数据
        full_data = price_data.get('full_data', pd.DataFrame())
        if not full_data.empty:
            # 使用真实的收盘价序列
            mid_price_series = [round(x, 2) for x in full_data['close'].tail(10).tolist()]
            ema_20_series = [round(x, 3) for x in full_data['ema_20'].tail(10).tolist()]
            macd_series = [round(x, 3) for x in full_data['macd'].tail(10).tolist()]
            rsi_7_series = [round(x, 3) for x in full_data['rsi_7'].tail(10).tolist()]
            rsi_14_series = [round(x, 3) for x in full_data['rsi_14'].tail(10).tolist()]
        else:
            # 如果数据为空，使用中性值
            current_price = price_data['price']
            mid_price_series = [current_price] * 10
            ema_20_series = [current_price] * 10
            macd_series = [0] * 10
            rsi_7_series = [50] * 10
            rsi_14_series = [50] * 10

        # 获取4小时真实数据
        h4_data = self.get_btc_4h_ohlcv_enhanced()
        if h4_data and not h4_data.get('full_data', pd.DataFrame()).empty:
            h4_full_data = h4_data['full_data']
            h4_ema_20 = safe_float(h4_data['technical_data'].get('ema_20', 0))
            h4_ema_50 = safe_float(h4_data['technical_data'].get('ema_50', 0))
            h4_atr_3 = safe_float(h4_data['technical_data'].get('atr_3', 0))
            h4_atr_14 = safe_float(h4_data['technical_data'].get('atr_14', 0))
            h4_volume = safe_float(h4_data.get('volume', 0))
            h4_volume_avg = safe_float(h4_data['technical_data'].get('volume_ma', 0))
            h4_macd_series = [round(x, 3) for x in h4_full_data['macd'].tail(10).tolist()]
            h4_rsi_14_series = [round(x, 3) for x in h4_full_data['rsi_14'].tail(10).tolist()]
        else:
            # 使用当前时间框架数据作为备用
            h4_ema_20 = safe_float(tech['ema_20'])
            h4_ema_50 = safe_float(tech['ema_50'])
            h4_atr_3 = safe_float(tech['atr_3'])
            h4_atr_14 = safe_float(tech['atr_14'])
            h4_volume = safe_float(price_data.get('volume', 0))
            h4_volume_avg = safe_float(tech.get('volume_ma', 0))
            h4_macd_series = macd_series
            h4_rsi_14_series = rsi_14_series

        analysis_text = f"""
【当前市场状态】
当前价格 = {price_data['price']:,.2f}
当前20周期EMA = {safe_float(tech['ema_20']):.3f}
当前MACD = {safe_float(tech['macd']):.3f}
当前RSI（7周期）= {safe_float(tech['rsi_7']):.3f}
当前波动率 = {safe_float(tech.get('volatility', 0)):.4f}

【日内序列数据（按{self.config.timeframe}，从旧到新）】
中间价格：
{mid_price_series}
EMA指标（20周期）：
{ema_20_series}
MACD指标：
{macd_series}
RSI指标（7周期）：
{rsi_7_series}
RSI指标（14周期）：
{rsi_14_series}

【长期背景数据（4小时时间框架）】
20周期EMA：{h4_ema_20:,.3f} vs. 50周期EMA：{h4_ema_50:,.3f}
3周期ATR：{h4_atr_3:.3f} vs. 14周期ATR：{h4_atr_14:.3f}
当前成交量：{h4_volume:,.3f} vs. 平均成交量：{h4_volume_avg:,.3f}
MACD指标：
{h4_macd_series}
RSI指标（14周期）：
{h4_rsi_14_series}

【趋势分析】
短期趋势: {trend.get('short_term', 'N/A')}
中期趋势: {trend.get('medium_term', 'N/A')}
整体趋势: {trend.get('overall', 'N/A')}
MACD方向: {trend.get('macd', 'N/A')}
RSI状态: {safe_float(tech['rsi_7']):.1f} ({'超买' if safe_float(tech['rsi_7']) > 70 else '超卖' if safe_float(tech['rsi_7']) < 30 else '中性'})
        """
        return analysis_text

    def safe_json_parse(self, json_str: str) -> Optional[Dict]:
        """安全解析JSON"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # 修复常见的JSON格式问题
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败，原始内容: {json_str}")
                logger.error(f"错误详情: {e}")
                return None

    def create_fallback_signal(self, price_data: Dict) -> Dict:
        """创建备用交易信号"""
        return {
            "signal": "HOLD",
            "reason": "因技术分析暂时不可用，采取保守策略",
            "stop_loss": price_data['price'] * 0.98,
            "take_profit": price_data['price'] * 1.02,
            "confidence": "LOW",
            "is_fallback": True
        }

    def analyze_with_ai(self, price_data: Dict) -> Dict:
        """使用AI分析市场并生成交易信号"""

        # 生成技术分析文本
        technical_analysis = self.generate_technical_analysis_text(price_data)

        # 构建K线数据文本
        kline_text = f"【最近5根{self.config.timeframe}K线数据】\n"
        for i, kline in enumerate(price_data['kline_data'][-5:]):
            trend = "阳线" if kline['close'] > kline['open'] else "阴线"
            change = ((kline['close'] - kline['open']) / kline['open']) * 100
            kline_text += f"K线{i + 1}: {trend} 开盘:{kline['open']:.2f} 收盘:{kline['close']:.2f} 涨跌:{change:+.2f}%\n"

        # 添加上次交易信号
        signal_text = ""
        if self.signal_history:
            last_signal = self.signal_history[-1]
            signal_text = f"\n【上次交易信号】\n信号: {last_signal.get('signal', 'N/A')}\n信心: {last_signal.get('confidence', 'N/A')}"

        # 获取情绪数据
        sentiment_data = self.sentiment_analyzer.get_sentiment_indicators()
        if sentiment_data:
            sign = '+' if sentiment_data['net_sentiment'] >= 0 else ''
            sentiment_text = f"【市场情绪】乐观{sentiment_data['positive_ratio']:.1%} 悲观{sentiment_data['negative_ratio']:.1%} 净值{sign}{sentiment_data['net_sentiment']:.3f}"
        else:
            sentiment_text = "【市场情绪】数据暂不可用"

        # 添加当前持仓信息
        current_pos = self.exchange_manager.get_current_position()
        position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}, 盈亏: {current_pos['unrealized_pnl']:.2f}USDT"
        pnl_text = f", 持仓盈亏: {current_pos['unrealized_pnl']:.2f} USDT" if current_pos else ""

        prompt = self._build_analysis_prompt(
            price_data, technical_analysis, kline_text,
            signal_text, sentiment_text, position_text, pnl_text
        )

        return self._get_ai_signal(prompt, price_data)

    def _build_analysis_prompt(self, price_data: Dict, technical_analysis: str,
                               kline_text: str, signal_text: str, sentiment_text: str,
                               position_text: str, pnl_text: str) -> str:
        """构建优化的分析提示词"""

        # 提取关键数据用于决策
        current_price = price_data['price']
        rsi = price_data['technical_data'].get('rsi_7', 50)
        trend = price_data['trend_analysis'].get('overall', '震荡整理')
        macd_trend = price_data['trend_analysis'].get('macd', 'neutral')
        volatility = price_data['technical_data'].get('volatility', 0.02)

        return f"""
# 角色设定
你是一个专业的加密货币量化交易员，专注于BTC/USDT {self.config.timeframe}周期的趋势跟踪策略。你的任务是基于客观技术分析给出明确的交易信号。

## 市场数据概览
- 当前价格: ${current_price:,.2f}
- 时间: {price_data['timestamp']}
- 价格变化: {price_data['price_change']:+.2f}%
- 当前持仓: {position_text}{pnl_text}
- 市场波动率: {volatility:.4f}

## 技术分析核心指标
{technical_analysis}

## 近期K线形态
{kline_text}

## 市场情绪
{sentiment_text}

## 交易历史
{signal_text}

# 交易决策框架

## 1. 趋势判断优先级（按重要性排序）
### 🥇 趋势方向确认
- **多头趋势**: 价格 > 20MA > 50MA，且均线向上发散 → 强烈BUY信号
- **空头趋势**: 价格 < 20MA < 50MA，且均线向下发散 → 强烈SELL信号  
- **震荡市场**: 均线纠缠，价格在区间内波动 → HOLD或区间交易

### 🥈 动量确认
- **RSI解读**: 
  - 30-70: 健康范围，不作为主要HOLD理由
  - >70: 超买，但强势趋势中可忽略
  - <30: 超卖，但下跌趋势中可忽略
- **MACD确认**: 金叉/死叉需结合趋势背景

### 🥉 关键位置突破
- 突破前高/前低 + 成交量放大 = 高信心信号
- 支撑阻力测试 + 反转形态 = 潜在反转信号

## 2. 信号生成规则【优化版 - 更积极的交易策略】

### 🔴 BUY条件（满足任一即可开仓，满足越多信心越高）:
1. **价格站上20EMA**，且20EMA有向上趋势（趋势确认）
2. **MACD金叉**或MACD > 0且持续上升（动量确认）
3. **RSI从超卖区(<30)反弹**至30-50区间（超卖反弹）
4. **价格突破近期阻力位**且收盘站稳（突破确认）
5. **成交量放大** + 阳线收盘（量价配合）

**信心等级判断**：
- 满足3项及以上 → **HIGH**（强烈买入信号）
- 满足2项 → **MEDIUM**（中等买入信号）
- 满足1项但趋势明确（强势上涨） → **LOW**（仍可交易，小仓位）

### 🔵 SELL条件（满足任一即可开仓，满足越多信心越高）:
1. **价格跌破20EMA**，且20EMA有向下趋势（趋势确认）
2. **MACD死叉**或MACD < 0且持续下降（动量确认）
3. **RSI从超买区(>70)回落**至50-70区间（超买回落）
4. **价格跌破近期支撑位**且收盘确认（跌破确认）
5. **成交量放大** + 阴线收盘（量价配合）

**信心等级判断**：
- 满足3项及以上 → **HIGH**（强烈卖出信号）
- 满足2项 → **MEDIUM**（中等卖出信号）
- 满足1项但趋势明确（强势下跌） → **LOW**（仍可交易，小仓位）

### 🟡 HOLD条件（必须同时满足以下大部分条件才选择HOLD）:
1. **价格在20EMA和50EMA之间反复穿插**（真正的震荡，非单边）
2. **MACD在零轴附近反复金叉死叉**（无明确方向）
3. **RSI在40-60区间横盘震荡**（无动量突破）
4. **成交量极度萎缩**（<平均成交量的50%）
5. **短期和中期趋势完全矛盾**

⚠️ **重要原则**：
- **趋势优先**：当市场处于"强势上涨"或"强势下跌"时，即使只满足1个BUY/SELL条件，也应该选择跟随趋势交易，而非HOLD！
- **HOLD是例外不是常态**：只有在真正无法判断方向时才HOLD，不要因为谨慎而错过明确的趋势机会
- **持仓管理**：如果已有持仓且与当前趋势一致，可以HOLD保持现状；但如果趋势反转，必须果断平仓反向

## 3. 风险管理规则

### 止损设置逻辑:
- 多头: 最近支撑下方 1-2%
- 空头: 最近阻力上方 1-2%  
- 基于ATR: 1.5-2倍ATR值

### 止盈设置逻辑:
- 多头: 最近阻力位置或2:1风险回报比
- 空头: 最近支撑位置或2:1风险回报比

## 4. 信心等级定义

### HIGH (高信心):
- 多个时间框架确认
- 技术指标高度一致
- 成交量配合
- 趋势明确强劲

### MEDIUM (中等信心):  
- 主要趋势明确但局部有噪音
- 关键技术位突破
- 部分指标确认

### LOW (低信心):
- 指标矛盾
- 成交量低迷
- 在关键位置犹豫

## 5. 特殊情况处理

### 持仓管理:
- 已有持仓且趋势延续 → 保持或适度加仓
- 趋势明确反转 → 及时反向操作
- 避免因已有持仓而错过新趋势

### 震荡市场策略:
- 布林带收窄 + 低波动 → 减少交易频率
- 等待明确突破信号

# 当前决策要点

**趋势状态**: {trend}
**RSI位置**: {rsi:.1f} ({'超买' if rsi > 70 else '超卖' if rsi < 30 else '中性'})
**MACD方向**: {macd_trend}
**波动率**: {volatility:.4f} ({'高波动' if volatility > 0.03 else '低波动' if volatility < 0.01 else '正常波动'})
**价格位置**: ${current_price:,.2f}

# 输出要求

基于以上分析，给出明确的交易决策。必须严格按照以下JSON格式回复，不要额外解释：

{{
    "signal": "BUY|SELL|HOLD",
    "reason": "简洁的技术分析理由，包含关键指标状态",
    "stop_loss": 具体数值,
    "take_profit": 具体数值,
    "confidence": "HIGH|MEDIUM|LOW"
}}

示例响应:
{{
    "signal": "BUY", 
    "reason": "多头排列确认，价格突破前高，RSI健康",
    "stop_loss": 68500.50,
    "take_profit": 72000.00,
    "confidence": "HIGH"
}}
"""

    def _get_ai_signal(self, prompt: str, price_data: Dict) -> Dict:
        """获取AI信号"""
        try:
            ai_provider = self.config.ai_provider
            ai_client = self.ai_manager.get_client(ai_provider)
            model_name = self.ai_manager.get_model_name(ai_provider)

            response = ai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system",
                     "content": f"您是一位专业的交易员，专注于{self.config.timeframe}周期趋势分析。请结合K线形态和技术指标做出判断，并严格遵循JSON格式要求。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0.2
            )

            result = response.choices[0].message.content
            logger.info(f"原始回复: {result}")

            # 提取JSON部分
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = result[start_idx:end_idx]
                signal_data = self.safe_json_parse(json_str)
            else:
                signal_data = None

            if not signal_data:
                signal_data = self.create_fallback_signal(price_data)

            # 验证必需字段
            required_fields = ['signal', 'reason', 'stop_loss', 'take_profit', 'confidence']
            if not all(field in signal_data for field in required_fields):
                signal_data = self.create_fallback_signal(price_data)

            # 保存信号到历史记录
            signal_data['timestamp'] = price_data['timestamp']
            self.signal_history.append(signal_data)
            if len(self.signal_history) > 30:
                self.signal_history.pop(0)

            # 信号统计
            self._analyze_signal_statistics(signal_data)

            return signal_data

        except Exception as e:
            logger.error(f"AI分析失败: {e}")
            return self.create_fallback_signal(price_data)

    def _analyze_signal_statistics(self, signal_data: Dict):
        """分析信号统计"""
        signal_count = len([s for s in self.signal_history if s.get('signal') == signal_data['signal']])
        total_signals = len(self.signal_history)
        logger.info(f"信号统计: {signal_data['signal']} (最近{total_signals}次中出现{signal_count}次)")

        # 信号连续性检查
        if len(self.signal_history) >= 3:
            last_three = [s['signal'] for s in self.signal_history[-3:]]
            if len(set(last_three)) == 1:
                logger.warning(f"⚠️ 注意：连续3次{signal_data['signal']}信号")

    def analyze_with_ai_with_retry(self, price_data: Dict, max_retries: int = 2) -> Dict:
        """带重试的AI分析"""
        for attempt in range(max_retries):
            try:
                signal_data = self.analyze_with_ai(price_data)
                if signal_data and not signal_data.get('is_fallback', False):
                    return signal_data

                logger.warning(f"第{attempt + 1}次尝试失败，进行重试...")
                time.sleep(1)

            except Exception as e:
                logger.error(f"第{attempt + 1}次尝试异常: {e}")
                if attempt == max_retries - 1:
                    return self.create_fallback_signal(price_data)
                time.sleep(1)

        return self.create_fallback_signal(price_data)

    def execute_intelligent_trade(self, signal_data: Dict, price_data: Dict):
        """执行智能交易"""
        current_position = self.exchange_manager.get_current_position()

        # 计算智能仓位
        position_size = self.position_manager.calculate_intelligent_position(
            signal_data, price_data, current_position
        )

        # 将仓位大小添加到信号数据中，用于性能统计
        signal_data['position_size'] = position_size

        logger.info(f"🎯 交易信号: {signal_data['signal']}")
        logger.info(f"📊 信心程度: {signal_data['confidence']}")
        logger.info(f"💰 智能仓位: {position_size:.2f} 张")
        logger.info(f"📝 理由: {signal_data['reason']}")
        logger.info(f"📦 当前持仓: {current_position}")

        # 🎯 模拟盘专用日志
        if self.config.test_mode:
            logger.info("🔄 模拟盘交易执行中...")

            # 记录详细的模拟交易信息
            log_trade_details = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'signal': signal_data['signal'],
                'position_size': position_size,
                'price': price_data['price'],
                'confidence': signal_data['confidence'],
                'current_position': current_position,
                'reason': signal_data['reason']
            }
            logger.info(f"📋 交易详情: {log_trade_details}")
        else:
            logger.info("🚀 实盘交易执行中...")

        # 风险管理 - 模拟盘可以更宽松
        if not self.config.test_mode and signal_data['confidence'] == 'LOW':
            logger.warning("⚠️ 低信心信号，实盘跳过执行")
            return

        try:
            # 记录交易前状态
            before_position = self.exchange_manager.get_current_position()
            before_balance = self.exchange_manager.exchange.fetch_balance()['USDT']['free']

            # 执行交易逻辑
            self._execute_trade_order(signal_data, current_position, position_size)
            logger.info("✅ 智能交易执行成功")

            time.sleep(2)

            # 记录交易后状态
            after_position = self.exchange_manager.get_current_position()
            after_balance = self.exchange_manager.exchange.fetch_balance()['USDT']['free']

            # 计算本次交易的盈亏（简化版）
            pnl_change = 0
            if before_position and after_position:
                pnl_change = after_position.get('unrealized_pnl', 0) - before_position.get('unrealized_pnl', 0)

            # 更新性能统计
            self.update_trade_performance(signal_data, price_data['price'], pnl_change)

            # 定期打印性能总结
            if self.trade_performance['total_trades'] % 10 == 0:
                self.print_performance_summary()

            logger.info(f"📊 更新后持仓: {after_position}")

        except Exception as e:
            logger.error(f"❌ 交易执行失败: {e}")
            self._handle_trade_error(e, signal_data, position_size)

    def _execute_trade_order(self, signal_data: Dict, current_position: Optional[Dict], position_size: float):
        """执行交易订单"""
        exchange = self.exchange_manager.exchange

        if signal_data['signal'] == 'BUY':
            self._handle_buy_signal(exchange, current_position, position_size)
        elif signal_data['signal'] == 'SELL':
            self._handle_sell_signal(exchange, current_position, position_size)
        elif signal_data['signal'] == 'HOLD':
            logger.info("建议观望，不执行交易")

    def _handle_buy_signal(self, exchange: ccxt.Exchange, current_position: Optional[Dict], position_size: float):
        """处理买入信号"""
        if current_position and current_position['side'] == 'short':
            # 平空仓并开多仓
            if current_position['size'] > 0:
                logger.info(f"平空仓 {current_position['size']:.2f} 张并开多仓 {position_size:.2f} 张...")
                exchange.create_market_order(
                    self.config.symbol, 'buy', current_position['size'],
                    params={'reduceOnly': True}
                )
                time.sleep(1)
            exchange.create_market_order(
                self.config.symbol, 'buy', position_size,
                params={}
            )
        elif current_position and current_position['side'] == 'long':
            # 调整多仓
            self._adjust_position(exchange, current_position, position_size, 'long')
        else:
            # 开多仓
            logger.info(f"开多仓 {position_size:.2f} 张...")
            exchange.create_market_order(
                self.config.symbol, 'buy', position_size,
                params={}
            )

    def _handle_sell_signal(self, exchange: ccxt.Exchange, current_position: Optional[Dict], position_size: float):
        """处理卖出信号"""
        if current_position and current_position['side'] == 'long':
            # 平多仓并开空仓
            if current_position['size'] > 0:
                logger.info(f"平多仓 {current_position['size']:.2f} 张并开空仓 {position_size:.2f} 张...")
                exchange.create_market_order(
                    self.config.symbol, 'sell', current_position['size'],
                    params={'reduceOnly': True}
                )
                time.sleep(1)
            exchange.create_market_order(
                self.config.symbol, 'sell', position_size,
                params={}
            )
        elif current_position and current_position['side'] == 'short':
            # 调整空仓
            self._adjust_position(exchange, current_position, position_size, 'short')
        else:
            # 开空仓
            logger.info(f"开空仓 {position_size:.2f} 张...")
            exchange.create_market_order(
                self.config.symbol, 'sell', position_size,
                params={}
            )

    def _adjust_position(self, exchange: ccxt.Exchange, current_position: Dict,
                         target_size: float, side: str):
        """调整持仓大小"""
        size_diff = target_size - current_position['size']

        if abs(size_diff) >= 0.01:  # 有可调整的差异
            if size_diff > 0:
                # 加仓
                add_size = round(size_diff, 2)
                logger.info(
                    f"{side}仓加仓 {add_size:.2f} 张 (当前:{current_position['size']:.2f} → 目标:{target_size:.2f})")
                order_side = 'buy' if side == 'long' else 'sell'
                exchange.create_market_order(
                    self.config.symbol, order_side, add_size,
                    params={}
                )
            else:
                # 减仓
                reduce_size = round(abs(size_diff), 2)
                logger.info(
                    f"{side}仓减仓 {reduce_size:.2f} 张 (当前:{current_position['size']:.2f} → 目标:{target_size:.2f})")
                order_side = 'sell' if side == 'long' else 'buy'
                exchange.create_market_order(
                    self.config.symbol, order_side, reduce_size,
                    params={'reduceOnly': True}
                )
        else:
            logger.info(
                f"已有{side}头持仓，仓位合适保持现状 (当前:{current_position['size']:.2f}, 目标:{target_size:.2f})")

    def _handle_trade_error(self, error: Exception, signal_data: Dict, position_size: float):
        """处理交易错误"""
        exchange = self.exchange_manager.exchange

        # 如果是持仓不存在的错误，尝试直接开新仓
        if "don't have any positions" in str(error):
            logger.info("尝试直接开新仓...")
            try:
                if signal_data['signal'] == 'BUY':
                    exchange.create_market_order(
                        self.config.symbol, 'buy', position_size,
                        params={}
                    )
                elif signal_data['signal'] == 'SELL':
                    exchange.create_market_order(
                        self.config.symbol, 'sell', position_size,
                        params={}
                    )
                logger.info("直接开仓成功")
            except Exception as e2:
                logger.error(f"直接开仓也失败: {e2}")

    def wait_for_next_period(self) -> int:
        """等待到下一个15分钟整点"""
        now = datetime.now()
        current_minute = now.minute
        current_second = now.second

        # 计算下一个整点时间
        next_period_minute = ((current_minute // 15) + 1) * 15
        if next_period_minute == 60:
            next_period_minute = 0

        # 计算需要等待的总秒数
        if next_period_minute > current_minute:
            minutes_to_wait = next_period_minute - current_minute
        else:
            minutes_to_wait = 60 - current_minute + next_period_minute

        seconds_to_wait = minutes_to_wait * 60 - current_second

        # 显示友好的等待时间
        display_minutes = minutes_to_wait - 1 if current_second > 0 else minutes_to_wait
        display_seconds = 60 - current_second if current_second > 0 else 0

        if display_minutes > 0:
            logger.info(f"🕒 等待 {display_minutes} 分 {display_seconds} 秒到整点...")
        else:
            logger.info(f"🕒 等待 {display_seconds} 秒到整点...")

        return seconds_to_wait

    def run_trading_cycle(self):
        """运行交易周期"""
        # 等待到整点再执行
        wait_seconds = self.wait_for_next_period()
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        logger.info("\n" + "=" * 30)
        logger.info(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 30)

        # 1. 获取增强版K线数据
        price_data = self.get_btc_ohlcv_enhanced()
        if not price_data:
            return

        logger.info(f"BTC当前价格: ${price_data['price']:,.2f}")
        logger.info(f"数据周期: {self.config.timeframe}")
        logger.info(f"价格变化: {price_data['price_change']:+.2f}%")

        # 2. 使用AI分析（带重试）
        signal_data = self.analyze_with_ai_with_retry(price_data)

        if signal_data.get('is_fallback', False):
            logger.warning("⚠️ 使用备用交易信号")

        # 3. 执行智能交易
        self.execute_intelligent_trade(signal_data, price_data)

    def main(self):
        """主函数"""
        logger.info("BTC/USDT OKX自动交易机器人启动成功！")
        logger.info("🎯 模拟盘策略测试模式 - 真实执行模拟交易")

        if self.config.test_mode:
            logger.info("✅ 当前为模拟盘模式，将执行真实模拟交易")
            logger.info("📊 启动性能跟踪系统...")
        else:
            logger.info("🚨 实盘交易模式，请谨慎操作！")

        logger.info(f"交易周期: {self.config.timeframe}")
        logger.info("已启用完整技术指标分析和持仓跟踪功能")
        logger.info("🎯 智能仓位管理已启用 - 仓位将根据市场条件动态调整")

        # 设置交易所
        if not self.exchange_manager.setup_exchange():
            logger.error("交易所初始化失败，程序退出")
            return

        logger.info("执行频率: 每15分钟整点执行")

        # 循环执行
        while True:
            self.run_trading_cycle()
            time.sleep(60)  # 每分钟检查一次


def main():
    """程序入口"""
    try:
        bot = TradingBot()
        bot.main()
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行异常: {e}")
        raise


if __name__ == "__main__":
    main()