import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
import time
import os
import datetime

# 设置中文显示
try:
    font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    fontprop = FontProperties(fname=font_path)
    matplotlib.rcParams['axes.unicode_minus'] = False
    chinese_font = {'fontproperties': fontprop}
except:
    print("无法加载中文字体，将使用默认字体")
    chinese_font = {}

# 创建结果保存目录
if not os.path.exists('results'):
    os.makedirs('results')

# 获取沪深A股所有股票列表
def get_stock_list():
    """获取沪深A股所有股票列表"""
    print("正在获取股票列表...")
    stock_info_a_code_name_df = ak.stock_info_a_code_name()
    # 打印列名以便调试
    print(f"股票列表列名: {stock_info_a_code_name_df.columns.tolist()}")
    return stock_info_a_code_name_df

# 计算技术指标
def calculate_technical_indicators(df):
    """计算各种技术指标"""
    # 确保数据类型正确
    for col in ['开盘', '收盘', '最高', '最低', '成交量']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 计算移动平均线
    df['MA5'] = df['收盘'].rolling(window=5).mean()
    df['MA10'] = df['收盘'].rolling(window=10).mean()
    df['MA20'] = df['收盘'].rolling(window=20).mean()
    df['MA60'] = df['收盘'].rolling(window=60).mean()
    
    # 计算RSI(相对强弱指标)
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 计算MACD
    df['EMA12'] = df['收盘'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['收盘'].ewm(span=26, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    
    # 计算KDJ
    low_min = df['最低'].rolling(window=9).min()
    high_max = df['最高'].rolling(window=9).max()
    df['RSV'] = 100 * ((df['收盘'] - low_min) / (high_max - low_min + 1e-9))
    df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # 计算布林带
    df['BOLL_MID'] = df['收盘'].rolling(window=20).mean()
    std = df['收盘'].rolling(window=20).std()
    df['BOLL_UPPER'] = df['BOLL_MID'] + 2 * std
    df['BOLL_LOWER'] = df['BOLL_MID'] - 2 * std
    
    # 计算ATR（平均真实波幅）用于设置止损
    tr1 = df['最高'] - df['最低']
    tr2 = abs(df['最高'] - df['收盘'].shift(1))
    tr3 = abs(df['最低'] - df['收盘'].shift(1))
    df['TR'] = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    return df

# 计算买入价格和止损价格
def calculate_price_targets(df):
    """计算买入价格和止损价格
    
    返回:
        dict: 包含买入价格和止损价格的字典
    """
    current_price = df['收盘'].iloc[-1]
    result = {
        'current_price': current_price,
        'buy_prices': [],
        'stop_loss_prices': []
    }
    
    try:
        # 1. 计算支撑位买入价格 - 取近期低点附近
        recent_lows = df['最低'].tail(10).nsmallest(3)  # 最近10天中最低的3个点
        support_level = recent_lows.mean()  # 取这些低点的平均值作为支撑位
        if support_level < current_price:
            # 支撑位买入策略
            support_buy_price = round(support_level, 2)
            result['buy_prices'].append({
                'price': support_buy_price,
                'type': '支撑位买入',
                'discount': f"{round((1 - support_buy_price / current_price) * 100, 2)}%"
            })
        
        # 2. 布林带下轨买入
        if 'BOLL_LOWER' in df.columns and not df['BOLL_LOWER'].isnull().iloc[-1]:
            boll_lower = df['BOLL_LOWER'].iloc[-1]
            if boll_lower < current_price:
                boll_buy_price = round(boll_lower, 2)
                result['buy_prices'].append({
                    'price': boll_buy_price,
                    'type': '布林带下轨买入',
                    'discount': f"{round((1 - boll_buy_price / current_price) * 100, 2)}%"
                })
        
        # 3. MA20回调买入点
        if 'MA20' in df.columns and not df['MA20'].isnull().iloc[-1]:
            ma20 = df['MA20'].iloc[-1]
            if ma20 < current_price:
                ma20_buy_price = round(ma20, 2)
                result['buy_prices'].append({
                    'price': ma20_buy_price,
                    'type': 'MA20均线回调买入',
                    'discount': f"{round((1 - ma20_buy_price / current_price) * 100, 2)}%"
                })
        
        # 4. 小幅回调买入点 - 当前价格的98%
        slight_pullback = round(current_price * 0.98, 2)
        result['buy_prices'].append({
            'price': slight_pullback,
            'type': '小幅回调买入(2%)',
            'discount': "2.00%"
        })
        
        # 按价格从高到低排序
        result['buy_prices'].sort(key=lambda x: x['price'], reverse=True)
        
        # 计算止损价格
        
        # 1. 近期低点止损 - 取最近20天的最低价作为止损点
        recent_low = df['最低'].tail(20).min()
        stop_loss_by_low = round(recent_low * 0.98, 2)  # 略低于近期最低点
        stop_loss_pct = round((1 - stop_loss_by_low / current_price) * 100, 2)
        
        result['stop_loss_prices'].append({
            'price': stop_loss_by_low,
            'type': '近期低点止损',
            'risk': f"{stop_loss_pct}%"
        })
        
        # 2. ATR止损 - 当前价格减去2倍ATR
        if 'ATR' in df.columns and not df['ATR'].isnull().iloc[-1]:
            atr = df['ATR'].iloc[-1]
            atr_stop_loss = round(current_price - 2 * atr, 2)
            atr_stop_loss_pct = round((1 - atr_stop_loss / current_price) * 100, 2)
            
            result['stop_loss_prices'].append({
                'price': atr_stop_loss,
                'type': 'ATR止损(2倍ATR)',
                'risk': f"{atr_stop_loss_pct}%"
            })
        
        # 3. 固定百分比止损 - 当前价格的92%（8%止损）
        fixed_pct_stop_loss = round(current_price * 0.92, 2)
        result['stop_loss_prices'].append({
            'price': fixed_pct_stop_loss,
            'type': '固定百分比止损',
            'risk': "8.00%"
        })
        
        # 按风险从小到大排序（价格从高到低）
        result['stop_loss_prices'].sort(key=lambda x: x['price'], reverse=True)
        
    except Exception as e:
        print(f"计算价格目标时出错: {str(e)}")
    
    return result

# 分析股票的买入信号
def analyze_buy_signals(df, stock_code, stock_name):
    """分析股票的买入信号"""
    signals = []
    score = 0
    
    # 检查数据是否足够
    if len(df) < 20:  # 至少需要20天数据
        return {"code": stock_code, "name": stock_name, "signals": ["数据不足"], "score": 0}
    
    # 确保所有必要的列都存在
    required_cols = ['收盘', 'MA5', 'MA20', 'RSI', 'MACD', 'DIF', 'DEA', 'K', 'D', 'J']
    for col in required_cols:
        if col not in df.columns or df[col].isnull().all():
            return {"code": stock_code, "name": stock_name, "signals": ["数据计算错误"], "score": 0}
    
    try:
        # 分析中期趋势（过去一个月）
        month_trend = analyze_monthly_trend(df)
        if month_trend['is_uptrend']:
            signals.append(f"过去一个月呈上升趋势: {month_trend['reason']}")
            score += 15
        
        # 1. MA5与MA20金叉信号(5日均线上穿20日均线)
        if df['MA5'].iloc[-1] > df['MA20'].iloc[-1] and df['MA5'].iloc[-2] <= df['MA20'].iloc[-2]:
            signals.append("MA金叉形成")
            score += 20
        
        # 2. RSI指标分析
        current_rsi = df['RSI'].iloc[-1]
        if 30 <= current_rsi <= 50:
            signals.append(f"RSI值为{current_rsi:.2f}，处于低位回升阶段")
            score += 15
        elif current_rsi < 30:
            signals.append(f"RSI值为{current_rsi:.2f}，股票可能超卖")
            score += 10
        
        # 3. MACD金叉
        if df['DIF'].iloc[-1] > df['DEA'].iloc[-1] and df['DIF'].iloc[-2] <= df['DEA'].iloc[-2]:
            # 判断MACD柱状图的强度
            macd_strength = df['MACD'].iloc[-1]
            if macd_strength > 0:
                signals.append(f"MACD金叉形成，柱状图为正 ({macd_strength:.3f})")
                score += 20
            else:
                signals.append("MACD金叉形成")
                score += 15
        
        # 4. KDJ金叉
        if df['K'].iloc[-1] > df['D'].iloc[-1] and df['K'].iloc[-2] <= df['D'].iloc[-2]:
            if df['K'].iloc[-1] < 50:  # K值在低位金叉更有效
                signals.append("KDJ低位金叉")
                score += 15
            else:
                signals.append("KDJ金叉")
                score += 10
        
        # 5. 放量上涨
        avg_volume = df['成交量'].iloc[-6:-1].mean()  # 前5天平均成交量
        current_volume = df['成交量'].iloc[-1]
        price_change = (df['收盘'].iloc[-1] / df['收盘'].iloc[-2] - 1) * 100
        
        if current_volume > 1.5 * avg_volume and price_change > 0:
            signals.append(f"放量上涨: 量比{current_volume/avg_volume:.2f}, 涨幅{price_change:.2f}%")
            score += 15
        
        # 6. 突破前高
        recent_high = df['最高'].iloc[-20:-1].max()
        if df['收盘'].iloc[-1] > recent_high:
            signals.append("突破20日前高")
            score += 15
        
        # 7. 多均线多头排列分析（新增）
        if is_bullish_ma_alignment(df):
            signals.append("均线呈多头排列，趋势向上")
            score += 15
            
        # 计算买入价格和止损价格
        price_targets = calculate_price_targets(df)
        
    except Exception as e:
        signals.append(f"分析过程出错: {str(e)}")
        score = 0
    
    return {
        "code": stock_code,
        "name": stock_name,
        "signals": signals,
        "score": score,
        "price_targets": price_targets if 'price_targets' in locals() else None
    }

# 分析月度趋势
def analyze_monthly_trend(df):
    """分析过去一个月的趋势情况"""
    result = {
        'is_uptrend': False,
        'reason': ''
    }
    
    try:
        # 如果数据不足30天，使用所有可用数据
        days = min(30, len(df))
        
        # 计算月初和月末价格
        start_price = df['收盘'].iloc[-days]
        end_price = df['收盘'].iloc[-1]
        price_change = (end_price / start_price - 1) * 100
        
        # 计算30天内收盘价上涨的天数比例
        up_days = sum(1 for i in range(-days+1, 0) if df['收盘'].iloc[i] > df['收盘'].iloc[i-1])
        up_ratio = up_days / (days - 1)
        
        # 判断均线系统是否向上
        ma_up = (df['MA5'].iloc[-1] > df['MA5'].iloc[-days//2]) and \
                (df['MA20'].iloc[-1] > df['MA20'].iloc[-days//2])
        
        # 综合判断趋势
        if price_change > 5 and up_ratio > 0.5 and ma_up:
            result['is_uptrend'] = True
            result['reason'] = f"月涨幅{price_change:.2f}%, 上涨占比{up_ratio:.2f}, 均线向上"
        elif price_change > 10:
            result['is_uptrend'] = True
            result['reason'] = f"强势上涨{price_change:.2f}%"
        elif ma_up and up_ratio > 0.6:
            result['is_uptrend'] = True
            result['reason'] = f"均线向上, 上涨天数占比{up_ratio:.2f}"
    
    except Exception as e:
        print(f"分析月度趋势出错: {str(e)}")
    
    return result

# 判断是否多头排列
def is_bullish_ma_alignment(df):
    """判断均线是否呈多头排列"""
    try:
        # 需要MA5 > MA10 > MA20 > MA60
        if df['MA5'].iloc[-1] > df['MA10'].iloc[-1] > df['MA20'].iloc[-1] > df['MA60'].iloc[-1]:
            # 确认趋势 - 至少持续3天
            days_aligned = 0
            for i in range(-3, 0):
                if df['MA5'].iloc[i] > df['MA10'].iloc[i] > df['MA20'].iloc[i] > df['MA60'].iloc[i]:
                    days_aligned += 1
            
            return days_aligned >= 2  # 至少2天多头排列
        return False
    except:
        return False

# 获取并分析单只股票数据
def analyze_single_stock(stock_code, stock_name, trade_days=30, end_date=None):
    """获取并分析单只股票的数据
    
    参数:
        stock_code: 股票代码
        stock_name: 股票名称
        trade_days: 需要获取的交易日数量，默认30个交易日
        end_date: 结束日期，默认为当前日期
    """
    try:
        # 获取结束日期
        if end_date is None:
            end_date = datetime.datetime.now()
            
        # 格式化结束日期为字符串 YYYYMMDD
        end_date_str = end_date.strftime("%Y%m%d")
        
        # 为确保获取足够的交易日数据，将自然日范围扩大到交易日数量的2倍左右
        # 考虑周末和节假日，30个交易日约等于42个自然日
        calendar_days = trade_days * 2  
        start_date = end_date - datetime.timedelta(days=calendar_days)
        start_date_str = start_date.strftime("%Y%m%d")
        
        print(f"获取 {stock_code} {stock_name} 从 {start_date_str} 到 {end_date_str} 的数据")
        
        # 获取日K线数据，明确指定日期范围
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                              start_date=start_date_str, end_date=end_date_str, 
                              adjust="qfq")
        
        # 保留最新的N个交易日数据
        if len(df) > trade_days:
            print(f"获取到 {len(df)} 天数据，将使用最新的 {trade_days} 个交易日数据")
            df = df.tail(trade_days)
        
        # 如果数据不足，给出警告
        if len(df) < trade_days * 0.7:  # 如果获取的数据不足预期交易日的70%
            print(f"警告: {stock_code} {stock_name} 只获取到 {len(df)} 个交易日数据，少于预期的 {trade_days} 个交易日")
            if len(df) < 10:  # 如果数据少于10天，则认为数据不足
                return {
                    "code": stock_code,
                    "name": stock_name,
                    "signals": ["数据不足，需要至少10个交易日数据"],
                    "score": 0
                }
        else:
            print(f"成功获取 {stock_code} {stock_name} 的 {len(df)} 个交易日数据")
            
        # 计算技术指标
        df = calculate_technical_indicators(df)
        
        # 分析买入信号
        result = analyze_buy_signals(df, stock_code, stock_name)
        
        # 如果得分超过30，生成图表
        if result['score'] >= 30:
            save_analysis_chart(df, stock_code, stock_name, result)
        
        return result
    
    except Exception as e:
        print(f"分析 {stock_code} 时出错: {str(e)}")
        return {
            "code": stock_code,
            "name": stock_name,
            "signals": [f"分析失败: {str(e)}"],
            "score": 0
        }

# 保存分析图表
def save_analysis_chart(df, stock_code, stock_name, result):
    """生成并保存股票分析图表"""
    try:
        plt.figure(figsize=(15, 12))
        
        # 使用所有可用数据，确保最少显示30天
        data_range = min(len(df), max(30, len(df)))
        
        # 价格和均线
        plt.subplot(3, 1, 1)
        plt.plot(df.index[-data_range:], df['收盘'].iloc[-data_range:], label='收盘价')
        plt.plot(df.index[-data_range:], df['MA5'].iloc[-data_range:], label='MA5')
        plt.plot(df.index[-data_range:], df['MA20'].iloc[-data_range:], label='MA20')
        plt.plot(df.index[-data_range:], df['MA60'].iloc[-data_range:], label='MA60')
        
        # 如果有布林带数据，也绘制布林带
        if 'BOLL_UPPER' in df.columns and 'BOLL_LOWER' in df.columns:
            plt.plot(df.index[-data_range:], df['BOLL_UPPER'].iloc[-data_range:], 'r--', label='布林上轨')
            plt.plot(df.index[-data_range:], df['BOLL_MID'].iloc[-data_range:], 'g--', label='布林中轨')
            plt.plot(df.index[-data_range:], df['BOLL_LOWER'].iloc[-data_range:], 'b--', label='布林下轨')
        
        # 标记买入价格范围
        if 'price_targets' in result and result['price_targets'] and 'buy_prices' in result['price_targets']:
            current_price = df['收盘'].iloc[-1]
            buy_prices = result['price_targets']['buy_prices']
            if buy_prices:
                min_buy_price = min([bp['price'] for bp in buy_prices])
                plt.axhline(y=min_buy_price, color='g', linestyle='-.')
                plt.fill_between(df.index[-data_range:], min_buy_price, current_price, color='g', alpha=0.1)
                plt.text(df.index[-data_range//2], min_buy_price * 0.99, f'买入区间', color='g', fontproperties=fontprop if 'fontproperties' in chinese_font else None)
            
            # 标记止损价格
            stop_prices = result['price_targets']['stop_loss_prices']
            if stop_prices:
                max_stop_price = max([sp['price'] for sp in stop_prices])
                plt.axhline(y=max_stop_price, color='r', linestyle='-.')
                plt.text(df.index[-data_range//2], max_stop_price * 0.99, f'止损位', color='r', fontproperties=fontprop if 'fontproperties' in chinese_font else None)
        
        plt.title(f'{stock_code} {stock_name} - 股价走势与技术指标', **chinese_font)
        plt.legend(prop=fontprop if 'fontproperties' in chinese_font else None)
        plt.grid(True)
        
        # MACD
        plt.subplot(3, 1, 2)
        plt.plot(df.index[-data_range:], df['DIF'].iloc[-data_range:], label='DIF')
        plt.plot(df.index[-data_range:], df['DEA'].iloc[-data_range:], label='DEA')
        plt.bar(df.index[-data_range:], df['MACD'].iloc[-data_range:], label='MACD', color=['r' if x > 0 else 'g' for x in df['MACD'].iloc[-data_range:]])
        plt.title('MACD指标', **chinese_font)
        plt.legend(prop=fontprop if 'fontproperties' in chinese_font else None)
        plt.grid(True)
        
        # KDJ和RSI
        plt.subplot(3, 1, 3)
        plt.plot(df.index[-data_range:], df['K'].iloc[-data_range:], label='K')
        plt.plot(df.index[-data_range:], df['D'].iloc[-data_range:], label='D')
        plt.plot(df.index[-data_range:], df['J'].iloc[-data_range:], label='J')
        plt.plot(df.index[-data_range:], df['RSI'].iloc[-data_range:], label='RSI', color='purple')
        plt.axhline(y=80, color='r', linestyle='--')
        plt.axhline(y=20, color='g', linestyle='--')
        plt.title('KDJ与RSI指标', **chinese_font)
        plt.legend(prop=fontprop if 'fontproperties' in chinese_font else None)
        plt.grid(True)
        
        plt.tight_layout()
        file_path = f'results/{stock_code}_{stock_name}_analysis.png'
        plt.savefig(file_path)
        plt.close()
        
        # 添加图表路径到结果中
        result['chart_path'] = file_path
    except Exception as e:
        print(f"生成图表出错({stock_code}): {str(e)}")

# 主函数
def main():
    # 获取所有股票列表
    stock_list = get_stock_list()
    total_stocks = len(stock_list)
    print(f"共获取到 {total_stocks} 只股票")
    
    # 创建结果列表
    results = []
    
    # 设置分析交易日数量
    analysis_trade_days = 30  # 分析过去30个交易日数据
    
    # 检查DataFrame的列名
    if '代码' in stock_list.columns:
        code_column = '代码'
        name_column = '名称'
    elif 'code' in stock_list.columns:
        code_column = 'code'
        name_column = 'name'
    elif 'symbol' in stock_list.columns:
        code_column = 'symbol'
        name_column = 'name'
    else:
        print(f"无法识别的列名格式: {stock_list.columns.tolist()}")
        return
    
    # 使用tqdm显示进度条
    for idx, row in tqdm(stock_list.iterrows(), total=total_stocks, desc="分析进度"):
        try:
            stock_code = row[code_column]
            stock_name = row[name_column]
            
            # 分析股票，指定获取过去30个交易日数据
            result = analyze_single_stock(stock_code, stock_name, trade_days=analysis_trade_days)
            
            # 如果有买入信号并且分数大于30，加入结果列表
            if result['score'] >= 30:
                results.append(result)
            
            # 防止频繁请求被限制
            time.sleep(0.01)  # 减少延时时间，加快分析速度
            
        except Exception as e:
            print(f"处理股票时出错: {str(e)}")
            continue
    
    # 按得分排序结果
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 输出结果
    print("\n\n===== 分析结果 =====")
    print(f"共找到 {len(results)} 只可能值得买入的股票")
    
    # 保存结果到CSV
    if results:
        # 准备CSV数据
        csv_data = []
        for result in results:
            signals_str = '; '.join(result['signals'])
            
            # 处理买入价格和止损价格
            buy_price_str = ''
            stop_loss_str = ''
            if 'price_targets' in result and result['price_targets']:
                if 'buy_prices' in result['price_targets'] and result['price_targets']['buy_prices']:
                    buy_prices = result['price_targets']['buy_prices']
                    buy_price_list = [f"{bp['type']}: {bp['price']}({bp['discount']}折扣)" for bp in buy_prices]
                    buy_price_str = '; '.join(buy_price_list)
                
                if 'stop_loss_prices' in result['price_targets'] and result['price_targets']['stop_loss_prices']:
                    stop_prices = result['price_targets']['stop_loss_prices']
                    stop_price_list = [f"{sp['type']}: {sp['price']}(风险{sp['risk']})" for sp in stop_prices]
                    stop_loss_str = '; '.join(stop_price_list)
            
            current_price = result['price_targets']['current_price'] if 'price_targets' in result and result['price_targets'] else 0
            
            row = {
                '股票代码': result['code'],
                '股票名称': result['name'],
                '当前价格': current_price,
                '买入信号': signals_str,
                '建议买入价格': buy_price_str,
                '建议止损价格': stop_loss_str,
                '评分': result['score'],
                '图表路径': result.get('chart_path', '')
            }
            csv_data.append(row)
        
        # 保存为CSV
        df_result = pd.DataFrame(csv_data)
        df_result.to_csv('results/buy_signals.csv', index=False, encoding='utf-8-sig')
        print("结果已保存到 results/buy_signals.csv")
    
    # 打印前10只股票的详细信息
    print("\n前10只推荐买入的股票：")
    for i, result in enumerate(results[:10], 1):
        print(f"\n{i}. {result['code']} {result['name']} (评分: {result['score']})")
        
        # 打印当前价格
        if 'price_targets' in result and result['price_targets']:
            current_price = result['price_targets']['current_price']
            print(f"  当前价格: {current_price}")
        
        print("  买入信号:")
        for signal in result['signals']:
            print(f"   - {signal}")
            
        # 打印买入价格建议
        if 'price_targets' in result and result['price_targets'] and 'buy_prices' in result['price_targets']:
            print("  买入价格建议:")
            for bp in result['price_targets']['buy_prices']:
                print(f"   - {bp['type']}: {bp['price']} (折扣: {bp['discount']})")
        
        # 打印止损价格建议
        if 'price_targets' in result and result['price_targets'] and 'stop_loss_prices' in result['price_targets']:
            print("  止损价格建议:")
            for sp in result['price_targets']['stop_loss_prices']:
                print(f"   - {sp['type']}: {sp['price']} (风险: {sp['risk']})")
            
        if 'chart_path' in result:
            print(f"  分析图表已保存到: {result['chart_path']}")

if __name__ == "__main__":
    main()