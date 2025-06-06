import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
import time
import os

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
    
    return df

# 分析股票的买入信号
def analyze_buy_signals(df, stock_code, stock_name):
    """分析股票的买入信号"""
    signals = []
    score = 0
    
    # 检查数据是否足够
    if len(df) < 60:
        return {"code": stock_code, "name": stock_name, "signals": ["数据不足"], "score": 0}
    
    # 确保所有必要的列都存在
    required_cols = ['收盘', 'MA5', 'MA20', 'RSI', 'MACD', 'DIF', 'DEA', 'K', 'D', 'J']
    for col in required_cols:
        if col not in df.columns or df[col].isnull().all():
            return {"code": stock_code, "name": stock_name, "signals": ["数据计算错误"], "score": 0}
    
    try:
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
            signals.append("MACD金叉形成")
            score += 20
        
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
            
    except Exception as e:
        signals.append(f"分析过程出错: {str(e)}")
        score = 0
    
    return {
        "code": stock_code,
        "name": stock_name,
        "signals": signals,
        "score": score
    }

# 获取并分析单只股票数据
def analyze_single_stock(stock_code, stock_name):
    """获取并分析单只股票的数据"""
    try:
        # 获取日K线数据
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                                start_date=None, end_date=None, 
                                adjust="qfq")
        
        # 计算技术指标
        df = calculate_technical_indicators(df)
        
        # 分析买入信号
        result = analyze_buy_signals(df, stock_code, stock_name)
        
        # 如果得分超过30，生成图表
        if result['score'] >= 30:
            save_analysis_chart(df, stock_code, stock_name, result)
        
        return result
    
    except Exception as e:
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
        
        # 价格和均线
        plt.subplot(3, 1, 1)
        plt.plot(df.index[-60:], df['收盘'].iloc[-60:], label='收盘价')
        plt.plot(df.index[-60:], df['MA5'].iloc[-60:], label='MA5')
        plt.plot(df.index[-60:], df['MA20'].iloc[-60:], label='MA20')
        plt.plot(df.index[-60:], df['MA60'].iloc[-60:], label='MA60')
        plt.title(f'{stock_code} {stock_name} - 股价走势与技术指标', **chinese_font)
        plt.legend(prop=fontprop if 'fontproperties' in chinese_font else None)
        plt.grid(True)
        
        # MACD
        plt.subplot(3, 1, 2)
        plt.plot(df.index[-60:], df['DIF'].iloc[-60:], label='DIF')
        plt.plot(df.index[-60:], df['DEA'].iloc[-60:], label='DEA')
        plt.bar(df.index[-60:], df['MACD'].iloc[-60:], label='MACD', color=['r' if x > 0 else 'g' for x in df['MACD'].iloc[-60:]])
        plt.title('MACD指标', **chinese_font)
        plt.legend(prop=fontprop if 'fontproperties' in chinese_font else None)
        plt.grid(True)
        
        # KDJ和RSI
        plt.subplot(3, 1, 3)
        plt.plot(df.index[-60:], df['K'].iloc[-60:], label='K')
        plt.plot(df.index[-60:], df['D'].iloc[-60:], label='D')
        plt.plot(df.index[-60:], df['J'].iloc[-60:], label='J')
        plt.plot(df.index[-60:], df['RSI'].iloc[-60:], label='RSI', color='purple')
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
    
    # 使用tqdm显示进度条
    for idx, row in tqdm(stock_list.iterrows(), total=total_stocks, desc="分析进度"):
        stock_code = row['代码']
        stock_name = row['名称']
        
        # 分析股票
        result = analyze_single_stock(stock_code, stock_name)
        
        # 如果有买入信号并且分数大于30，加入结果列表
        if result['score'] >= 30:
            results.append(result)
        
        # 防止频繁请求被限制
        time.sleep(0.5)
    
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
            row = {
                '股票代码': result['code'],
                '股票名称': result['name'],
                '买入信号': signals_str,
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
        print("  买入信号:")
        for signal in result['signals']:
            print(f"   - {signal}")
        if 'chart_path' in result:
            print(f"  分析图表已保存到: {result['chart_path']}")

if __name__ == "__main__":
    main()