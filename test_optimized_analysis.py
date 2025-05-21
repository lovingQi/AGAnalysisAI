"""
测试优化后的股票分析功能
用于验证一个月数据分析和增强的买入信号识别
"""
import sys
import os
import datetime
import time
from stock_analysis_all import analyze_single_stock, calculate_technical_indicators, analyze_buy_signals, save_analysis_chart

def test_single_stock(stock_code, stock_name, trade_days=30, end_date=None):
    """测试单只股票的分析"""
    print(f"\n=== 开始分析股票: {stock_code} {stock_name} ===")
    
    # 确保结果目录存在
    if not os.path.exists('results'):
        os.makedirs('results')
    
    try:
        # 分析股票
        result = analyze_single_stock(stock_code, stock_name, trade_days, end_date)
        
        # 打印分析结果
        print(f"\n分析结果: {stock_code} {stock_name}")
        print(f"评分: {result['score']}")
        print("买入信号:")
        for signal in result['signals']:
            print(f" - {signal}")
        
        if 'chart_path' in result:
            print(f"分析图表已保存至: {result['chart_path']}")
        
        return result
    except Exception as e:
        print(f"分析 {stock_code} {stock_name} 时发生错误: {str(e)}")
        return {
            "code": stock_code,
            "name": stock_name,
            "signals": [f"分析出错: {str(e)}"],
            "score": 0
        }

def main():
    # 测试股票列表 - 可以手动添加要测试的股票
    test_stocks = [
        {'code': '600519', 'name': '贵州茅台'},  # 白酒龙头
        {'code': '000001', 'name': '平安银行'},  # 银行股
        {'code': '300750', 'name': '宁德时代'},  # 新能源
        {'code': '000333', 'name': '美的集团'},  # 家电
        {'code': '600036', 'name': '招商银行'}   # 银行
    ]
    
    # 设置分析日期参数
    trade_days = 30
    
    # 使用2023年的可靠日期以确保有足够数据
    end_date = datetime.datetime(2023, 12, 29)  # 2023年12月29日，保证是交易日
    
    print(f"开始测试优化后的股票分析功能，获取最近的{trade_days}个交易日数据")
    print(f"测试日期截止至: {end_date.strftime('%Y-%m-%d')}")
    print(f"待测试股票: {[stock['name'] for stock in test_stocks]}")
    
    # 分析结果
    results = []
    
    # 对每只股票进行分析
    for i, stock in enumerate(test_stocks):
        print(f"\n[{i+1}/{len(test_stocks)}] 正在分析: {stock['name']}")
        
        try:
            result = test_single_stock(stock['code'], stock['name'], trade_days, end_date)
            results.append(result)
            
            # 添加适当延时以避免请求过于频繁
            if i < len(test_stocks) - 1:  # 最后一个股票不需要延时
                print("等待1秒以避免请求过于频繁...")
                time.sleep(1)
                
        except Exception as e:
            print(f"分析 {stock['code']} {stock['name']} 时出错: {str(e)}")
    
    # 按评分排序
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 打印总结
    print("\n\n===== 分析总结 =====")
    print(f"共分析了 {len(results)} 只股票")
    
    print("\n评分排序结果:")
    for i, result in enumerate(results, 1):
        signal_count = len(result['signals'])
        print(f"{i}. {result['code']} {result['name']} - 评分: {result['score']} (信号数: {signal_count})")
        
    # 保存排名前3的股票信息到CSV
    if results:
        import pandas as pd
        top_stocks = [r for r in results if r['score'] > 0][:3]
        if top_stocks:
            csv_data = []
            
            for result in top_stocks:
                signals_str = '; '.join(result['signals'])
                row = {
                    '股票代码': result['code'],
                    '股票名称': result['name'],
                    '买入信号': signals_str,
                    '评分': result['score']
                }
                csv_data.append(row)
            
            # 保存为CSV
            df_result = pd.DataFrame(csv_data)
            output_file = 'results/top_stock_signals.csv'
            df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"前3名股票结果已保存到 {output_file}")
        else:
            print("没有评分大于0的股票，不保存CSV文件")

if __name__ == "__main__":
    main() 