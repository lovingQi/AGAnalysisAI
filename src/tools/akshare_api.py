import os
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

from data.cache import get_cache
from data.models import (
    Price,
    PriceResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    LineItem,
    LineItemResponse,
    CompanyNews,
    CompanyNewsResponse,
)

# 全局缓存实例
_cache = get_cache()

def convert_ticker_to_akshare_format(ticker: str) -> str:
    """将标准股票代码转换为akshare格式的代码"""
    # 移除可能的市场后缀
    ticker = ticker.split('.')[0]
    
    # 检查是否已经是akshare格式
    if ticker.startswith('sh') or ticker.startswith('sz'):
        return ticker
    
    # 根据代码判断市场
    code = ticker.strip()
    if code.startswith('6'):
        return f"sh{code}"
    elif code.startswith(('0', '3')):
        return f"sz{code}"
    elif code.startswith('8'):
        return f"bj{code}"
    else:
        # 默认沪市
        return f"sh{code}"

def convert_date_format(date_str: str) -> str:
    """将YYYY-MM-DD格式转换为akshare使用的YYYYMMDD格式"""
    return date_str.replace('-', '')

def get_a_stock_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """获取A股价格数据"""
    # 检查缓存
    if cached_data := _cache.get_prices(f"A_{ticker}"):
        # 按日期范围过滤并转换为Price对象
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    # 转换为akshare格式
    ak_ticker = convert_ticker_to_akshare_format(ticker)
    
    try:
        # 使用akshare获取股票历史数据
        stock_data = ak.stock_zh_a_hist(
            symbol=ak_ticker,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # 前复权
        )
        
        # 检查是否有数据
        if stock_data.empty:
            return []
        
        # 转换为Price对象列表
        prices = []
        for _, row in stock_data.iterrows():
            date_str = row['日期']
            if isinstance(date_str, str):
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                date_formatted = date_obj.strftime('%Y-%m-%d')
            else:
                date_formatted = date_str.strftime('%Y-%m-%d')
                
            price = Price(
                open=float(row['开盘']),
                close=float(row['收盘']),
                high=float(row['最高']),
                low=float(row['最低']),
                volume=int(row['成交量']),
                time=date_formatted
            )
            prices.append(price)
            
        # 缓存结果
        _cache.set_prices(f"A_{ticker}", [p.model_dump() for p in prices])
        return prices
        
    except Exception as e:
        print(f"获取A股价格数据出错 ({ticker}): {e}")
        return []

def get_a_stock_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "年报",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """获取A股财务指标"""
    # 检查缓存
    if cached_data := _cache.get_financial_metrics(f"A_{ticker}"):
        # 按日期范围过滤
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        if filtered_data:
            return filtered_data[:limit]
            
    # 转换为akshare格式
    ak_ticker = convert_ticker_to_akshare_format(ticker)
    stock_code = ak_ticker[2:]  # 去掉市场前缀
    
    try:
        # 获取财务指标数据
        # 注意：akshare的财务指标与美股API的字段不完全一致，需要映射
        financial_data = ak.stock_financial_analysis_indicator(stock=stock_code)
        
        # 检查是否有数据
        if financial_data.empty:
            return []
            
        # 获取市值数据用于计算指标
        market_cap = get_a_stock_market_cap(ticker, end_date)
        
        # 转换为FinancialMetrics对象列表
        metrics_list = []
        for _, row in financial_data.iterrows():
            # 通常数据是按年份排列的，提取年份作为报告期
            report_period = str(row['年份']) + "-12-31"
            if report_period > end_date:
                continue
                
            # 创建指标对象
            metrics = FinancialMetrics(
                ticker=ticker,
                report_period=report_period,
                period="ttm" if period == "年报" else period,
                currency="CNY",
                market_cap=market_cap,
                
                # 收益率指标
                return_on_equity=float(row['净资产收益率(%)']) / 100 if not pd.isna(row['净资产收益率(%)']) else None,
                return_on_assets=float(row['总资产收益率(%)']) / 100 if not pd.isna(row['总资产收益率(%)']) else None,
                
                # 盈利能力指标
                gross_margin=float(row['销售毛利率(%)']) / 100 if not pd.isna(row['销售毛利率(%)']) else None,
                operating_margin=float(row['营业利润率(%)']) / 100 if not pd.isna(row['营业利润率(%)']) else None,
                net_margin=float(row['净利率(%)']) / 100 if not pd.isna(row['净利率(%)']) else None,
                
                # 流动性指标
                current_ratio=float(row['流动比率']) if not pd.isna(row['流动比率']) else None,
                quick_ratio=float(row['速动比率']) if not pd.isna(row['速动比率']) else None,
                
                # 负债指标
                debt_to_equity=float(row['资产负债率(%)']) / 100 if not pd.isna(row['资产负债率(%)']) else None,
                
                # 每股指标
                earnings_per_share=float(row['每股收益']) if not pd.isna(row['每股收益']) else None,
                book_value_per_share=float(row['每股净资产']) if not pd.isna(row['每股净资产']) else None,
                
                # 增长指标
                revenue_growth=float(row['营业总收入同比增长(%)']) / 100 if not pd.isna(row['营业总收入同比增长(%)']) else None,
                earnings_growth=float(row['净利润同比增长(%)']) / 100 if not pd.isna(row['净利润同比增长(%)']) else None,
                
                # 其他指标填充为None
                enterprise_value=None,
                price_to_earnings_ratio=None,
                price_to_book_ratio=None,
                price_to_sales_ratio=None,
                enterprise_value_to_ebitda_ratio=None,
                enterprise_value_to_revenue_ratio=None,
                free_cash_flow_yield=None,
                peg_ratio=None,
                return_on_invested_capital=None,
                asset_turnover=None,
                inventory_turnover=None,
                receivables_turnover=None,
                days_sales_outstanding=None,
                operating_cycle=None,
                working_capital_turnover=None,
                cash_ratio=None,
                operating_cash_flow_ratio=None,
                debt_to_assets=None,
                interest_coverage=None,
                book_value_growth=None,
                earnings_per_share_growth=None,
                free_cash_flow_growth=None,
                operating_income_growth=None,
                ebitda_growth=None,
                payout_ratio=None,
                free_cash_flow_per_share=None,
            )
            metrics_list.append(metrics)
            
            # 限制数量
            if len(metrics_list) >= limit:
                break
                
        # 缓存结果
        _cache.set_financial_metrics(f"A_{ticker}", [m.model_dump() for m in metrics_list])
        return metrics_list
        
    except Exception as e:
        print(f"获取A股财务指标出错 ({ticker}): {e}")
        return []

def search_a_stock_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "年报",
    limit: int = 10,
) -> list[LineItem]:
    """获取A股特定财务项目"""
    # 转换为akshare格式
    ak_ticker = convert_ticker_to_akshare_format(ticker)
    stock_code = ak_ticker[2:]  # 去掉市场前缀
    
    try:
        # 获取资产负债表
        balance_sheet = ak.stock_financial_report_sina(stock=stock_code, symbol="资产负债表")
        # 获取利润表
        income_stmt = ak.stock_financial_report_sina(stock=stock_code, symbol="利润表")
        # 获取现金流量表
        cash_flow = ak.stock_financial_report_sina(stock=stock_code, symbol="现金流量表")
        
        # 合并数据
        all_data = {}
        for df, statement_type in zip([balance_sheet, income_stmt, cash_flow], 
                                      ["balance_sheet", "income_stmt", "cash_flow"]):
            if df is not None and not df.empty:
                # 转置DataFrame，使得指标名称成为列
                df_t = df.set_index('报表日期').T
                # 重命名索引为更规范的名称
                df_t.index.name = 'item_name'
                # 重置索引使item_name成为列
                df_t = df_t.reset_index()
                
                # 遍历每一行，将数据添加到all_data中
                for _, row in df_t.iterrows():
                    item_name = row['item_name']
                    for col in df_t.columns:
                        if col != 'item_name':
                            date_str = col
                            if date_str <= end_date:
                                if date_str not in all_data:
                                    all_data[date_str] = {}
                                all_data[date_str][f"{statement_type}_{item_name}"] = row[col]
        
        # 转换为LineItem对象列表
        result_items = []
        
        # 定义映射关系
        line_item_mapping = {
            "capital_expenditure": "cash_flow_购建固定资产、无形资产和其他长期资产支付的现金",
            "depreciation_and_amortization": "income_stmt_折旧与摊销",
            "net_income": "income_stmt_净利润",
            "outstanding_shares": "balance_sheet_实收资本（或股本）",
            "total_assets": "balance_sheet_资产总计",
            "total_liabilities": "balance_sheet_负债合计",
            "dividends_and_other_cash_distributions": "cash_flow_分配股利、利润或偿付利息支付的现金",
            "issuance_or_purchase_of_equity_shares": "cash_flow_吸收投资收到的现金"
        }
        
        # 处理每个日期的数据
        for date_str in sorted(all_data.keys(), reverse=True)[:limit]:
            date_data = all_data[date_str]
            
            # 创建基本LineItem对象
            line_item = LineItem(
                ticker=ticker,
                report_period=date_str,
                period=period,
                currency="CNY"
            )
            
            # 添加请求的具体项目
            for item in line_items:
                if item in line_item_mapping and line_item_mapping[item] in date_data:
                    value = date_data[line_item_mapping[item]]
                    # 尝试转换为数值
                    try:
                        if isinstance(value, str):
                            value = float(value.replace(',', ''))
                        else:
                            value = float(value)
                        setattr(line_item, item, value)
                    except (ValueError, TypeError):
                        setattr(line_item, item, None)
                else:
                    setattr(line_item, item, None)
                    
            result_items.append(line_item)
            
        return result_items
        
    except Exception as e:
        print(f"获取A股财务项目出错 ({ticker}): {e}")
        return []

def get_a_stock_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """获取A股市值数据"""
    # 转换为akshare格式
    ak_ticker = convert_ticker_to_akshare_format(ticker)
    
    try:
        # 获取当前行情数据
        stock_data = ak.stock_zh_a_spot()
        
        # 查找对应的股票
        ticker_row = stock_data[stock_data['代码'] == ak_ticker[2:]]
        if ticker_row.empty:
            # 如果找不到，尝试获取历史数据
            hist_data = get_a_stock_prices(ticker, end_date, end_date)
            if hist_data:
                # 使用收盘价和流通股本估算市值
                stock_info = ak.stock_individual_info_em(symbol=ak_ticker[2:])
                circulating_shares = None
                for _, row in stock_info.iterrows():
                    if row['item'] == '流通股本(亿)':
                        circulating_shares = float(row['value']) * 100000000
                        break
                
                if circulating_shares:
                    return hist_data[0].close * circulating_shares
                
            return None
            
        # 获取市值（以亿为单位，转换为具体数值）
        if '总市值' in ticker_row.columns:
            return float(ticker_row['总市值'].iloc[0]) * 100000000  # 转换为具体数值
        
        return None
        
    except Exception as e:
        print(f"获取A股市值数据出错 ({ticker}): {e}")
        return None

def get_a_stock_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 20,
) -> list[CompanyNews]:
    """获取A股相关新闻"""
    # 检查缓存
    if cached_data := _cache.get_company_news(f"A_{ticker}"):
        # 按日期范围过滤
        filtered_data = [CompanyNews(**news) for news in cached_data 
                        if (start_date is None or news["date"] >= start_date)
                        and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data
            
    # 转换为标准股票代码（去掉市场前缀）
    stock_code = convert_ticker_to_akshare_format(ticker)[2:]
    
    try:
        # 获取股票新闻
        news_data = ak.stock_news_em(symbol=stock_code)
        
        # 检查是否有数据
        if news_data.empty:
            return []
            
        # 转换为CompanyNews对象列表
        news_list = []
        for _, row in news_data.iterrows():
            date_str = row['日期']
            
            # 检查日期是否在范围内
            if (start_date is None or date_str >= start_date) and date_str <= end_date:
                news = CompanyNews(
                    ticker=ticker,
                    title=row['新闻标题'],
                    author="东方财富网",
                    source="东方财富网",
                    date=date_str,
                    url=row['新闻链接'],
                    sentiment=None  # 东方财富网没有提供情感分析
                )
                news_list.append(news)
                
                # 限制数量
                if len(news_list) >= limit:
                    break
                    
        # 缓存结果
        _cache.set_company_news(f"A_{ticker}", [n.model_dump() for n in news_list])
        return news_list
        
    except Exception as e:
        print(f"获取A股新闻出错 ({ticker}): {e}")
        return []

def a_stock_prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """将价格对象列表转换为DataFrame"""
    if not prices:
        return pd.DataFrame()
        
    data = []
    for p in prices:
        data.append({
            'date': p.time,
            'open': p.open,
            'high': p.high,
            'low': p.low,
            'close': p.close,
            'volume': p.volume
        })
        
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df

def get_a_stock_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取A股价格数据并转换为DataFrame"""
    prices = get_a_stock_prices(ticker, start_date, end_date)
    return a_stock_prices_to_df(prices) 