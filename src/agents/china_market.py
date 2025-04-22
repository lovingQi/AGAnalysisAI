from graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
import pandas as pd
import akshare as ak
from tools.akshare_api import (
    get_a_stock_prices,
    get_a_stock_financial_metrics,
    search_a_stock_line_items,
    get_a_stock_market_cap,
    get_a_stock_news
)
from utils.llm import call_llm
from utils.progress import progress


class ChinaMarketSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def china_market_agent(state: AgentState):
    """针对中国市场特点进行分析的代理"""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # 收集所有分析数据
    analysis_data = {}
    china_market_analysis = {}

    for ticker in tickers:
        progress.update_status("china_market_agent", ticker, "获取A股特有数据")
        
        # 获取北向资金数据
        northbound_capital = get_northbound_capital(ticker, end_date)
        
        # 获取限售股解禁数据
        restricted_shares = get_restricted_shares_info(ticker, end_date)
        
        # 获取公司所属板块和行业数据
        sector_info = get_sector_info(ticker)
        
        # 获取市场情绪指标 - 融资融券数据
        margin_trading = get_margin_trading_data(ticker, end_date)
        
        # 获取政策相关性分析
        policy_impact = analyze_policy_impact(ticker, sector_info)

        progress.update_status("china_market_agent", ticker, "分析A股特有因素")
        
        # 进行综合分析
        china_market_output = generate_china_market_output(
            ticker=ticker,
            northbound_capital=northbound_capital,
            restricted_shares=restricted_shares,
            sector_info=sector_info,
            margin_trading=margin_trading,
            policy_impact=policy_impact,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        # 将分析结果按统一格式存储
        china_market_analysis[ticker] = {
            "signal": china_market_output.signal,
            "confidence": china_market_output.confidence,
            "reasoning": china_market_output.reasoning,
        }

        progress.update_status("china_market_agent", ticker, "完成")

    # 创建消息
    message = HumanMessage(content=json.dumps(china_market_analysis), name="china_market_agent")

    # 如果需要显示推理过程
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(china_market_analysis, "中国市场分析代理")

    # 将信号添加到分析师信号列表
    state["data"]["analyst_signals"]["china_market_agent"] = china_market_analysis

    return {"messages": [message], "data": state["data"]}


def get_northbound_capital(ticker: str, end_date: str) -> dict:
    """获取北向资金持股数据"""
    try:
        # 转换为纯数字代码格式
        if ticker.startswith(('sh', 'sz')):
            code = ticker[2:]
        else:
            code = ticker
            
        # 获取北向资金持股数据
        north_data = ak.stock_hk_ggt_components_em()
        
        # 查找对应的股票
        stock_data = north_data[north_data['代码'] == code]
        if stock_data.empty:
            return {
                "has_northbound_capital": False,
                "details": "未发现北向资金持股信息"
            }
            
        # 提取相关数据
        latest_data = stock_data.iloc[0]
        return {
            "has_northbound_capital": True,
            "holding_ratio": float(latest_data['持股占比'].replace('%', '')) / 100 if '持股占比' in latest_data else None,
            "recent_change": latest_data['持股变动'] if '持股变动' in latest_data else None,
            "details": "北向资金活跃持股"
        }
    except Exception as e:
        print(f"获取北向资金数据错误 ({ticker}): {e}")
        return {
            "has_northbound_capital": False,
            "details": f"获取北向资金数据错误: {e}"
        }


def get_restricted_shares_info(ticker: str, end_date: str) -> dict:
    """获取限售股解禁数据"""
    try:
        # 转换为纯数字代码格式
        if ticker.startswith(('sh', 'sz')):
            code = ticker[2:]
        else:
            code = ticker
            
        # 获取限售股解禁数据
        restricted_data = ak.stock_restricted_shares()
        
        # 过滤当前股票的数据
        stock_data = restricted_data[restricted_data['代码'] == code]
        if stock_data.empty:
            return {
                "has_upcoming_release": False,
                "details": "近期无限售股解禁"
            }
            
        # 按日期排序并获取最近的解禁数据
        stock_data = stock_data.sort_values('解禁日期')
        
        # 检查是否有即将到来的解禁
        upcoming_releases = stock_data[stock_data['解禁日期'] > end_date]
        if upcoming_releases.empty:
            return {
                "has_upcoming_release": False,
                "details": "近期无限售股解禁"
            }
            
        # 获取最近的解禁信息
        next_release = upcoming_releases.iloc[0]
        
        return {
            "has_upcoming_release": True,
            "release_date": next_release['解禁日期'],
            "release_ratio": float(next_release['解禁比例'].replace('%', '')) / 100 if '解禁比例' in next_release else None,
            "release_amount": float(next_release['解禁数量(亿)']) * 100000000 if '解禁数量(亿)' in next_release else None,
            "details": f"即将于{next_release['解禁日期']}解禁，占比{next_release['解禁比例']}"
        }
    except Exception as e:
        print(f"获取限售股解禁数据错误 ({ticker}): {e}")
        return {
            "has_upcoming_release": False,
            "details": f"获取限售股解禁数据错误: {e}"
        }


def get_sector_info(ticker: str) -> dict:
    """获取公司所属板块和行业数据"""
    try:
        # 转换为纯数字代码格式
        if ticker.startswith(('sh', 'sz')):
            code = ticker[2:]
        else:
            code = ticker
            
        # 获取个股信息
        stock_info = ak.stock_individual_info_em(symbol=code)
        
        # 提取行业和板块信息
        sector = None
        industry = None
        board = None
        
        for _, row in stock_info.iterrows():
            if row['item'] == '行业':
                industry = row['value']
            elif row['item'] == '板块':
                board = row['value']
            elif row['item'] == '概念':
                sector = row['value']
                
        # 获取行业整体表现
        try:
            industry_perf = None
            if industry:
                industry_data = ak.stock_sector_detail(sector=industry)
                if not industry_data.empty:
                    avg_change = industry_data['涨跌幅'].astype(float).mean()
                    industry_perf = {
                        "average_change": avg_change,
                        "rank": "strong" if avg_change > 2 else "weak" if avg_change < -2 else "neutral"
                    }
        except:
            industry_perf = None
            
        return {
            "industry": industry,
            "board": board,
            "sector": sector,
            "industry_performance": industry_perf,
            "details": f"行业: {industry}, 板块: {board}, 概念: {sector}"
        }
    except Exception as e:
        print(f"获取板块行业数据错误 ({ticker}): {e}")
        return {
            "industry": None,
            "board": None,
            "sector": None,
            "details": f"获取板块行业数据错误: {e}"
        }


def get_margin_trading_data(ticker: str, end_date: str) -> dict:
    """获取融资融券数据"""
    try:
        # 转换为纯数字代码格式
        if ticker.startswith(('sh', 'sz')):
            code = ticker[2:]
        else:
            code = ticker
            
        # 获取融资融券数据
        margin_data = ak.stock_margin_detail_em(symbol=code)
        
        if margin_data.empty:
            return {
                "has_margin_trading": False,
                "details": "无融资融券数据"
            }
            
        # 按日期排序
        margin_data = margin_data.sort_values('日期', ascending=False)
        
        # 获取最新数据
        latest_data = margin_data.iloc[0]
        
        # 计算融资融券余额变化率
        if len(margin_data) > 1:
            previous_data = margin_data.iloc[1]
            financing_change = (latest_data['融资余额'] - previous_data['融资余额']) / previous_data['融资余额']
            margin_change = (latest_data['融券余额'] - previous_data['融券余额']) / previous_data['融券余额']
        else:
            financing_change = 0
            margin_change = 0
            
        return {
            "has_margin_trading": True,
            "financing_balance": latest_data['融资余额'],
            "margin_balance": latest_data['融券余额'],
            "financing_change": financing_change,
            "margin_change": margin_change,
            "details": f"融资余额: {latest_data['融资余额']}, 融券余额: {latest_data['融券余额']}"
        }
    except Exception as e:
        print(f"获取融资融券数据错误 ({ticker}): {e}")
        return {
            "has_margin_trading": False,
            "details": f"获取融资融券数据错误: {e}"
        }


def analyze_policy_impact(ticker: str, sector_info: dict) -> dict:
    """分析政策对公司的潜在影响"""
    # 根据行业和板块分析政策敏感度
    industry = sector_info.get("industry")
    sector = sector_info.get("sector")
    
    # 高政策敏感度行业
    high_sensitivity_industries = [
        "银行", "证券", "保险", "房地产", "医药生物", "电力", "煤炭", 
        "石油", "天然气", "建筑", "环保", "教育", "互联网"
    ]
    
    # 中等政策敏感度行业
    medium_sensitivity_industries = [
        "汽车", "钢铁", "有色金属", "机械设备", "电子", "通信", 
        "计算机", "农林牧渔", "食品饮料"
    ]
    
    # 判断政策敏感度
    sensitivity = "low"
    if industry in high_sensitivity_industries:
        sensitivity = "high"
    elif industry in medium_sensitivity_industries:
        sensitivity = "medium"
        
    # 获取最新的政策新闻
    try:
        # 获取政策相关新闻
        policy_news = []
        
        # 如果有行业信息，获取行业相关政策
        if industry:
            industry_news = ak.stock_news_em(symbol=industry)
            if not industry_news.empty:
                policy_news.extend(industry_news.iloc[:5]['新闻标题'].tolist())
                
        # 如果政策新闻不足5条，尝试获取板块相关新闻
        if len(policy_news) < 5 and sector:
            sector_news = ak.stock_news_em(symbol=sector)
            if not sector_news.empty:
                policy_news.extend(sector_news.iloc[:5-len(policy_news)]['新闻标题'].tolist())
                
        policy_news = policy_news[:5]  # 最多保留5条
    except:
        policy_news = []
        
    return {
        "policy_sensitivity": sensitivity,
        "recent_policy_news": policy_news,
        "details": f"政策敏感度: {sensitivity}, 相关政策新闻数: {len(policy_news)}"
    }


def generate_china_market_output(
    ticker: str,
    northbound_capital: dict,
    restricted_shares: dict,
    sector_info: dict,
    margin_trading: dict,
    policy_impact: dict,
    model_name: str,
    model_provider: str,
) -> ChinaMarketSignal:
    """根据中国市场特有因素生成分析结果"""
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一位专业的A股分析师，擅长分析中国特色市场因素对股票的影响。针对以下特有的A股因素，请生成投资信号：

                1. 北向资金持股情况：北向资金的持股变动是重要指标，外资增持通常是看好信号，减持则可能是看空
                2. 限售股解禁：大比例的限售股解禁可能导致短期股价承压
                3. 行业与板块表现：行业整体强势往往带动个股上涨
                4. 融资融券数据：融资余额大幅增加通常反映市场看多情绪，融券余额增加则反映看空情绪
                5. 政策敏感度：高政策敏感行业的企业更容易受政策变动影响
                
                请分析这些因素对股票的影响，并给出综合判断：
                - 看涨 (bullish)：代表这些A股特有因素整体看好该股票
                - 看跌 (bearish)：代表这些A股特有因素整体看空该股票
                - 中性 (neutral)：代表这些A股特有因素影响有限或相互抵消
                
                在你的分析中，请重点考虑以下因素：
                1. 北向资金持股变动是重要的市场情绪指标
                2. 大比例限售股解禁通常为利空因素
                3. 行业和板块的整体表现往往会影响个股
                4. 融资融券数据反映市场情绪
                5. 高政策敏感度行业的企业需特别关注政策风险

                请提供详细的分析推理，解释你如何综合这些因素得出最终结论。
                """,
            ),
            (
                "human",
                """请基于以下中国市场特有因素，分析{ticker}股票：

                北向资金数据:
                {northbound_capital}
                
                限售股解禁信息:
                {restricted_shares}
                
                行业和板块数据:
                {sector_info}
                
                融资融券数据:
                {margin_trading}
                
                政策影响分析:
                {policy_impact}

                请以JSON格式返回你的分析结果，格式如下：
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100,
                  "reasoning": "string"
                }}
                """,
            ),
        ]
    )

    prompt = template.invoke({
        "ticker": ticker,
        "northbound_capital": json.dumps(northbound_capital, ensure_ascii=False, indent=2),
        "restricted_shares": json.dumps(restricted_shares, ensure_ascii=False, indent=2),
        "sector_info": json.dumps(sector_info, ensure_ascii=False, indent=2),
        "margin_trading": json.dumps(margin_trading, ensure_ascii=False, indent=2),
        "policy_impact": json.dumps(policy_impact, ensure_ascii=False, indent=2)
    })

    # 默认信号
    def create_default_china_market_signal():
        return ChinaMarketSignal(
            signal="neutral", 
            confidence=0.0, 
            reasoning="分析出错，默认为中性信号"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=ChinaMarketSignal,
        agent_name="china_market_agent",
        default_factory=create_default_china_market_signal,
    ) 