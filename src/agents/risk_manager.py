from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from tools.api import get_prices, prices_to_df
import json
import re

# 添加A股API导入
try:
    from tools.akshare_api import get_a_stock_prices, a_stock_prices_to_df
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Controls position sizing based on real-world risk factors for multiple tickers."""
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    # Initialize risk analysis for each ticker
    risk_analysis = {}
    current_prices = {}  # Store prices here to avoid redundant API calls

    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Analyzing price data")

        # 检查是否为A股代码
        is_a_stock = is_chinese_stock(ticker)
        
        if is_a_stock and AKSHARE_AVAILABLE:
            # 使用A股数据API
            prices = get_a_stock_prices(
                ticker=ticker,
                start_date=data["start_date"],
                end_date=data["end_date"],
            )
            if prices:
                prices_df = a_stock_prices_to_df(prices)
            else:
                prices_df = None
        else:
            # 使用美股数据API
            prices = get_prices(
                ticker=ticker,
                start_date=data["start_date"],
                end_date=data["end_date"],
            )
            if prices:
                prices_df = prices_to_df(prices)
            else:
                prices_df = None

        if not prices_df or prices_df.empty:
            progress.update_status("risk_management_agent", ticker, "Failed: No price data found")
            risk_analysis[ticker] = {
                "remaining_position_limit": 0.0,
                "current_price": 0.0,
                "reasoning": {
                    "error": "No price data available for this ticker"
                },
            }
            continue

        progress.update_status("risk_management_agent", ticker, "Calculating position limits")

        # Calculate portfolio value
        current_price = prices_df["close"].iloc[-1]
        current_prices[ticker] = current_price  # Store the current price

        # Calculate current position value for this ticker
        current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0)

        # Calculate total portfolio value using stored prices
        total_portfolio_value = portfolio.get("cash", 0) + sum(portfolio.get("cost_basis", {}).get(t, 0) for t in portfolio.get("cost_basis", {}))

        # Base limit is 20% of portfolio for any single position
        position_limit = total_portfolio_value * 0.20

        # For existing positions, subtract current position value from limit
        remaining_position_limit = position_limit - current_position_value

        # Ensure we don't exceed available cash
        max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))

        risk_analysis[ticker] = {
            "remaining_position_limit": float(max_position_size),
            "current_price": float(current_price),
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position": float(current_position_value),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
            },
        }

        progress.update_status("risk_management_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name="risk_management_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }

def is_chinese_stock(ticker):
    """
    判断是否为中国股票代码
    支持以下格式：
    1. 以sh或sz开头: sh600000, sz000001
    2. 6位数字开头为沪市，0或3开头为深市: 600000, 000001, 300059
    3. 带后缀的代码: 600000.SH, 000001.SZ
    """
    # 检查是否以sh或sz开头
    if ticker.startswith(('sh', 'sz', 'bj')):
        return True
    
    # 检查是否为纯数字代码
    if re.match(r'^[0-9]{6}$', ticker):
        return True
    
    # 检查是否为带后缀的代码
    if re.match(r'^[0-9]{6}\.(SH|SZ|BJ)$', ticker, re.IGNORECASE):
        return True
    
    return False
