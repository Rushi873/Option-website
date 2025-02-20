from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel
import threading
import time
import logging
from database import get_db_connection
from fetch_data import update_option_data
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from nsepython import option_chain, nse_fno 
from datetime import datetime
from typing import List
import openai
import os
from dotenv import load_dotenv
import yfinance as yf
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup

load_dotenv()
app = FastAPI()

# ✅ Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Enable CORS for frontend (Port 8080)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Track Selected Asset
selected_asset = None


@app.post("/update_selected_asset")
def update_selected_asset(asset: str):
    global selected_asset
    selected_asset = asset
    logger.info(f"User selected asset: {asset}")

    # ✅ Fetch new option chain data for selected asset and store it in MySQL
    update_asset_data(asset)

    return {"message": f"Selected asset updated to {asset}"}


def update_asset_data(asset):
    try:
        logger.info(f"Fetching new data for {asset}...")

        # ✅ Get Spot Price from NSEPython
        spot_price = option_chain(asset)["records"]["underlyingValue"]

        # ✅ Get Option Chain Data
        option_data = option_chain(asset)["records"]["data"]

        conn = get_db_connection()
        cursor = conn.cursor()

        # ✅ Delete old data for this asset before inserting new
        cursor.execute("DELETE FROM option_data.option_chain WHERE asset_id = (SELECT id FROM option_data.assets WHERE asset_name = %s)", (asset,))
        cursor.execute("DELETE FROM option_data.expiries WHERE asset_id = (SELECT id FROM option_data.assets WHERE asset_name = %s)", (asset,))

        # ✅ Insert Expiry Dates (Convert to YYYY-MM-DD format)
        expiry_dates = set()
        for item in option_data:
            raw_expiry = item["expiryDate"]
            formatted_expiry = datetime.strptime(raw_expiry, "%d-%b-%Y").strftime("%Y-%m-%d")
            expiry_dates.add(formatted_expiry)

        for expiry in expiry_dates:
            cursor.execute("""
                INSERT INTO option_data.expiries (asset_id, expiry_date)
                VALUES ((SELECT id FROM option_data.assets WHERE asset_name = %s), %s)
            """, (asset, expiry))

        # ✅ Insert Option Chain Data (Ensure all required fields have values)
        for option in option_data:
            strike_price = option["strikePrice"]
            raw_expiry = option["expiryDate"]
            expiry_date = datetime.strptime(raw_expiry, "%d-%b-%Y").strftime("%Y-%m-%d")

            for option_type in ["CE", "PE"]:
                if option_type in option:
                    opt = option[option_type]

                    # ✅ Ensure all required fields have values (defaults if missing)
                    identifier = opt.get("identifier", f"{asset}_{expiry_date}_{strike_price}_{option_type}")
                    open_interest = opt.get("openInterest", 0)
                    change_in_oi = opt.get("changeinOpenInterest", 0)
                    total_traded_volume = opt.get("totalTradedVolume", 0)
                    implied_volatility = opt.get("impliedVolatility", 0)
                    last_price = opt.get("lastPrice", 0)
                    bid_qty = opt.get("bidQty", 0)
                    bid_price = opt.get("bidprice", 0)
                    ask_qty = opt.get("askQty", 0)
                    ask_price = opt.get("askPrice", 0)
                    total_buy_qty = opt.get("totalBuyQuantity", 0)
                    total_sell_qty = opt.get("totalSellQuantity", 0)

                    cursor.execute("""
                        INSERT INTO option_data.option_chain 
                        (asset_id, expiry_id, strike_price, option_type, identifier, open_interest, change_in_oi, total_traded_volume, 
                        implied_volatility, last_price, bid_qty, bid_price, ask_qty, ask_price, total_buy_qty, total_sell_qty) 
                        VALUES (
                            (SELECT id FROM option_data.assets WHERE asset_name = %s),
                            (SELECT id FROM option_data.expiries WHERE asset_id = (SELECT id FROM option_data.assets WHERE asset_name = %s) AND expiry_date = %s),
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """, (asset, asset, expiry_date, strike_price, option_type, identifier, 
                          open_interest, change_in_oi, total_traded_volume, implied_volatility, last_price, 
                          bid_qty, bid_price, ask_qty, ask_price, total_buy_qty, total_sell_qty))

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"✅ Updated MySQL data for asset: {asset}")

    except Exception as e:
        logger.error(f"❌ Error fetching and storing data for {asset}: {str(e)}")





# ✅ Fetch Available Assets
@app.get("/get_assets")
def get_assets():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT asset_name FROM option_data.assets")  
    assets = [row["asset_name"] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    if not assets:
        raise HTTPException(status_code=404, detail="No assets found")
    return {"assets": assets}

# ✅ Fetch Unique Expiry Dates (from MySQL)
@app.get("/expiry_dates")
def get_expiry_dates(asset: str = Query(...)):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT DISTINCT expiry_date  
        FROM option_data.expiries 
        WHERE asset_id = (SELECT id FROM option_data.assets WHERE asset_name = %s)
        ORDER BY expiry_date ASC
    """, (asset,))

    expiries = list(set(row["expiry_date"] for row in cursor.fetchall()))  # Ensure uniqueness
    
    cursor.close()
    conn.close()

    if not expiries:
        raise HTTPException(status_code=404, detail="No expiry dates found")
    
    return {"expiry_dates": sorted(expiries)}  # Ensure they remain sorted


# ✅ Fetch Option Chain (from MySQL)
@app.get("/get_option_chain")
def get_option_chain(asset: str = Query(...), expiry: str = Query(...)):
    update_asset_data(asset)
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT 
            oc.strike_price,
            oc.option_type,
            oc.open_interest,
            oc.change_in_oi,
            oc.implied_volatility,
            oc.last_price
        FROM option_data.option_chain AS oc
        JOIN option_data.assets AS a ON oc.asset_id = a.id
        JOIN option_data.expiries AS e ON oc.expiry_id = e.id
        WHERE a.asset_name = %s AND e.expiry_date = %s
        ORDER BY oc.strike_price ASC
    """, (asset, expiry))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No option chain data found.")

    option_chain = {}
    for row in rows:
        strike = row["strike_price"]
        if strike not in option_chain:
            option_chain[strike] = {"put": {}, "call": {}}
        if row["option_type"] == "PE":
            option_chain[strike]["put"] = {
                "open_interest": row["open_interest"],
                "change_in_oi": row["change_in_oi"],
                "implied_volatility": row["implied_volatility"],
                "last_price": row["last_price"]
            }
        else:
            option_chain[strike]["call"] = {
                "open_interest": row["open_interest"],
                "change_in_oi": row["change_in_oi"],
                "implied_volatility": row["implied_volatility"],
                "last_price": row["last_price"]
            }
    return {"option_chain": option_chain}


# ✅ Fetch Spot Price
@app.get("/get_spot_price")
def get_spot_price(asset: str = Query(...)):
    try:
        spot_price = option_chain(asset)["records"]["underlyingValue"]
        if not spot_price:
            raise HTTPException(status_code=500, detail="Spot price not found.")
        return {"spot_price": spot_price}
    except Exception as e:
        logger.error(f"Error fetching spot price for {asset}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching spot price: {str(e)}")


# ✅ Strategy & Position Models
class Strategy(BaseModel):
    strategy_name: str
    user_id: int  # None for predefined strategies

class StrategyPosition(BaseModel):
    strategy_id: int
    strike_price: float
    expiry_date: str
    option_type: str  # "CE" or "PE"
    lots: int
    entry_price: float

# ---------------------------------------------------------------
# ✅ STRATEGY MANAGEMENT (Predefined + User-Created)
# ---------------------------------------------------------------

# ✅ Save a Strategy (User-Defined)
@app.post("/save_strategy")
def save_strategy(strategy: Strategy):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO option_data.strategies (user_id, strategy_name) 
        VALUES (%s, %s)
    """, (strategy.user_id, strategy.strategy_name))
    
    conn.commit()
    cursor.close()
    conn.close()
    return {"message": "Strategy saved successfully"}

# ✅ Fetch All Strategies (User-Created + Predefined)
@app.get("/get_strategies")
def get_strategies(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, strategy_name 
        FROM option_data.strategies
        WHERE user_id = %s OR user_id IS NULL
    """, (user_id,))
    
    strategies = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return {"strategies": strategies}

# ✅ Fetch Positions for a Strategy
@app.get("/get_strategy_positions")
def get_strategy_positions(strategy_id: int, expiry_date: str = None):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, strike_price, option_type, lots, entry_price, expiry_date
        FROM option_data.strategy_positions
        WHERE strategy_id = %s
    """, (strategy_id,))
    
    positions = cursor.fetchall()

    # For predefined strategies, dynamically assign expiry
    for pos in positions:
        if pos["expiry_date"] is None and expiry_date:
            pos["expiry_date"] = expiry_date

    cursor.close()
    conn.close()
    
    return {"positions": positions}

# ✅ Add Position to Strategy
@app.post("/add_position")
def add_position(position: StrategyPosition):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO option_data.strategy_positions (strategy_id, strike_price, expiry_date, option_type, lots, entry_price) 
        VALUES (%s, %s, %s, %s, %s, %s)
    """, 
    (position.strategy_id, position.strike_price, position.expiry_date, position.option_type, position.lots, position.entry_price))

    conn.commit()
    cursor.close()
    conn.close()
    return {"message": "Position added successfully"}

# ✅ Remove Position from Strategy
@app.delete("/remove_position")
def remove_position(position_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM option_data.strategy_positions WHERE id = %s", (position_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return {"message": "Position removed successfully"}

# ---------------------------------------------------------------
# ✅ PAYOFF CHART & GREEKS
# ---------------------------------------------------------------


LOT_SIZE = 50  # Assume each lot contains 50 contracts (modify as needed)

# ✅ Function to Calculate Option P&L
def calculate_option_pnl(position: StrategyPosition, price: float) -> float:
    """
    Calculates the payoff for each option position considering:
    - Call (CE) & Put (PE) options.
    - Long (Buy) & Short (Sell) positions.
    - Lot size to correctly scale P&L.
    """
    # Determine intrinsic value based on option type
    if position.option_type == "CE":  # Call Option
        intrinsic_value = max(price - position.strike_price, 0)
    elif position.option_type == "PE":  # Put Option
        intrinsic_value = max(position.strike_price - price, 0)
    else:
        return 0  # Invalid option type

    # Calculate P&L with lot size multiplication
    pnl = (intrinsic_value - position.entry_price) * (position.lots * LOT_SIZE)

    return pnl

# ✅ Payoff Calculation API
@app.post("/calculate_payoff")
def calculate_payoff(positions: List[StrategyPosition], spot_price: float):
    """
    Generates a payoff chart for the given strategy positions.
    - Evaluates across a ±10% price range.
    - Aggregates P&L across multiple positions.
    """
    price_range = np.linspace(spot_price * 0.9, spot_price * 1.1, 50)  # ±10% range, 50 steps
    payoff_data = []

    for price in price_range:
        total_pnl = sum(calculate_option_pnl(pos, price) for pos in positions)
        payoff_data.append({"underlying_price": round(price, 2), "pnl": round(total_pnl, 2)})

    return {"payoff_chart": payoff_data}




# ---------------------------------------------------------------
# ✅ LLM
# ---------------------------------------------------------------

# Configure Gemini API
genai.configure(api_key="AIzaSyDd_UVZ_1OeLahVrJ0A-hbazQcr1FOpgPE")

# Define request model
class StockRequest(BaseModel):
    asset: str  # Example: "ITC"

# Fetch stock data from Yahoo Finance
def fetch_stock_data(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        stock_data = stock.history(period="1d")

        if stock_data.empty:
            return None

        current_price = stock_data["Close"].iloc[-1]
        volume = stock_data["Volume"].iloc[-1]
        moving_avg_50 = stock.history(period="50d")["Close"].mean()
        moving_avg_200 = stock.history(period="200d")["Close"].mean()

        info = stock.info
        market_cap = info.get("marketCap", "N/A")
        pe_ratio = info.get("trailingPE", "N/A")
        eps = info.get("trailingEps", "N/A")

        return {
            "current_price": current_price,
            "volume": volume,
            "moving_avg_50": moving_avg_50,
            "moving_avg_200": moving_avg_200,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "eps": eps
            
        }
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def fetch_latest_news(asset):
    stock_symbol = "^NSEI" if asset.upper() == "NIFTY" else f"{asset.upper()}.NS"
    yahoo_news_url = f"https://finance.yahoo.com/quote/{stock_symbol}/news/"

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(yahoo_news_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        news_section = soup.select_one("#nimbus-app > section > section > section > article > section.mainContent.yf-tnbau3 > section")
        
        news_list = []
        if news_section:
            news_items = news_section.find_all("h3", limit=3)  # Get the first 3 news headlines
            
            for news_item in news_items:
                headline = news_item.text.strip()
                summary_tag = news_item.find_next("p")
                summary = summary_tag.text.strip() if summary_tag else "No summary available."
                news_list.append({"headline": headline, "summary": summary})

        if not news_list:
            news_list.append({"headline": "No major updates found.", "summary": ""})

        return news_list

    except Exception as e:
        print(f"Error fetching news: {e}")
        return [{"headline": "No major updates found.", "summary": ""}]


# API Endpoint to get stock analysis
@app.post("/get_stock_analysis")
def get_stock_analysis(request: StockRequest):
    stock_symbol = "^NSEI" if request.asset.upper() == "NIFTY" else f"{request.asset.upper()}.NS"

    stock_data = fetch_stock_data(stock_symbol)
    if not stock_data:
        raise HTTPException(status_code=404, detail="Stock data not found")

    latest_news = fetch_latest_news(request.asset)

    # ✅ Structured AI Prompt (Forces Detailed Analysis)
    prompt = f"""
    Provide a detailed stock analysis for {stock_symbol}, breaking it into the following sections:

    **1️⃣ Technical Analysis:** 
    - **Current Price:** ₹{stock_data["current_price"]}
    - **Trading Volume:** {stock_data["volume"]}
    - **50-day & 200-day Moving Averages:** ₹{stock_data["moving_avg_50"]:.2f} | ₹{stock_data["moving_avg_200"]:.2f}
    - **Trend Analysis:** Identify if there is a **Golden Cross (bullish)** or **Death Cross (bearish)** based on moving average crossover.
    - **Support & Resistance Levels:** Identify the **nearest support and resistance levels** based on historical price action.
    - **Momentum Analysis:** Determine if the stock has **strong buying or selling pressure** based on volume and trend strength.

    **2️⃣ Fundamental Analysis:**  
    - **Market Capitalization:** ₹{stock_data["market_cap"]}
    - **P/E Ratio:** {stock_data["pe_ratio"]} (Compare to sector average)
    - **Earnings Per Share (EPS):** ₹{stock_data["eps"]}
    - **Valuation Analysis:** Determine whether the stock is **undervalued, fairly valued, or overvalued** based on its P/E ratio and industry comparison.
    - **Revenue & Profit Trends:** Assess whether the company's **revenue and profit** are **growing, stagnating, or declining**.
    - **Debt & Financial Stability:** Analyze the company's **debt levels, cash flow, and overall financial health**.

    **4️⃣ Market Sentiment & News:**  
    - **Latest Headline:** {latest_news[0]["headline"]}
    - **Summary:** {latest_news[0]["summary"]}
    - **Sentiment Analysis:** Determine whether the latest news is **positive, neutral, or negative** and its potential impact on stock price.

    - **Latest Headline:** {latest_news[1]["headline"]}
    - **Summary:** {latest_news[1]["summary"]}
    - **Sentiment Analysis:** Determine whether the latest news is **positive, neutral, or negative** and its potential impact on stock price.


    **5️⃣ Market Outlook & Trading Strategy:**  
    - **Short-Term Trend (1-3 months):** Provide an analysis of expected price movement based on technical indicators and market sentiment.
    - **Medium-Term Trend (3-12 months):** Identify whether the stock is a **buy, hold, or sell** based on valuation, industry trends, and competitor performance.
    - **Risk Factors:** Highlight any risks that could **impact future stock performance**.
    - **Action Plan:** Recommend whether investors should **buy on dips, wait for confirmation, or exit the stock**.

    Provide a professional and insightful analysis rather than generic responses. You should not add a date in the heading.
    """


    # Fetch AI response
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)

    return {"analysis": response.text}



# ---------------------------------------------------------------
# ✅ LLM ends
# ---------------------------------------------------------------



# ✅ Background Live Update (Update Only Selected Asset)
def live_update():
    while True:
        if selected_asset:  
            logger.info(f"Updating data for: {selected_asset}")
            try:
                update_option_data(selected_asset)
            except Exception as e:
                logger.error(f"Error updating {selected_asset}: {str(e)}")
        time.sleep(3)

# ✅ Start background thread
threading.Thread(target=live_update, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

