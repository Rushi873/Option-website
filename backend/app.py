import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel
import threading
import time
import logging
from database import get_db_connection
from fetch_data import update_option_data
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
#from nsepython import *
from jugaad_data.nse import NSELive
from datetime import datetime, date
from typing import List, Dict
#import QuantLib as ql
import mibian
import os
import base64
import io
from dotenv import load_dotenv
import yfinance as yf
import google.generativeai as genai
import opstrat
import requests
import numpy as np
from scipy.stats import norm
import math
from bs4 import BeautifulSoup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

load_dotenv()
app = FastAPI()
n = NSELive()

# ✅ Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#FOR LOCAL HOSTING
# ✅ Enable CORS for frontend (Port 80)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:80", "https://option-strategy-vaqz.onrender.com", "https://option-strategy.onrender.com"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Change from here to 
# # ✅ Serve static files
# app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")

# # ✅ CORS setup — allow frontend to call backend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # You can narrow this to your frontend domain later
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# and finish





# ✅ Track Selected Asset, # Global cache
selected_asset = None
OPTION_CACHE = {}

def get_cached_option(asset):
    now = time.time()
    if asset in OPTION_CACHE:
        option, timestamp = OPTION_CACHE[asset]
        if now - timestamp < 3:  
            return option

    # Else fetch fresh and cache
    if asset in ["NIFTY", "BANKNIFTY"]:
        option = n.index_option_chain(asset)
    else:
        option = n.equities_option_chain(asset)

    OPTION_CACHE[asset] = (option, now)
    return option


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

        option = get_cached_option(asset)
        spot_price = option["records"]["underlyingValue"]
        option_data = option["records"]["data"]

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





# ✅ Fetch Available Assets, and remove expiry dates, that are expired
@app.get("/get_assets")
def get_assets():
    today = datetime.today().strftime("%Y-%m-%d")
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        # ✅ Fetch all assets
        cursor.execute("SELECT asset_name FROM option_data.assets")  
        assets = [row["asset_name"] for row in cursor.fetchall()]

        if not assets:
            raise HTTPException(status_code=404, detail="No assets found")

        # ✅ For each asset, clean expired data
        for asset in assets:
            # Step 1: Delete from option_chain for expired expiries
            cursor.execute("""
                DELETE FROM option_data.option_chain 
                WHERE expiry_id IN (
                    SELECT id FROM option_data.expiries 
                    WHERE expiry_date < %s 
                    AND asset_id = (SELECT id FROM option_data.assets WHERE asset_name = %s)
                )
            """, (today, asset))

            # Step 2: Delete expired expiry rows
            cursor.execute("""
                DELETE FROM option_data.expiries 
                WHERE expiry_date < %s 
                AND asset_id = (SELECT id FROM option_data.assets WHERE asset_name = %s)
            """, (today, asset))

        conn.commit()

        return {"assets": assets}
    
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error cleaning data: {str(e)}")

    finally:
        cursor.close()
        conn.close()


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


@app.get("/get_spot_price")
def get_spot_price(asset: str = Query(...)):
    try:
        option = get_cached_option(asset)  # ✅ This checks the cache
        spot_price = option["records"].get("underlyingValue")

        if spot_price is None or spot_price == 0:
            raise HTTPException(status_code=500, detail="Spot price not found.")

        return {"spot_price": round(spot_price, 2)}

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
# ✅ STRATEGY MANAGEMENT (Predefined + User-Created), write the code
# ---------------------------------------------------------------

# ✅ Save a Strategy (User-Defined)
@app.post("/save_strategy")
def save_strategy(strategy: Strategy):
    print('save strategy')
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
    print('get strategy')
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
    print('get strategy position')
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
    print('add')
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
    print('hi')
    return {"message": "Position added successfully"}

# ✅ Remove Position from Strategy
@app.delete("/remove_position")
def remove_position(position_id: int):
    print('remove')
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM option_data.strategy_positions WHERE id = %s", (position_id,))
    conn.commit()
    cursor.close()
    conn.close()
    print('bye')
    return {"message": "Position removed successfully"}

# ---------------------------------------------------------------
# ✅ Position 
# ---------------------------------------------------------------

# In-memory strategy storage
strategy_positions: List[dict] = []

# Define expected data structure
class Position(BaseModel):
    symbol: str       # e.g., "NIFTY"
    strike: float     # e.g., 19200.0
    type: str         # e.g., "CE" or "PE"
    quantity: int     # e.g., number of lots
    price: float      # e.g., 45.5

@app.post("/add_strategy")
def add_strategy(position: Position):
    # Convert Pydantic model to dict and append to in-memory list
    pos_dict = position.dict()
    strategy_positions.append(pos_dict)
    
    print(f"Added position: {pos_dict}")
    print(f"Total positions now: {len(strategy_positions)}")

    return {
        "status": "added",
        "position": pos_dict,
        "total_positions": len(strategy_positions)
    }

@app.get("/get_strategies")
def get_strategies():
    print(f"Returning {len(strategy_positions)} strategy positions")
    return strategy_positions

@app.post("/clear_strategy")
def clear_strategy():
    print(f"Clearing {len(strategy_positions)} strategy positions...")
    strategy_positions.clear()
    return {
        "status": "cleared",
        "total_positions": len(strategy_positions)
    }



# ---------------------------------------------------------------
# ✅ PAYOFF CHART & GREEKS
# ---------------------------------------------------------------

# ✅ Define Strategy Model
class StrategyItem(BaseModel):
    option_type: str  # "CE" or "PE"
    strike_price: float
    tr_type: str  # "b" or "s"
    option_price: float
    expiry_date: str

class PayoffRequest(BaseModel):
    asset: str
    strategy: List[Dict[str, str]]


# ✅ Fetch lot size from the database
def get_lot_size(asset_name):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT lot_size FROM option_data.assets WHERE asset_name=%s", (asset_name,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
        else:
            print(f"Error: No data found for asset_name: {asset_name}")
            return None
    except Exception as e:
        print(f"Error in get_lot_size: {e}")
        return None

def extract_iv(asset_name, strike_price, expiry_date, option_type):
    """
    Extract implied volatility from NSE option chain data.

    Params:
        data (dict): result from n.equities_option_chain() or n.index_option_chain()
        strike_price (float): strike to match
        expiry_date (str): in YYYY-MM-DD format
        option_type (str): 'CE' or 'PE'
    """
    try:
        # Convert expiry_date to the format used in NSE data: '24-Apr-2025'
        target_expiry = datetime.strptime(expiry_date, "%Y-%m-%d").strftime("%d-%b-%Y")
        option = get_cached_option(asset_name)  

        for item in option["records"]["data"]:
            if item["strikePrice"] == strike_price and item["expiryDate"] == target_expiry:
                if option_type.upper() in item:
                    return item[option_type.upper()].get("impliedVolatility")

        print(f"No match found for {strike_price}, {target_expiry}, {option_type}")
        return None
    except Exception as e:
        print(f"Error extracting IV: {e}")
        return None




def calculate_option_taxes(strategy_data, asset):
    stt_short_rate = 0.001           # 0.1%
    stt_exercise_rate = 0.00125      # 0.125%
    stamp_duty_rate = 0.00003        # 0.003%
    sebi_rate = 0.000001             # ₹10 / crore = 0.0001%
    txn_charge_rate = 0.0003503      # NSE: 0.03503%
    gst_rate = 0.18
    brokerage_flat = 20              # ₹20 per order

    total = {
        "stt": 0,
        "stamp_duty": 0,
        "sebi_fee": 0,
        "txn_charges": 0,
        "brokerage": 0,
        "gst": 0
    }

    breakdown = []
    spot_price = get_cached_option(asset)["records"]["underlyingValue"]

    for leg in strategy_data:
        tr_type = leg["tr_type"]
        option_type = leg["op_type"]
        strike = leg["strike"]
        premium = leg.get("op_pr", 0)
        lots = leg.get("lot", 1)
        lot_size = leg.get("lot_size", 1)

        turnover = premium * lots * lot_size
        leg_costs = {}

        # STT
        stt_val = 0
        if tr_type == "s":  # Sell
            stt_val = turnover * stt_short_rate
            note = "0.1% STT on premium (sell)"
        elif tr_type == "b":
            intrinsic = 0
            if option_type == "c" and spot_price > strike:
                intrinsic = (spot_price - strike) * lots * lot_size
            elif option_type == "p" and spot_price < strike:
                intrinsic = (strike - spot_price) * lots * lot_size
            if intrinsic > 0:
                stt_val = intrinsic * stt_exercise_rate
                note = "0.125% STT on intrinsic value (exercised ITM)"
            else:
                note = "No STT (OTM unexercised)"

        # Stamp Duty (only for Buy)
        stamp = turnover * stamp_duty_rate if tr_type == "b" else 0

        # SEBI Fee
        sebi_fee = turnover * sebi_rate

        # Transaction Charges (same for buy/sell)
        txn_fee = turnover * txn_charge_rate

        # Brokerage (₹20 per leg)
        brokerage = brokerage_flat

        # GST on (brokerage + sebi + txn)
        gst = gst_rate * (brokerage + sebi_fee + txn_fee)

        # Accumulate
        total["stt"] += stt_val
        total["stamp_duty"] += stamp
        total["sebi_fee"] += sebi_fee
        total["txn_charges"] += txn_fee
        total["brokerage"] += brokerage
        total["gst"] += gst

        # Breakdown
        leg_costs.update({
            "type": option_type.upper(),
            "strike": strike,
            "lots": lots,
            "lot_size": lot_size,
            "premium": f"{premium:.2f}",
            "turnover": f"{turnover:.2f}",
            "stt": round(stt_val, 2),
            "stamp_duty": round(stamp, 2),
            "sebi_fee": round(sebi_fee, 2),
            "txn_charge": round(txn_fee, 2),
            "brokerage": brokerage,
            "gst": round(gst, 2),
            "note": note
        })
        breakdown.append(leg_costs)

    total_all = sum(total.values())

    return {
        "spot_price_used": spot_price,
        "total_cost": round(total_all, 2),
        "charges": {k: round(v, 2) for k, v in total.items()},
        "breakdown": breakdown
    }





def generate_payoff_chart(strategy_data, asset):
    """ Generate the payoff chart and return it as a base64 image """
    try:
        # Close all existing figures first
        plt.close('all')
        plt.ioff()  # Prevent popups

        # Expand each leg according to the number of lots
        expanded_strategy = []
        for leg in strategy_data:
            lots = leg.get("lot", 1)
            single_leg = leg.copy()
            single_leg.pop("lot")  # Remove 'lot' to match opstrat format
            single_leg.pop("lot_size", None)  # Remove 'lot_size' if present
            expanded_strategy.extend([single_leg] * lots)

        # Plot using opstrat
        option = get_cached_option(asset)
        spot_price = option["records"]["underlyingValue"]
        opstrat.multi_plotter(spot=spot_price, op_list=expanded_strategy)

        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        ax = plt.gca()
        ax.set_title(f"Option Strategy Payoff (Spot: {spot_price})")
        ax.grid(True, linestyle='--', alpha=0.7)

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=100)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode("utf-8")

        plt.close(fig)

        return img_base64
    except Exception as e:
        logging.error(f"Error generating payoff chart: {e}", exc_info=True)
        return None



def calculate_strategy_metrics(strategy_data, asset):
    """
    Calculate P&L metrics for multi-leg Indian option strategies.
    Enhanced with contract cost breakdown and lot size display.

    Returns:
        dict: Strategy P&L metrics.
    """
    option = get_cached_option(asset)
    spot_price = option["records"]["underlyingValue"]
    asset_lot_size = strategy_data[0].get('lot_size', 1)  # Assume same across legs for display

    # Price range for simulation
    lower_bound = max(1, spot_price * 0.1)
    upper_bound = spot_price * 3
    price_range = np.linspace(lower_bound, upper_bound, 2000)

    total_payoff = np.zeros_like(price_range)
    total_option_cost = 0
    breakdowns = []

    for leg in strategy_data:
        op_type = leg['op_type'].lower()
        strike = float(leg['strike'])
        tr_type = leg['tr_type'].lower()
        op_pr = float(leg['op_pr'])
        lot = int(leg.get('lot', 1))
        lot_size = int(leg.get('lot_size', asset_lot_size))
        quantity = lot * lot_size

        leg_cost = op_pr * quantity
        action = "Received" if tr_type == 's' else "Paid"
        sign = "+" if tr_type == 's' else "-"

        # Track total option cost (net premium)
        total_option_cost += leg_cost if tr_type == 's' else -leg_cost

        # Detailed leg breakdown
        breakdowns.append(
            f"{tr_type.upper()} {op_type.upper()} | Strike ₹{strike} | Premium ₹{op_pr} × Lots {lot} × LotSize {lot_size} = ₹{leg_cost:.2f} ({action})"
        )

        # Calculate intrinsic value at expiry
        if op_type == 'c':
            intrinsic = np.maximum(price_range - strike, 0)
        elif op_type == 'p':
            intrinsic = np.maximum(strike - price_range, 0)
        else:
            raise ValueError("op_type must be 'c' or 'p'.")

        # Payoff per leg
        if tr_type == 'b':
            payoff = (intrinsic - op_pr) * quantity
        elif tr_type == 's':
            payoff = (op_pr - intrinsic) * quantity
        else:
            raise ValueError("tr_type must be 'b' or 's'.")

        total_payoff += payoff

    raw_max_profit = np.max(total_payoff)
    raw_max_loss = np.min(total_payoff)

    # Detect infinite-like boundary behavior
    start_slope = total_payoff[1] - total_payoff[0]
    end_slope = total_payoff[-1] - total_payoff[-2]

    is_infinite_profit = end_slope > 1 and total_payoff[-1] > raw_max_profit * 0.95
    is_infinite_loss = start_slope < -1 and total_payoff[0] < raw_max_loss * 0.95

    max_profit = "∞" if is_infinite_profit else round(raw_max_profit, 2)
    if is_infinite_loss:
        max_loss = "-∞"
    elif abs(raw_max_loss) >= abs(total_option_cost) and raw_max_loss < 0:
        max_loss = f"{round(raw_max_loss, 2)} (100% loss)"
    else:
        max_loss = round(raw_max_loss, 2)

    # Breakeven points - Improved logic to detect zero-crossings robustly
    payoff_diff = np.diff(np.sign(total_payoff))
    zero_cross_indices = np.where(payoff_diff != 0)[0]

    # Interpolate the breakeven points between the two points that cross zero
    breakeven_raw = []
    for i in zero_cross_indices:
        x1, x2 = price_range[i], price_range[i + 1]
        y1, y2 = total_payoff[i], total_payoff[i + 1]
        if y2 - y1 == 0:
            continue  # Avoid division by zero
        x_breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
        breakeven_raw.append(x_breakeven)

    # Clean up breakeven points with clustering
    def cluster_points(points, gap=0.75):
        if not points:
            return []
        points.sort()
        clustered = [[points[0]]]
        for p in points[1:]:
            if abs(p - clustered[-1][-1]) <= gap:
                clustered[-1].append(p)
            else:
                clustered.append([p])
        return [round(np.mean(group), 2) for group in clustered]

    breakeven_points = cluster_points(breakeven_raw)

    # Reward to Risk Ratio Handling
    try:
        if isinstance(raw_max_profit, str):
            raw_max_profit = raw_max_profit.replace("₹", "").replace(",", "").strip()
            if raw_max_profit in ["∞", "-∞"]:
                max_profit = raw_max_profit
            else:
                max_profit = float(raw_max_profit)
        else:
            max_profit = raw_max_profit

        if isinstance(raw_max_loss, str):
            raw_max_loss = raw_max_loss.replace("₹", "").replace(",", "").strip()
            if raw_max_loss in ["∞", "-∞"]:
                max_loss = raw_max_loss
            else:
                max_loss = float(raw_max_loss)
        else:
            max_loss = raw_max_loss

        # Reward to Risk Calculation
        if max_profit == "∞" and max_loss not in ["-∞", None]:
            reward_to_risk = "∞"
        elif max_profit in [None, ""] or max_loss in [None, ""]:
            reward_to_risk = "N/A"
        elif max_loss == 0:
            reward_to_risk = "∞"
        else:
            reward_to_risk = round(max_profit / abs(max_loss), 2)

    except:
        reward_to_risk = "N/A"

    return {
        "spot_price": round(spot_price, 2),
        "lot_size": asset_lot_size,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven_points": breakeven_points,
        "reward_to_risk_ratio": reward_to_risk,
        "total_option_price": round(total_option_cost, 2),
        "cost_breakdown": breakdowns
    }



def calculate_option_greeks(strategy_data, asset, interest_rate=6.588):
    """
    Calculate option Greeks using mibian.BS model.
    Each Greek is scaled by 100 (no lot or lot_size used).
    """

    greeks_list = []

    # Live spot
    option_data = get_cached_option(asset)
    spot_price = float(option_data["records"]["underlyingValue"])

    for option in strategy_data:
        try:
            S = spot_price
            K = float(option['strike'])
            T_days = int(option['days_to_expiry'])
            sigma = float(option['iv'])               # in %
            flag = option['op_type'].lower()          # 'c' or 'p'
            tr_type = option['tr_type'].lower()       # 'b' or 's'
            lot = int(option['lot'])
            scale = lot * 100

            # Greeks calculation via mibian
            bs = mibian.BS([S, K, interest_rate, T_days], volatility=sigma)

            if flag == 'c':
                delta = bs.callDelta
                theta = bs.callTheta
                rho = bs.callRho
            elif flag == 'p':
                delta = bs.putDelta
                theta = bs.putTheta
                rho = bs.putRho
            else:
                raise ValueError("Invalid option type (must be 'c' or 'p')")

            gamma = bs.gamma
            vega = bs.vega

            # Flip sign for sell
            if tr_type == 's':
                delta *= -1
                gamma *= -1
                theta *= -1
                vega *= -1
                rho *= -1

            # Multiply each Greek by 100 (Opstra-style scaling)
            greeks_list.append({
                'option': option,
                'delta': round(delta * scale, 2),
                'gamma': round(gamma * scale, 4),
                'theta': round(theta * scale, 2),   # Daily decay
                'vega': round(vega * scale, 2),
                'rho': round(rho * scale, 2)
            })

        except Exception as e:
            logging.warning(f"Skipping Greek calc due to error: {e}")
            continue

    return greeks_list





def sanitize_strategy_item(item):
    try:
        required_keys = ['option_type', 'strike_price', 'tr_type', 'option_price', 'expiry_date']
        if not all(key in item for key in required_keys):
            print("Missing required strategy keys.")
            return None

        option_type = item['option_type'].strip().upper()
        if option_type not in ['CE', 'PE']:
            print("Invalid option_type. Must be 'CE' or 'PE'.")
            return None

        tr_type = item['tr_type'].strip().lower()
        if tr_type not in ['b', 's']:
            print("Invalid tr_type. Must be 'b' or 's'.")
            return None

        strike_price = float(item['strike_price'])
        option_price = float(item['option_price'])

        # Convert expiry_date string to datetime.date
        expiry_date = datetime.strptime(item['expiry_date'], "%Y-%m-%d").date()

        # Optional lot size, defaults to 1
        lots = int(item["lots"])

        return {
            "option_type": option_type,
            "strike_price": strike_price,
            "tr_type": tr_type,
            "option_price": option_price,
            "expiry_date": expiry_date,
            "lots": lots
        }

    except Exception as e:
        print(f"Error sanitizing strategy item: {e}")
        return None


def sanitize_json_floats(data):
    if isinstance(data, dict):
        return {k: sanitize_json_floats(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json_floats(i) for i in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return 0.0
        return float(data)
    else:
        return data





@app.post("/get_payoff_chart")
async def get_payoff_chart(request: PayoffRequest):
    try:
        print(f"--- Received request for asset: {request.asset} ---")
        print(f"--- Raw strategy data: {request.strategy} ---")

        lot_size = get_lot_size(request.asset)
        option = get_cached_option(request.asset)
        spot_price = option["records"]["underlyingValue"]
        print(f"Calculated spot price: {spot_price}")
        
        strategy_data = []
        strikes = []

        for item in request.strategy:
            print(f"Processing leg: {item}")
            sanitized = sanitize_strategy_item(item)
            if not sanitized:
                print("Skipping leg due to sanitization failure.")
                continue
        
            strike = sanitized["strike_price"]
            option_type = sanitized["option_type"]
            tr_type = sanitized["tr_type"]
            op_pr = sanitized["option_price"]
            expiry_date = sanitized["expiry_date"]
            lots = sanitized["lots"]
        
            days_to_expiry = (expiry_date - datetime.now().date()).days
        
            iv = extract_iv(request.asset, strike, expiry_date.strftime("%Y-%m-%d"), option_type)
            print(f"IV for strike {strike}, type {option_type}: {iv}")
            if iv is None:
                print("Skipping leg due to missing IV.")
                continue
        
            strategy_leg = {
                "op_type": "p" if option_type.upper() == "PE" else "c",
                "strike": strike,
                "tr_type": tr_type,
                "op_pr": op_pr,
                "lot": abs(lots),
                "lot_size": lot_size,
                "iv": max(iv, 1e-6),
                "days_to_expiry": days_to_expiry
            }
        
            strategy_data.append(strategy_leg)
            strikes.append(strike)

        print(f"--- Final strategy data used for processing: {strategy_data} ---")

        if not strategy_data:
            print("No valid strategy legs with IV found.")
            return JSONResponse(status_code=400, content={"detail": "No valid strategy legs with IV found."})

        img_base64 = generate_payoff_chart(strategy_data, request.asset)
        if img_base64:
            print("Payoff chart generated successfully.")

            # Pass spot_price to all functions that may need it
            metrics = calculate_strategy_metrics(strategy_data, asset=request.asset)
            print(f"Calculated metrics: {metrics}")
            
            tax = calculate_option_taxes(strategy_data, asset=request.asset)
            print(f"Calculated taxes: {tax}")
            
            greeks = calculate_option_greeks(strategy_data, asset=request.asset, interest_rate = 0.06588)
            print(f"Calculated greeks: {greeks}")
            
            response_data = {
                "image": img_base64,
                "success": True,
                "metrics": metrics,
                "tax": tax,
                "greeks": greeks
            }

            return sanitize_json_floats(response_data)

        else:
            print("Error generating payoff chart.")
            return JSONResponse(status_code=500, content={"detail": "Error generating chart."})

    except Exception as e:
        print(f"Unhandled error in get_payoff_chart: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})













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


        
def build_stock_analysis_prompt(stock_symbol: str, stock_data: dict, latest_news: list) -> str:
    news_block = "".join([
        f"- **Headline:** {news['headline']}\n"
        f"  - **Summary:** {news['summary']}\n"
        f"  - **Sentiment:** Assess whether this news is **positive, neutral, or negative**.\n\n"
        for news in latest_news[:2]
    ])

    return f"""
Provide a detailed stock analysis for **{stock_symbol}**, organized in the following sections:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **1️⃣ Technical Analysis**

- **Current Price:** ₹{stock_data["current_price"]}
- **Trading Volume:** {stock_data["volume"]}
- **50-day & 200-day Moving Averages:** ₹{stock_data["moving_avg_50"]:.2f} | ₹{stock_data["moving_avg_200"]:.2f}
- **Trend Analysis:** Indicate whether there is a **Golden Cross** (bullish) or **Death Cross** (bearish) using the moving averages.
- **Support & Resistance:** Identify the **nearest support and resistance** levels using historical price action.
- **Momentum:** Evaluate the strength of **buying or selling pressure** using volume and price trends.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📉 **2️⃣ Fundamental Analysis**

- **Market Capitalization:** ₹{stock_data["market_cap"]}
- **P/E Ratio:** {stock_data["pe_ratio"]} (Compare with sector average)
- **Earnings Per Share (EPS):** ₹{stock_data["eps"]}
- **Valuation:** Is the stock **undervalued, fairly valued, or overvalued** based on industry comparisons?
- **Revenue & Profit Trends:** Are revenue and profit **growing, flat, or declining**?
- **Financial Health:** Evaluate **debt, cash flow, and solvency**.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📰 **3️⃣ Market Sentiment & News**

{news_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 **4️⃣ Market Outlook & Strategy**

- **Short-Term Trend (1–3 months):** Based on charts and sentiment, where is price likely headed?
- **Medium-Term View (3–12 months):** Should investors **buy, hold, or sell**?
- **Key Risks:** Mention external or internal risks to the company.
- **Suggested Action:** Recommend whether to **buy on dips**, **wait for breakout**, or **exit**.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write like a professional research analyst. Be specific, avoid generic advice, and structure your analysis in clear paragraphs.
"""

    


# API Endpoint to get stock analysis
@app.post("/get_stock_analysis")
def get_stock_analysis(request: StockRequest):
    stock_symbol = "^NSEI" if request.asset.upper() == "NIFTY" else f"{request.asset.upper()}.NS"

    stock_data = fetch_stock_data(stock_symbol)
    if not stock_data:
        raise HTTPException(status_code=404, detail="Stock data not found")

    latest_news = fetch_latest_news(request.asset)
    prompt = build_stock_analysis_prompt(stock_symbol, stock_data, latest_news)

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

# FOR LOCAL HOSTING
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
    
# LIVE 
# # ✅ Entry point
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)

