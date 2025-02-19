from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import threading
import time
import logging
from database import get_db_connection
from fetch_data import update_option_data
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from nsepython import option_chain  
from datetime import datetime
from typing import List

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


# ✅ Strategy & Position Management
class Strategy(BaseModel):
    strategy_name: str
    user_id: int

class StrategyPosition(BaseModel):
    strategy_id: int
    strike_price: float
    expiry_date: str
    option_type: str  # "CE" or "PE"
    lots: int

# ✅ Save a Strategy
@app.post("/save_strategy")
def save_strategy(strategy: Strategy):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO option_data.strategies (user_id, strategy_name) VALUES (%s, %s)", 
                   (strategy.user_id, strategy.strategy_name))
    conn.commit()
    cursor.close()
    conn.close()
    return {"message": "Strategy saved successfully"}

# ✅ Fetch User-Saved Strategies
@app.get("/get_saved_strategies")
def get_saved_strategies(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, strategy_name FROM option_data.strategies WHERE user_id = %s", (user_id,))
    strategies = cursor.fetchall()
    cursor.close()
    conn.close()
    return {"saved_strategies": strategies}

# ✅ Fetch Strategy Positions
@app.get("/get_strategy_positions")
def get_strategy_positions(strategy_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM option_data.strategy_positions WHERE strategy_id = %s", (strategy_id,))
    positions = cursor.fetchall()
    cursor.close()
    conn.close()
    return {"strategy_positions": positions}

# ✅ Add Position to Strategy
@app.post("/add_position")
def add_position(position: StrategyPosition):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO option_data.strategy_positions (strategy_id, strike_price, expiry_date, option_type, lots) VALUES (%s, %s, %s, %s, %s)", 
                   (position.strategy_id, position.strike_price, position.expiry_date, position.option_type, position.lots))
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
