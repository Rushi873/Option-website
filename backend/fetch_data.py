from nsepython import option_chain
from .database import get_db_connection
from datetime import datetime
import time

def update_option_data(asset_name):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        print(f"Fetching {asset_name} data...")
        chain = option_chain(asset_name)

        cursor.execute("INSERT INTO assets (asset_name) VALUES (%s) ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id)", (asset_name,))
        asset_id = cursor.lastrowid

        expiry_ids = {}
        for expiry in chain['records']['expiryDates']:
            formatted_date = datetime.strptime(expiry, "%d-%b-%Y").strftime("%Y-%m-%d")
            cursor.execute("INSERT INTO expiries (asset_id, expiry_date) VALUES (%s, %s) ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id)", (asset_id, formatted_date))
            expiry_ids[expiry] = cursor.lastrowid

        for entry in chain['records']['data']:
            strike_price = entry['strikePrice']
            expiry_id = expiry_ids[entry['expiryDate']]

            for option_type in ["CE", "PE"]:
                if option_type in entry:
                    data = entry[option_type]
                    cursor.execute("""
                        INSERT INTO option_chain (asset_id, expiry_id, strike_price, option_type, identifier, open_interest, change_in_oi, 
                                                 total_traded_volume, implied_volatility, last_price, bid_qty, bid_price, ask_qty, ask_price, 
                                                 total_buy_qty, total_sell_qty) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        open_interest = VALUES(open_interest),
                        change_in_oi = VALUES(change_in_oi),
                        total_traded_volume = VALUES(total_traded_volume),
                        implied_volatility = VALUES(implied_volatility),
                        last_price = VALUES(last_price),
                        bid_qty = VALUES(bid_qty),
                        bid_price = VALUES(bid_price),
                        ask_qty = VALUES(ask_qty),
                        ask_price = VALUES(ask_price),
                        total_buy_qty = VALUES(total_buy_qty),
                        total_sell_qty = VALUES(total_sell_qty),
                        last_updated = CURRENT_TIMESTAMP
                    """, (
                        asset_id, expiry_id, strike_price, option_type, data["identifier"],
                        data["openInterest"], data["changeinOpenInterest"], data["totalTradedVolume"],
                        data["impliedVolatility"], data["lastPrice"], data["bidQty"], data["bidprice"],
                        data["askQty"], data["askPrice"], data["totalBuyQuantity"], data["totalSellQuantity"]
                    ))

        conn.commit()
        print(f"Updated {asset_name}")

    except Exception as e:
        conn.rollback()
        print(f"Error updating {asset_name}: {e}")

    finally:
        cursor.close()
        conn.close()

def live_update(selected_assets):
    while True:
        for asset in selected_assets:
            update_option_data(asset)
        time.sleep(3)
