from fastapi import FastAPI, HTTPException, Query
import mysql.connector
import logging

# ✅ Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="your_username",
        password="your_password",
        database="your_database"
    )

@app.get("/get_option_chain")
def get_option_chain(asset: str = Query(..., description="Asset name"), expiry: str = Query(..., description="Expiry date")):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        logger.info(f"Fetching option chain for Asset: {asset}, Expiry: {expiry}")

        # ✅ Log asset_id lookup
        cursor.execute("SELECT id FROM assets WHERE asset_name = %s", (asset,))
        asset_row = cursor.fetchone()
        if not asset_row:
            raise HTTPException(status_code=404, detail=f"Asset '{asset}' not found in database.")

        asset_id = asset_row["id"]
        logger.info(f"Asset ID found: {asset_id}")

        # ✅ Fetch option chain data
        cursor.execute("""
            SELECT strike, put_ltp, put_oi, put_iv, call_iv, call_oi, call_ltp 
            FROM option_chain 
            WHERE asset_id = %s AND expiry_date = %s
            ORDER BY strike ASC
        """, (asset_id, expiry))

        data = cursor.fetchall()

        if not data:
            logger.warning(f"No option chain data found for {asset} ({expiry})")
            raise HTTPException(status_code=404, detail=f"No option chain data found for {asset} ({expiry})")

        return {"option_chain": data}

    except mysql.connector.Error as sql_err:
        logger.error(f"MySQL Error: {sql_err}")
        raise HTTPException(status_code=500, detail=f"MySQL Error: {str(sql_err)}")

    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")

    finally:
        cursor.close()
        conn.close()
