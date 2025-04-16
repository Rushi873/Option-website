# ===============================================================
# Imports (Keep as is)
# ===============================================================
import sys
import os
import asyncio
import time
import logging
import threading
import base64
import io
import json
import functools # Import functools
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Union, Optional
from contextlib2 import asynccontextmanager
from collections import defaultdict

# --- Environment & Config ---
from dotenv import load_dotenv

# --- FastAPI & Web ---
from fastapi import FastAPI, HTTPException, Query, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# --- Pydantic Models ---
from pydantic import BaseModel, Field

# --- Data Sources ---
from jugaad_data.nse import NSELive # For option chain source
import yfinance as yf              # For stock data
import requests                    # For news scraping
from bs4 import BeautifulSoup       # For news scraping

# --- Calculation & Plotting ---
import numpy as np
# Removed Plotly import
import math
import mibian
# *** Ensure Matplotlib Imports are Correct ***
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
# Set some optimized Matplotlib parameters globally (optional)
# plt.style.use('fast') # potentially faster style, less visually complex
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 0.6
plt.rcParams['agg.path.chunksize'] = 10000 # Process paths in chunks
# Import SciPy optimize if needed for breakeven (ensure it's installed)
try:
    from scipy.optimize import brentq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger = logging.getLogger(__name__) # Need logger early for this message
    logger.warning("SciPy not found. Breakeven calculation will use linear interpolation fallback.")


# --- Caching ---
from cachetools import TTLCache

# --- AI/LLM ---
import google.generativeai as genai

# --- Database ---
# Assume database.py exists in the same directory or adjust path
try:
    from .database import initialize_database_pool, get_db_connection
except ImportError:
    # Fallback if running directly for testing, adjust as needed
    from database import initialize_database_pool, get_db_connection

import mysql.connector # Keep for catching specific DB errors if needed


# ===============================================================
# Initial Setup (Keep as is)
# ===============================================================

# --- Append Project Root (If necessary) ---
# sys.path.append(str(Path(__file__).resolve().parent)) # Uncomment if needed

# --- Load Environment Variables ---
load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger(__name__) # Define logger instance

# ===============================================================
# Configuration & Constants (Keep as is, but ensure GEMINI_API_KEY is secure)
# ===============================================================
# --- API Base ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# --- Caching ---
# TTL set to 60 seconds for option chain for faster reflection of updates
option_chain_cache = TTLCache(maxsize=50, ttl=60) # Increased TTL slightly
option_chain_lock = threading.Lock()
stock_data_cache = TTLCache(maxsize=100, ttl=600)
news_cache = TTLCache(maxsize=100, ttl=900)
analysis_cache = TTLCache(maxsize=50, ttl=1800)

# --- Background Update Thread Config ---
try:
    LIVE_UPDATE_INTERVAL_SECONDS = int(os.getenv("LIVE_UPDATE_INTERVAL", 3)) # Default 3s NOW
    if LIVE_UPDATE_INTERVAL_SECONDS <= 0: raise ValueError()
except:
    LIVE_UPDATE_INTERVAL_SECONDS = 3 # Default to 3 seconds NOW # <--- CHANGED HERE
    logger.warning(f"Invalid LIVE_UPDATE_INTERVAL value, defaulting to {LIVE_UPDATE_INTERVAL_SECONDS} seconds.")
logger.info(f"Background live update interval set to: {LIVE_UPDATE_INTERVAL_SECONDS} seconds.")

# --- Tax/Charge Constants ---
STT_SHORT_RATE = 0.000625
STT_EXERCISE_RATE = 0.00125
STAMP_DUTY_RATE = 0.00003
SEBI_RATE = 0.000001
NSE_TXN_CHARGE_RATE = 0.00053
GST_RATE = 0.18
BROKERAGE_FLAT_PER_ORDER = 20
DEFAULT_INTEREST_RATE_PCT = 6.59 # Example, check current rates

# --- Payoff Calculation Constants ---
PAYOFF_LOWER_BOUND_FACTOR = 0.80 # Adjusted factors for potentially better range
PAYOFF_UPPER_BOUND_FACTOR = 1.20
PAYOFF_POINTS = 350 # Increased points for smoother curve
BREAKEVEN_CLUSTER_GAP_PCT = 0.005

# --- LLM Configuration ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError(
            "CRITICAL: Gemini API Key (GEMINI_API_KEY) not found in environment variables. "
            "Analysis features will not work."
        )
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured successfully.")
except ValueError as ve:
    logger.error(str(ve))
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}. Analysis endpoint will fail.", exc_info=True)


# ===============================================================
# Global State (Keep as is)
# ===============================================================
selected_asset: Optional[str] = None
# strategy_positions: List[dict] = [] # Removed, as related endpoints were commented out
shutdown_event = threading.Event()
background_thread_instance: Optional[threading.Thread] = None
n: Optional[NSELive] = None

# ===============================================================
# Helper Functions (Keep as is)
# ===============================================================
def _safe_get_float(data: Dict, key: str, default: Optional[float] = None) -> Optional[float]:
    value = data.get(key)
    if value is None: return default
    try: return float(value)
    except (TypeError, ValueError):
        # logger.debug(f"Invalid float value '{value}' for key '{key}'. Using default {default}.") # Less noisy debug
        return default

def _safe_get_int(data: Dict, key: str, default: Optional[int] = None) -> Optional[int]:
    value = data.get(key)
    if value is None: return default
    try: return int(float(value)) # Allow float conversion first
    except (TypeError, ValueError):
        # logger.debug(f"Invalid int value '{value}' for key '{key}'. Using default {default}.") # Less noisy debug
        return default

def get_cached_option(asset: str) -> Optional[dict]:
    """
    Fetches option chain data, prioritizing cache, then live NSE fetch.
    Used by API endpoints primarily.
    """
    global n
    now = time.time()
    cache_key = f"option_chain_{asset}"

    with option_chain_lock:
        cached_data = option_chain_cache.get(cache_key)

    if cached_data:
        logger.debug(f"Cache hit for option chain: {asset}")
        return cached_data

    logger.info(f"Cache miss (API). Fetching live option chain for: {asset}")
    try:
        if not n:
            raise RuntimeError("NSELive client not initialized. Cannot fetch live data.")

        asset_upper = asset.upper()
        option_data = None
        if asset_upper in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
            logger.debug(f"Fetching INDEX option chain for {asset_upper}")
            option_data = n.index_option_chain(asset_upper)
        else:
            logger.debug(f"Fetching EQUITY option chain for {asset_upper}")
            option_data = n.equities_option_chain(asset_upper)

        if option_data and isinstance(option_data, dict):
            with option_chain_lock:
                option_chain_cache[cache_key] = option_data
            logger.info(f"Successfully fetched/cached LIVE option chain via API request for: {asset}")
            # Add spot price logging here for verification
            spot = option_data.get("records", {}).get("underlyingValue")
            logger.info(f"Live data for {asset} includes spot price: {spot}")
            return option_data
        else:
            logger.warning(f"Received empty or invalid data from NSELive for {asset}")
            return None
    except Exception as e:
        logger.error(f"Error fetching option chain from NSELive for {asset}: {e}", exc_info=False)
        return None

def fetch_and_update_single_asset_data(asset_name: str):
    """
    (Used by background thread)
    Fetches LIVE data directly from NSELive and updates the DATABASE for ONE asset.
    DEPENDS ON: assets table having last_spot_price, last_spot_update_time columns.
    REMAINS SYNCHRONOUS for background thread compatibility.
    """
    global n # Need access to the global NSELive client instance
    func_name = "fetch_and_update_single_asset_data"
    logger.info(f"[{func_name}] Starting live fetch & DB update for: {asset_name}")
    start_time = datetime.now()
    conn_obj = None # Initialize conn_obj for finally block
    option_source_data = None

    # --- Step 1: Directly Fetch Live Data ---
    try:
        if not n:
            raise RuntimeError("NSELive client ('n') is not initialized. Cannot fetch live data.")

        asset_upper = asset_name.upper()
        logger.debug(f"[{func_name}] Calling NSELive directly for {asset_upper}...")

        if asset_upper in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
            option_source_data = n.index_option_chain(asset_upper)
        else:
            option_source_data = n.equities_option_chain(asset_upper)

        if not option_source_data or not isinstance(option_source_data, dict):
            logger.warning(f"[{func_name}] Received empty or invalid data directly from NSELive for {asset_name}.")
            return # Exit if no data received

        live_spot = option_source_data.get("records", {}).get("underlyingValue")
        logger.info(f"[{func_name}] Successfully fetched live data for {asset_name} directly. Live Spot: {live_spot}")

    except RuntimeError as rte:
         logger.error(f"[{func_name}] Runtime Error during live fetch for {asset_name}: {rte}")
         return # Exit if client not ready
    except Exception as fetch_err:
        logger.error(f"[{func_name}] Error during direct NSELive fetch for {asset_name}: {fetch_err}", exc_info=False)
        return # Exit if fetch fails

    # --- Step 2: Process and Update Database ---
    try:
        if "records" not in option_source_data or "data" not in option_source_data["records"]:
            logger.error(f"[{func_name}] Invalid live data structure for {asset_name} after fetch.")
            return
        option_data_list = option_source_data["records"]["data"]
        # Get spot price here for DB update
        db_live_spot = _safe_get_float(option_source_data.get("records", {}), "underlyingValue")

        # If list is empty, still try to update spot price
        if not option_data_list:
            logger.warning(f"[{func_name}] No option data found in live source for {asset_name}. Attempting spot price update only.")
            # Fall through to DB connection block to update spot

        # Database interaction (remains synchronous)
        with get_db_connection() as conn: # Use the synchronous context manager
            if conn is None: raise ConnectionError("Failed to get DB connection.")
            conn_obj = conn # Assign conn to conn_obj for potential rollback
            with conn.cursor(dictionary=True) as cursor:
                # Get asset_id (sync)
                cursor.execute("SELECT id FROM option_data.assets WHERE asset_name = %s", (asset_name,))
                result = cursor.fetchone()
                if not result:
                    logger.error(f"[{func_name}] Asset '{asset_name}' not found in DB.")
                    return # Exit if asset not found
                asset_id = result["id"]

                # *** Update Spot Price in Assets Table (Check Column Exists) ***
                if db_live_spot is not None:
                    try:
                        # Ensure the columns last_spot_price and last_spot_update_time exist!
                        cursor.execute(
                            "UPDATE option_data.assets SET last_spot_price = %s, last_spot_update_time = NOW() WHERE id = %s",
                            (db_live_spot, asset_id)
                        )
                        if cursor.rowcount > 0:
                             logger.info(f"[{func_name}] Updated spot price ({db_live_spot}) in DB for asset ID {asset_id}")
                        else:
                             logger.warning(f"[{func_name}] Spot price update query affected 0 rows for asset ID {asset_id}.")
                    except mysql.connector.Error as spot_upd_err:
                         logger.error(f"[{func_name}] FAILED to update spot price in DB for {asset_name} (Check if columns exist): {spot_upd_err}")
                         # Don't necessarily exit, maybe chain update can proceed
                else:
                    logger.warning(f"[{func_name}] Could not extract spot price from live data for DB update ({asset_name}).")

                # If no option data, commit spot price update and exit
                if not option_data_list:
                    conn.commit()
                    logger.info(f"[{func_name}] Committed spot price update only for {asset_name}.")
                    return

                # --- Process Expiries (Sync) ---
                expiry_dates_formatted = set()
                for item in option_data_list:
                    raw_expiry = item.get("expiryDate")
                    if raw_expiry:
                        try:
                            expiry_dates_formatted.add(datetime.strptime(raw_expiry, "%d-%b-%Y").strftime("%Y-%m-%d"))
                        except (ValueError, TypeError):
                             logger.debug(f"[{func_name}] Skipping invalid expiry format: {raw_expiry}")
                             pass # Skip invalid formats

                # Delete Old Expiries (sync)
                today_str = date.today().strftime("%Y-%m-%d")
                cursor.execute("DELETE FROM option_data.expiries WHERE asset_id = %s AND expiry_date < %s", (asset_id, today_str))
                logger.debug(f"[{func_name}] Deleted expiries before {today_str} for asset ID {asset_id} (Affected: {cursor.rowcount}).")


                # Upsert Current Expiries & Fetch IDs (sync)
                expiry_id_map = {}
                if expiry_dates_formatted:
                    ins_data = [(asset_id, e) for e in expiry_dates_formatted]
                    # Use INSERT IGNORE or ON DUPLICATE KEY UPDATE depending on desired behavior for existing dates
                    upsert_expiry_sql = "INSERT INTO option_data.expiries (asset_id, expiry_date) VALUES (%s, %s) ON DUPLICATE KEY UPDATE expiry_date = VALUES(expiry_date)"
                    cursor.executemany(upsert_expiry_sql, ins_data)
                    logger.debug(f"[{func_name}] Upserted {len(ins_data)} expiries for asset ID {asset_id} (Affected: {cursor.rowcount}).")

                    # Fetch IDs for the relevant expiries
                    placeholders = ', '.join(['%s'] * len(expiry_dates_formatted))
                    select_expiry_sql = f"SELECT id, expiry_date FROM option_data.expiries WHERE asset_id = %s AND expiry_date IN ({placeholders})"
                    cursor.execute(select_expiry_sql, (asset_id, *expiry_dates_formatted))
                    for row in cursor.fetchall():
                         # Ensure expiry_date is stringified correctly
                         expiry_key = row["expiry_date"].strftime("%Y-%m-%d") if isinstance(row["expiry_date"], date) else str(row["expiry_date"])
                         expiry_id_map[expiry_key] = row["id"]
                    logger.debug(f"[{func_name}] Fetched {len(expiry_id_map)} expiry IDs for mapping.")


                # Prepare Option Chain Data (sync)
                option_chain_data_to_upsert = []
                skipped = 0
                processed_options = 0 # Count individual CE/PE options processed
                for item in option_data_list:
                    try:
                        # Safely extract strike and expiry
                        raw_strike = item.get('strikePrice')
                        raw_expiry = item.get('expiryDate')
                        if raw_strike is None or raw_expiry is None:
                            skipped += 1; continue

                        strike = float(raw_strike) # Convert strike safely
                        expiry_date_str = datetime.strptime(raw_expiry, "%d-%b-%Y").strftime("%Y-%m-%d")
                        expiry_id = expiry_id_map.get(expiry_date_str)

                        if expiry_id is None:
                            logger.debug(f"[{func_name}] Skipping row for expiry {expiry_date_str} as ID not found in map.")
                            skipped += 1; continue

                        # Process CE and PE safely
                        for opt_type in ["CE", "PE"]:
                            details = item.get(opt_type)
                            if isinstance(details, dict):
                                processed_options += 1 # Increment option count
                                idf = details.get("identifier", f"{asset_name}_{expiry_date_str}_{strike}_{opt_type}") # Default identifier
                                # Use safe getters for all numeric fields
                                row_data = (
                                    asset_id, expiry_id, strike, opt_type, idf,
                                    _safe_get_int(details, "openInterest", 0),
                                    _safe_get_int(details, "changeinOpenInterest", 0),
                                    _safe_get_int(details, "totalTradedVolume", 0),
                                    _safe_get_float(details, "impliedVolatility", 0.0),
                                    _safe_get_float(details, "lastPrice", 0.0),
                                    _safe_get_int(details, "bidQty", 0),
                                    _safe_get_float(details, "bidprice", 0.0),
                                    _safe_get_int(details, "askQty", 0),
                                    _safe_get_float(details, "askPrice", 0.0),
                                    _safe_get_int(details, "totalBuyQuantity", 0),
                                    _safe_get_int(details, "totalSellQuantity", 0)
                                )
                                option_chain_data_to_upsert.append(row_data)
                            else:
                                # Log if CE/PE details are missing or not a dict
                                logger.debug(f"[{func_name}] Missing or invalid details for {opt_type} at strike {strike}, expiry {expiry_date_str}.")
                                # Don't increment skipped here unless you count missing CE/PE as skipped rows

                    except (ValueError, TypeError, KeyError) as row_err:
                        # Catch specific errors during row processing
                        logger.warning(f"[{func_name}] Error processing live data row: {row_err}. Row snapshot: {item.get('strikePrice')}, {item.get('expiryDate')}")
                        skipped += 1
                    except Exception as general_row_err:
                         # Catch unexpected errors
                        logger.error(f"[{func_name}] Unexpected error processing live data row: {general_row_err}", exc_info=False)
                        skipped += 1

                logger.debug(f"[{func_name}] Prepared {len(option_chain_data_to_upsert)} option rows ({processed_options} options total) for {asset_name} DB upsert. Skipped {skipped} input rows.")

                # Upsert Option Chain Data (sync)
                if option_chain_data_to_upsert:
                    upsert_chain_sql = """
                        INSERT INTO option_data.option_chain (
                            asset_id, expiry_id, strike_price, option_type, identifier,
                            open_interest, change_in_oi, total_traded_volume, implied_volatility,
                            last_price, bid_qty, bid_price, ask_qty, ask_price,
                            total_buy_qty, total_sell_qty
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON DUPLICATE KEY UPDATE
                            identifier=VALUES(identifier), open_interest=VALUES(open_interest),
                            change_in_oi=VALUES(change_in_oi), total_traded_volume=VALUES(total_traded_volume),
                            implied_volatility=VALUES(implied_volatility), last_price=VALUES(last_price),
                            bid_qty=VALUES(bid_qty), bid_price=VALUES(bid_price),
                            ask_qty=VALUES(ask_qty), ask_price=VALUES(ask_price),
                            total_buy_qty=VALUES(total_buy_qty), total_sell_qty=VALUES(total_sell_qty)
                    """
                    cursor.executemany(upsert_chain_sql, option_chain_data_to_upsert)
                    logger.debug(f"[{func_name}] Upserted option chain for {asset_name} (Affected: {cursor.rowcount}).")
                else:
                     logger.info(f"[{func_name}] No valid option chain data prepared for upsert for {asset_name}.")

                # Commit (sync)
                conn.commit()
                logger.info(f"[{func_name}] Successfully committed DB update for {asset_name}.")

    # Error handling for DB part
    except (mysql.connector.Error, ConnectionError) as db_err:
        logger.error(f"[{func_name}] DB/Connection error during update for {asset_name}: {db_err}")
        try:
            # Check if conn_obj was assigned and is connected before rollback
            if conn_obj and conn_obj.is_connected():
                 conn_obj.rollback()
                 logger.info(f"[{func_name}] Rollback attempted due to DB error.")
        except Exception as rb_err:
            logger.error(f"[{func_name}] Rollback failed: {rb_err}")
    except Exception as e:
        logger.error(f"[{func_name}] Unexpected error during DB update phase for {asset_name}: {e}", exc_info=True)
        try:
             # Check if conn_obj was assigned and is connected before rollback
            if conn_obj and conn_obj.is_connected():
                 conn_obj.rollback()
                 logger.info(f"[{func_name}] Rollback attempted due to unexpected error.")
        except Exception as rb_err:
            logger.error(f"[{func_name}] Rollback failed: {rb_err}")
    # No finally block needed as `with get_db_connection()` handles closing

    duration = datetime.now() - start_time
    logger.info(f"[{func_name}] Finished task for asset: {asset_name}. Duration: {duration}")


def live_update_runner():
    """ Background thread target function. Calls the DIRECT fetch/update function. """
    global selected_asset
    thread_name = threading.current_thread().name
    logger.info(f"Background update thread '{thread_name}' started. Interval: {LIVE_UPDATE_INTERVAL_SECONDS}s.")
    # Initial wait before first run
    time.sleep(5)
    while not shutdown_event.is_set():
        asset_to_update = selected_asset
        if asset_to_update and isinstance(asset_to_update, str) and asset_to_update.strip():
            logger.info(f"[{thread_name}] Updating DB data for selected asset: {asset_to_update}")
            start_time = time.monotonic()
            try:
                fetch_and_update_single_asset_data(asset_to_update) # Updates DB
                duration = time.monotonic() - start_time
                logger.info(f"[{thread_name}] Finished DB update cycle for {asset_to_update}. Duration: {duration:.3f}s")
            except Exception as e:
                duration = time.monotonic() - start_time
                logger.error(f"[{thread_name}] Error in DB update cycle for {asset_to_update} after {duration:.3f}s: {e}", exc_info=True)
        else:
            logger.debug(f"[{thread_name}] No asset selected. Background thread waiting...")

        # Use wait instead of sleep to allow faster shutdown
        shutdown_event.wait(timeout=LIVE_UPDATE_INTERVAL_SECONDS)
    logger.info(f"Background update thread '{thread_name}' stopping.")


# ===============================================================
# FastAPI Application Setup & Lifespan (Keep as is)
# ===============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global background_thread_instance, n
    logger.info("Application starting up...")
    try: initialize_database_pool(); logger.info("Database pool initialized.")
    except Exception as db_err: logger.exception("CRITICAL: DB Pool Init Failed.")
    try: n = NSELive(); logger.info("NSELive client initialized.")
    except Exception as nse_err: logger.error(f"Failed to initialize NSELive client: {nse_err}. Live fetches will fail."); n = None
    logger.info("Starting background DB update thread...")
    shutdown_event.clear()
    background_thread_instance = threading.Thread(target=live_update_runner, name="LiveDBUpdateThread", daemon=True)
    background_thread_instance.start()
    yield
    logger.info("Application shutting down...") # Shutdown sequence...
    shutdown_event.set()
    if background_thread_instance and background_thread_instance.is_alive():
        logger.info("Waiting for background thread to finish...")
        background_thread_instance.join(timeout=LIVE_UPDATE_INTERVAL_SECONDS + 2)
        if background_thread_instance.is_alive(): logger.warning("Background thread did not stop gracefully.")
        else: logger.info("Background thread stopped.")
    logger.info("Closing database pool (if function exists)...")
    # Add pool closing logic if your database.py provides it
    # e.g., if 'close_database_pool' exists in database.py:
    # try:
    #     from database import close_database_pool
    #     close_database_pool()
    #     logger.info("Database pool closed.")
    # except ImportError:
    #     logger.info("No close_database_pool function found.")
    # except Exception as pool_close_err:
    #     logger.error(f"Error closing database pool: {pool_close_err}")
    logger.info("Application shutdown complete.")


app = FastAPI(
    title="Option Strategy Analyzer API",
    description="API for fetching option data, calculating strategies, and performing analysis.",
    version="1.2.0", # Incremented version (Bug fixes)
    lifespan=lifespan
)

# --- CORS Middleware ---
# Make sure your Render deployment allows these origins
ALLOWED_ORIGINS = [
    "http://localhost", "http://localhost:3000", "http://127.0.0.1:8000",
    "https://option-strategy-vaqz.onrender.com",
    "https://option-strategy-website.onrender.com"
    # Add any other specific origins if needed
]
logger.info(f"Configuring CORS for origins: {ALLOWED_ORIGINS}")
app.add_middleware( CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ===============================================================
# Pydantic Models (Keep as is)
# ===============================================================
class AssetUpdateRequest(BaseModel): asset: str
class SpotPriceResponse(BaseModel): spot_price: float; timestamp: Optional[str] = None # Added timestamp
class StockRequest(BaseModel): asset: str
class PositionInput(BaseModel): symbol: str; strike: float; type: str = Field(pattern="^(CE|PE)$"); quantity: int; price: float
class StrategyLegInputPayoff(BaseModel): option_type: str = Field(pattern="^(CE|PE)$"); strike_price: Union[float, str]; tr_type: str = Field(pattern="^(b|s)$"); option_price: Union[float, str]; expiry_date: str; lots: Union[int, str]; lot_size: Optional[Union[int, str]] = None
class PayoffRequest(BaseModel): asset: str; strategy: List[StrategyLegInputPayoff]
class DebugAssetSelectRequest(BaseModel): asset: str


# ===============================================================
# Calculation Functions (Modify news fetch, payoff chart)
# ===============================================================
def get_lot_size(asset_name: str) -> int | None:
    logger.debug(f"Fetching lot size for asset: {asset_name}")
    sql = "SELECT lot_size FROM option_data.assets WHERE asset_name = %s"
    lot_size = None
    try:
        with get_db_connection() as conn:
            with conn.cursor(dictionary=True) as cursor: # Use dictionary cursor
                cursor.execute(sql, (asset_name,))
                result = cursor.fetchone()
                if result and result.get("lot_size") is not None:
                    try: lot_size = int(result["lot_size"]); logger.debug(f"Found lot size for {asset_name}: {lot_size}")
                    except (ValueError, TypeError): logger.error(f"Invalid non-integer lot size '{result['lot_size']}' in DB for {asset_name}.")
                elif not result: logger.warning(f"No asset found with name: {asset_name} in assets table.")
                else: logger.warning(f"Lot size is NULL in DB for {asset_name}.")
    except ConnectionError as e: logger.error(f"DB Connection Error fetching lot size for {asset_name}: {e}", exc_info=False) # Less verbose log
    except mysql.connector.Error as e: logger.error(f"DB Query Error fetching lot size for {asset_name}: {e}", exc_info=False)
    except Exception as e: logger.error(f"Unexpected error fetching lot size for {asset_name}: {e}", exc_info=True)
    return lot_size


def extract_iv(asset_name: str, strike_price: float, expiry_date: str, option_type: str) -> float | None:
    """Extracts Implied Volatility from cached option chain data."""
    logger.debug(f"Attempting to extract IV for {asset_name} {expiry_date} {strike_price} {option_type}")
    try:
        target_expiry_nse_fmt = datetime.strptime(expiry_date, "%Y-%m-%d").strftime("%d-%b-%Y").upper() # Match NSE format
    except ValueError:
        logger.error(f"Invalid expiry date format provided to extract_iv: {expiry_date}")
        return None

    try:
        # Use the same cache the API relies on
        option_data = get_cached_option(asset_name) # Fetches live if cache misses

        if not isinstance(option_data, dict):
            logger.warning(f"Cached/Live data not dict for {asset_name} in extract_iv")
            return None
        records = option_data.get("records")
        data_list = records.get("data") if isinstance(records, dict) else None
        if not isinstance(data_list, list):
            logger.warning(f"Records.data not list for {asset_name} in extract_iv")
            return None

        option_key = option_type.upper() # CE or PE

        for item in data_list:
            if isinstance(item, dict):
                 item_strike = _safe_get_float(item, "strikePrice")
                 item_expiry_nse_fmt = item.get("expiryDate", "").upper() # Get expiry from data

                 # Check type and value equality carefully
                 if (item_strike is not None and
                     abs(item_strike - strike_price) < 0.01 and # Float comparison tolerance
                     item_expiry_nse_fmt == target_expiry_nse_fmt):
                    option_details = item.get(option_key)
                    if isinstance(option_details, dict):
                        iv = option_details.get("impliedVolatility")
                        # Ensure IV is a valid positive number
                        if iv is not None and isinstance(iv, (int, float)) and iv > 0:
                            logger.debug(f"Found IV {iv} for {asset_name} {target_expiry_nse_fmt} {strike_price} {option_key}")
                            return float(iv)
                        else:
                            logger.debug(f"IV missing or invalid (<=0): value={iv} for {asset_name} {strike_price} {option_key} on {target_expiry_nse_fmt}")
                    else:
                         logger.debug(f"Details for option type {option_key} not found or not dict for strike {strike_price} on {target_expiry_nse_fmt}")

        logger.warning(f"No matching contract/valid IV found for {asset_name} {strike_price}@{target_expiry_nse_fmt} {option_key}")
        return None
    except Exception as e:
        logger.error(f"Error extracting IV for {asset_name}: {e}", exc_info=True)
        return None


# ===============================================================
# 1. Calculate Option Taxes (Keep as corrected previously)
# ===============================================================
def calculate_option_taxes(strategy_data: List[Dict[str, Any]], asset: str) -> Optional[Dict[str, Any]]:
    # (Assuming this function is already robust and correct from previous iterations)
    # ... (Keep the existing robust implementation) ...
    # If issues persist, this function might need revisiting, but focusing on user's request first.
    func_name = "calculate_option_taxes"
    logger.info(f"[{func_name}] Calculating for {len(strategy_data)} leg(s), asset: {asset}")
    # --- Fetch Prerequisites ---
    try:
        # Get spot price reliably - try DB first, then cache/live
        spot_price_info = get_latest_spot_price_from_db(asset) # Use new helper
        if spot_price_info:
            spot_price = spot_price_info['spot_price']
            logger.info(f"[{func_name}] Using Spot Price from DB: {spot_price}")
        else:
             # Fallback to cached/live data if DB fails
             cached_data = get_cached_option(asset)
             if not cached_data or "records" not in cached_data:
                 raise ValueError("Missing or invalid market data (cache/live) for tax calc")
             spot_price = _safe_get_float(cached_data.get("records", {}), "underlyingValue")
             logger.info(f"[{func_name}] Using Spot Price from Cache/Live: {spot_price}")

        if spot_price is None or spot_price <= 0:
            raise ValueError(f"Spot price missing or invalid ({spot_price})")

        default_lot_size = get_lot_size(asset)
        if default_lot_size is None or default_lot_size <= 0:
            raise ValueError(f"Default lot size missing or invalid ({default_lot_size})")
        logger.debug(f"[{func_name}] Using Spot: {spot_price}, Default Lot Size: {default_lot_size}")

    except ValueError as val_err:
        logger.error(f"[{func_name}] Failed initial data fetch for {asset}: {val_err}")
        return None
    except Exception as e:
        logger.error(f"[{func_name}] Unexpected error fetching initial data for {asset}: {e}", exc_info=True)
        return None

    # --- Initialize Totals ---
    totals = defaultdict(float) # Use defaultdict for easier summing
    breakdown = []

    # --- Process Each Leg ---
    for i, leg in enumerate(strategy_data):
        try:
            # --- Extract and Validate Leg Data ---
            tr_type = str(leg.get('tr_type', '')).lower()
            op_type = str(leg.get('op_type', '')).lower() # 'c' or 'p' expected from preparation step
            strike = _safe_get_float(leg, 'strike')
            premium = _safe_get_float(leg, 'op_pr')
            lots = _safe_get_int(leg, 'lot')

            # Determine Lot Size (use leg-specific if valid, else default)
            leg_lot_size = default_lot_size
            raw_ls = leg.get('lot_size')
            if raw_ls is not None:
                temp_ls = _safe_get_int({'ls': raw_ls}, 'ls') # Use helper safely
                if temp_ls is not None and temp_ls > 0:
                    leg_lot_size = temp_ls

            # Validate parameters
            error_msg = None
            if tr_type not in ['b','s']: error_msg = f"invalid tr_type ({tr_type})"
            elif op_type not in ['c','p']: error_msg = f"invalid op_type ({op_type})"
            elif strike is None or strike <= 0: error_msg = f"invalid strike ({strike})"
            elif premium is None or premium < 0: error_msg = f"invalid premium ({premium})" # Allow 0 premium
            elif lots is None or lots <= 0: error_msg = f"invalid lots ({lots})"
            elif leg_lot_size <= 0: error_msg = f"invalid lot_size ({leg_lot_size})"

            if error_msg:
                 raise ValueError(f"Leg {i}: {error_msg}. Data: {leg}")

            # --- Calculate Charges for the Leg ---
            quantity = lots * leg_lot_size
            turnover = premium * quantity
            stt_val = 0.0
            stt_note = ""

            # Note: Tax calculation assumes expiry scenario based on current spot.
            # Real-time calculation during trade lifecycle might differ.

            if tr_type == 's': # Sell side STT (on premium)
                stt_val = turnover * STT_SHORT_RATE
                stt_note = f"{STT_SHORT_RATE*100:.4f}% STT (Sell Premium)"
            elif tr_type == 'b': # Buy side STT (on intrinsic value if ITM on expiry/exercise)
                intrinsic = 0.0
                is_itm = False
                if op_type == 'c' and spot_price > strike:
                    intrinsic = (spot_price - strike) * quantity
                    is_itm = True
                elif op_type == 'p' and spot_price < strike:
                    intrinsic = (strike - spot_price) * quantity
                    is_itm = True

                # STT applied only if ITM and intrinsic > 0 at expiry (estimated using current spot)
                if is_itm and intrinsic > 0:
                    stt_val = intrinsic * STT_EXERCISE_RATE
                    stt_note = f"{STT_EXERCISE_RATE*100:.4f}% STT (Est. Exercise ITM)"
                else:
                    stt_note = "No STT (Est. Buy OTM/ATM)"

            # Other charges
            stamp_duty_value = turnover * STAMP_DUTY_RATE if tr_type == 'b' else 0.0
            sebi_fee_value = turnover * SEBI_RATE
            txn_charge_value = turnover * NSE_TXN_CHARGE_RATE
            brokerage_value = float(BROKERAGE_FLAT_PER_ORDER) # Per leg/order
            gst_on_charges = (brokerage_value + sebi_fee_value + txn_charge_value) * GST_RATE

            # --- Accumulate Totals ---
            totals["stt"] += stt_val
            totals["stamp_duty"] += stamp_duty_value
            totals["sebi_fee"] += sebi_fee_value
            totals["txn_charges"] += txn_charge_value
            totals["brokerage"] += brokerage_value
            totals["gst"] += gst_on_charges

            # --- Store Leg Breakdown ---
            leg_total_cost = stt_val + stamp_duty_value + sebi_fee_value + txn_charge_value + brokerage_value + gst_on_charges
            breakdown.append({
                "leg_index": i,
                "transaction_type": tr_type.upper(),
                "option_type": op_type.upper(), # Use c/p consistent with other parts
                "strike": strike,
                "lots": lots,
                "lot_size": leg_lot_size,
                "quantity": quantity,
                "premium_per_share": round(premium, 2),
                "turnover": round(turnover, 2),
                "stt": round(stt_val, 2),
                "stt_note": stt_note,
                "stamp_duty": round(stamp_duty_value, 2),
                "sebi_fee": round(sebi_fee_value, 4),
                "txn_charge": round(txn_charge_value, 4),
                "brokerage": round(brokerage_value, 2),
                "gst": round(gst_on_charges, 2),
                "total_leg_cost": round(leg_total_cost, 2)
            })

        except ValueError as val_err:
            logger.error(f"[{func_name}] Validation Error processing leg {i} for {asset}: {val_err}")
            raise ValueError(f"Invalid data in tax leg {i+1}: {val_err}") from val_err
        except Exception as leg_err:
            logger.error(f"[{func_name}] Unexpected Error processing tax leg {i} for {asset}: {leg_err}", exc_info=True)
            raise ValueError(f"Unexpected error processing tax leg {i+1}") from leg_err

    # --- Finalize and Return ---
    final_charges_summary = {k: round(v, 2) for k, v in totals.items()}
    final_total_cost = round(sum(totals.values()), 2)

    logger.info(f"[{func_name}] Calculation complete for {asset}. Total estimated charges: {final_total_cost:.2f}")

    rate_info = {
        "STT_SHORT_RATE": STT_SHORT_RATE, "STT_EXERCISE_RATE": STT_EXERCISE_RATE,
        "STAMP_DUTY_RATE": STAMP_DUTY_RATE, "SEBI_RATE": SEBI_RATE,
        "NSE_TXN_CHARGE_RATE": NSE_TXN_CHARGE_RATE, "GST_RATE": GST_RATE,
        "BROKERAGE_FLAT_PER_ORDER": BROKERAGE_FLAT_PER_ORDER
    }
    return {
        "calculation_details": {
            "asset": asset,
            "spot_price_used": spot_price,
            "default_lot_size_used": default_lot_size,
            "rate_info": rate_info
        },
        "total_estimated_cost": final_total_cost,
        "charges_summary": final_charges_summary,
        "breakdown_per_leg": breakdown
    }

# ===============================================================
# 2. Generate Payoff Chart (using Matplotlib - ENHANCED AXIS)
# ===============================================================
def generate_payoff_chart_matplotlib(
    strategy_data: List[Dict[str, Any]],
    asset: str,
    strategy_metrics: Optional[Dict[str, Any]] # Pass calculated metrics
) -> Optional[str]:
    """Generates payoff chart using Matplotlib with improved Y-axis scaling."""
    func_name = "generate_payoff_chart_matplotlib"
    logger.info(f"[{func_name}] Generating chart for {len(strategy_data)} leg(s), asset: {asset}")
    start_time = time.monotonic()

    fig = None # Initialize fig to None
    try:
        # 1. Fetch Prerequisites
        # Spot price and lot size are needed again for payoff range calculation
        spot_price_info = get_latest_spot_price_from_db(asset)
        if spot_price_info: spot_price = spot_price_info['spot_price']
        else:
            cached_data = get_cached_option(asset)
            spot_price = _safe_get_float(cached_data.get("records", {}), "underlyingValue") if cached_data else None

        default_lot_size = get_lot_size(asset)
        if not spot_price or spot_price <= 0 or not default_lot_size or default_lot_size <= 0:
            raise ValueError(f"Invalid spot ({spot_price}) or lot size ({default_lot_size}) for chart generation")

        # 2. Calculate Payoff Data
        # Define price range based on spot and factors
        lower_bound = max(spot_price * PAYOFF_LOWER_BOUND_FACTOR, 0.1) # Ensure > 0
        upper_bound = spot_price * PAYOFF_UPPER_BOUND_FACTOR
        price_range = np.linspace(lower_bound, upper_bound, PAYOFF_POINTS)
        total_payoff = np.zeros_like(price_range)
        unique_strikes = set()
        processed_legs_count = 0

        for i, leg in enumerate(strategy_data): # Loop through legs
            try: # Extract/validate leg - Reuse logic from metrics calc is better
                tr_type = leg['tr_type'].lower()
                op_type = leg['op_type'].lower() # Expect 'c' or 'p'
                strike = float(leg['strike'])
                premium = float(leg['op_pr'])
                lots = int(leg['lot'])
                # Use leg-specific lot size if available and valid, else default
                leg_lot_size = default_lot_size
                raw_ls = leg.get('lot_size')
                if raw_ls is not None:
                    temp_ls = _safe_get_int({'ls': raw_ls}, 'ls')
                    if temp_ls is not None and temp_ls > 0: leg_lot_size = temp_ls

                # Basic validation
                if not(tr_type in ('b','s') and op_type in ('c','p') and strike > 0 and premium >= 0 and lots > 0 and leg_lot_size > 0):
                    raise ValueError(f"Invalid params: type={op_type}, tr={tr_type}, k={strike}, prem={premium}, lots={lots}, ls={leg_lot_size}")
                unique_strikes.add(strike)
            except Exception as e:
                raise ValueError(f"Leg {i+1} processing error for chart: {e}") from e

            # Calculate payoff for this leg across the price range
            quantity = lots * leg_lot_size
            leg_prem_tot = premium * quantity # Total premium cost/credit for the leg
            intrinsic_value = np.maximum(price_range - strike, 0) if op_type == 'c' else np.maximum(strike - price_range, 0)
            leg_payoff = (intrinsic_value * quantity - leg_prem_tot) if tr_type == 'b' else (leg_prem_tot - intrinsic_value * quantity)
            total_payoff += leg_payoff
            processed_legs_count += 1

        if processed_legs_count == 0:
            logger.warning(f"[{func_name}] No valid legs processed for chart: {asset}.")
            return None

        # --- 3. Determine Y-axis Limits ---
        actual_min_pl = np.min(total_payoff)
        actual_max_pl = np.max(total_payoff)
        logger.debug(f"[{func_name}] Actual PL range in plot: Min={actual_min_pl:.2f}, Max={actual_max_pl:.2f}")

        # Get theoretical max/min from passed metrics
        theo_max_profit = np.inf
        theo_max_loss = -np.inf
        if strategy_metrics and isinstance(strategy_metrics.get("metrics"), dict):
            max_p_str = strategy_metrics["metrics"].get("max_profit", "∞")
            max_l_str = strategy_metrics["metrics"].get("max_loss", "-∞")
            try: theo_max_profit = float(max_p_str) if max_p_str != "∞" else np.inf
            except (ValueError, TypeError): pass
            try: theo_max_loss = float(max_l_str) if max_l_str != "-∞" else -np.inf
            except (ValueError, TypeError): pass
        logger.debug(f"[{func_name}] Theoretical PL range: Min={theo_max_loss}, Max={theo_max_profit}")

        # Determine plot limits: Use theoretical if finite and outside actual range, else use actual
        y_min_limit = actual_min_pl
        if np.isfinite(theo_max_loss) and theo_max_loss < actual_min_pl:
            y_min_limit = theo_max_loss

        y_max_limit = actual_max_pl
        if np.isfinite(theo_max_profit) and theo_max_profit > actual_max_pl:
            y_max_limit = theo_max_profit

        # Add padding (ensure range isn't zero)
        y_range = y_max_limit - y_min_limit
        if abs(y_range) < 1e-6: y_range = max(abs(y_max_limit), abs(y_min_limit), 1.0) # Avoid zero range

        y_padding = y_range * 0.10 # 10% padding
        final_y_min = y_min_limit - y_padding
        final_y_max = y_max_limit + y_padding
        # Ensure zero line is clearly visible if PL crosses zero
        if y_min_limit < 0 < y_max_limit:
            final_y_min = min(final_y_min, y_min_limit * 1.1) # Ensure loss shown
            final_y_max = max(final_y_max, y_max_limit * 1.1) # Ensure profit shown
        elif y_max_limit <= 0 : # Only loss
             final_y_max = max(final_y_max, y_padding) # Show a bit above zero
        elif y_min_limit >= 0: # Only profit
             final_y_min = min(final_y_min, -y_padding) # Show a bit below zero

        logger.debug(f"[{func_name}] Final Y-Axis limits: Min={final_y_min:.2f}, Max={final_y_max:.2f}")


        # 4. --- Create Matplotlib Figure ---
        plt.close('all') # Close previous figures
        plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=95) # Adjust size/dpi as needed

        # 5. --- Plot and Style ---
        ax.plot(price_range, total_payoff, color='mediumblue', linewidth=2.0, label="Strategy Payoff", zorder=10)
        ax.axhline(0, color='black', linewidth=1.0, linestyle='-', alpha=0.9, zorder=1) # Zero P/L line
        # Use spot price fetched at the start of this function
        ax.axvline(spot_price, color='dimgrey', linestyle='--', linewidth=1.2, label=f'Spot {spot_price:.2f}', zorder=1)

        # Plot strikes (similar logic as before, but adjust y position based on new limits)
        strike_line_color = 'darkorange'
        # Place strike labels below the lowest point, considering the final y-axis limit
        text_y_offset_factor = 0.05 # Relative offset based on final y-range
        final_y_range = final_y_max - final_y_min
        text_y_pos = final_y_min - final_y_range * text_y_offset_factor

        for k in sorted(list(unique_strikes)):
             ax.axvline(k, color=strike_line_color, linestyle=':', linewidth=1.0, alpha=0.75, zorder=1)
             # Use calculated text_y_pos for labels
             ax.text(k, text_y_pos, f'{k:g}', color=strike_line_color, ha='center', va='top', fontsize=9, alpha=0.95, weight='medium')

        # Fill profit/loss areas
        ax.fill_between(price_range, total_payoff, 0, where=total_payoff >= 0, facecolor='palegreen', alpha=0.5, interpolate=True, zorder=0)
        ax.fill_between(price_range, total_payoff, 0, where=total_payoff < 0, facecolor='lightcoral', alpha=0.5, interpolate=True, zorder=0)

        # Titles and Labels
        ax.set_title(f"{asset} Strategy Payoff", fontsize=15, weight='bold')
        ax.set_xlabel("Underlying Price at Expiry", fontsize=11)
        ax.set_ylabel("Profit / Loss (₹)", fontsize=11) # Explicit label
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, which='major', linestyle=':', linewidth=0.6, alpha=0.7)

        # Apply calculated axis limits
        x_padding = (upper_bound - lower_bound) * 0.02
        ax.set_xlim(lower_bound - x_padding, upper_bound + x_padding)
        ax.set_ylim(final_y_min, final_y_max) # Use calculated limits

        fig.tight_layout(pad=1.0) # Adjust padding

        # 6. --- Save to Buffer ---
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=fig.dpi)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode("utf-8")

        duration = time.monotonic() - start_time
        logger.info(f"[{func_name}] Successfully generated Matplotlib chart for {asset} in {duration:.3f}s")
        return img_base64

    except ValueError as val_err:
        logger.error(f"[{func_name}] Value Error: {val_err}")
        return None
    except Exception as e:
        logger.error(f"[{func_name}] Unexpected error generating chart for {asset}: {e}", exc_info=True)
        return None
    finally:
        # Ensure the figure is closed to release memory
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass # Ignore errors during close

# ===============================================================
# 3. Calculate Strategy Metrics (REVISED Breakeven Calculation)
# ===============================================================
def calculate_strategy_metrics(strategy_data: List[Dict[str, Any]], asset: str) -> Optional[Dict[str, Any]]:
    """
    Calculates Profit & Loss metrics for a multi-leg options strategy,
    using theoretical P/L limits and refined breakeven calculation.
    """
    func_name = "calculate_strategy_metrics"
    logger.info(f"[{func_name}] Calculating metrics for {len(strategy_data)} leg(s), asset: {asset}")

    # --- 1. Fetch Essential Data ---
    try:
        # Use DB helper for spot price consistency
        spot_price_info = get_latest_spot_price_from_db(asset)
        if spot_price_info: spot_price = spot_price_info['spot_price']
        else: # Fallback
            cached_data = get_cached_option(asset)
            spot_price = _safe_get_float(cached_data.get("records", {}), "underlyingValue") if cached_data else None

        if spot_price is None or not isinstance(spot_price, (int, float)) or spot_price <= 0:
             raise ValueError(f"Spot price missing/invalid ({spot_price})")
        spot_price = float(spot_price)

        default_lot_size = get_lot_size(asset)
        if default_lot_size is None or default_lot_size <= 0:
            raise ValueError("Default lot size missing/invalid")
    except Exception as e:
        logger.error(f"[{func_name}] Failed prerequisite data fetch for {asset}: {e}")
        return None

    # --- 2. Process Strategy Legs & Store Details ---
    total_net_premium = 0.0; cost_breakdown = []; processed_legs = 0
    net_long_call_qty = 0; net_short_call_qty = 0; net_long_put_qty = 0; net_short_put_qty = 0
    payoff_at_S_equals_0 = 0.0; legs_for_payoff_calc = []
    all_strikes_list = []

    for i, leg in enumerate(strategy_data):
        try:
            tr_type = str(leg.get('tr_type', '')).lower()
            option_type = str(leg.get('op_type', '')).lower() # 'c' or 'p'
            strike = _safe_get_float(leg, 'strike'); premium = _safe_get_float(leg, 'op_pr')
            lots = _safe_get_int(leg, 'lot'); leg_lot_size = _safe_get_int(leg, 'lot_size', default_lot_size)
            if tr_type not in ('b','s') or option_type not in ('c','p') or strike is None or strike <= 0 or premium is None or premium < 0 or lots is None or lots <= 0 or leg_lot_size <= 0:
                raise ValueError("Invalid leg parameters")
            quantity = lots * leg_lot_size; leg_premium_total = premium * quantity; action_verb = ""
            all_strikes_list.append(strike)

            if tr_type == 'b': # Buy
                total_net_premium -= leg_premium_total; action_verb = "Paid"
                if option_type == 'c': net_long_call_qty += quantity
                else: net_long_put_qty += quantity
            else: # Sell
                total_net_premium += leg_premium_total; action_verb = "Received"
                if option_type == 'c': net_short_call_qty += quantity
                else: net_short_put_qty += quantity

            # Calculate payoff component if underlying price goes to 0
            if option_type == 'p': # Puts have value at S=0
                intrinsic_at_zero = strike # Intrinsic value is K - S = K - 0 = K
                payoff_at_S_equals_0 += (intrinsic_at_zero * quantity - leg_premium_total) if tr_type == 'b' else (leg_premium_total - intrinsic_at_zero * quantity)
            else: # Calls expire worthless at S=0
                payoff_at_S_equals_0 += -leg_premium_total if tr_type == 'b' else leg_premium_total

            cost_breakdown.append({ "leg_index": i, "action": tr_type.upper(), "type": option_type.upper(), "strike": strike, "premium_per_share": premium, "lots": lots, "lot_size": leg_lot_size, "quantity": quantity, "total_premium": round(leg_premium_total if tr_type=='s' else -leg_premium_total, 2), "effect": action_verb })
            # Store details needed *only* for payoff calculation function
            legs_for_payoff_calc.append({'tr_type': tr_type, 'op_type': option_type, 'strike': strike, 'premium': premium, 'quantity': quantity})
            processed_legs += 1
        except (ValueError, KeyError, TypeError) as leg_err:
            logger.error(f"[{func_name}] Error processing metrics leg {i}: {leg_err}. Data: {leg}")
            return None # Fail fast if leg data is bad
    if processed_legs == 0: logger.error(f"[{func_name}] No valid legs processed for metrics."); return None

    # --- 3. Define Payoff Function (Helper) ---
    def _calculate_payoff_at_price(S: float, legs: List[Dict]) -> float:
        total_pnl = 0.0
        for leg in legs:
            intrinsic = 0.0; premium = leg['premium']; strike = leg['strike']
            quantity = leg['quantity']; op_type = leg['op_type']; tr_type = leg['tr_type']
            leg_prem_tot = premium * quantity
            if op_type == 'c': intrinsic = max(S - strike, 0)
            else: intrinsic = max(strike - S, 0) # op_type == 'p'
            pnl = (intrinsic * quantity - leg_prem_tot) if tr_type == 'b' else (leg_prem_tot - intrinsic * quantity)
            total_pnl += pnl
        return total_pnl

    # --- 4. Determine Theoretical Max Profit / Loss ---
    net_calls = net_long_call_qty - net_short_call_qty # Net quantity of calls (positive if net long)
    net_puts = net_long_put_qty - net_short_put_qty   # Net quantity of puts (positive if net long)
    all_strikes_unique_sorted = sorted(list(set(all_strikes_list)))

    # Determine Max Profit
    if net_calls > 0 or net_puts < 0: # Potential for infinite profit (long call or short put dominant)
        max_profit_val = np.inf
    else: # Bounded profit - check payoff at S=0 and at all strikes
        max_profit_val = payoff_at_S_equals_0
        for k in all_strikes_unique_sorted:
            max_profit_val = max(max_profit_val, _calculate_payoff_at_price(k, legs_for_payoff_calc))

    # Determine Max Loss
    if net_calls < 0 or net_puts > 0: # Potential for infinite loss (short call or long put dominant)
        max_loss_val = -np.inf
    else: # Bounded loss - check payoff at S=0 and at all strikes
        max_loss_val = payoff_at_S_equals_0
        for k in all_strikes_unique_sorted:
            max_loss_val = min(max_loss_val, _calculate_payoff_at_price(k, legs_for_payoff_calc))

    # --- 5. Breakeven Points (Using Root Finding) ---
    logger.debug(f"[{func_name}] Starting breakeven search...")
    breakeven_points_found = []
    unique_strikes_and_zero = sorted(list(set([0.0] + all_strikes_list))) # Include 0

    # Define the target function for the root finder (payoff should be zero)
    payoff_func = lambda s: _calculate_payoff_at_price(s, legs_for_payoff_calc)

    # Search intervals: (0 to first_strike), (strike_i to strike_i+1), (last_strike to infinity_proxy)
    search_intervals = []
    # Below first strike
    first_strike = unique_strikes_and_zero[1] if len(unique_strikes_and_zero) > 1 else spot_price # Use spot if no strikes
    search_intervals.append((max(0, first_strike * 0.1), first_strike * 1.05)) # Search below first strike, slightly past

    # Between strikes
    for i in range(len(all_strikes_unique_sorted) - 1):
        search_intervals.append((all_strikes_unique_sorted[i] * 0.95, all_strikes_unique_sorted[i+1] * 1.05))

    # Above last strike
    last_strike = all_strikes_unique_sorted[-1] if all_strikes_unique_sorted else spot_price
    # Define a reasonable upper search bound based on spot price
    upper_search_limit = max(last_strike * 1.5, spot_price * (PAYOFF_UPPER_BOUND_FACTOR + 0.2))
    search_intervals.append((last_strike * 0.95, upper_search_limit))

    processed_intervals = set()
    root_finder_used = "None"

    for p1_raw, p2_raw in search_intervals:
        p1 = max(0.0, p1_raw) # Ensure lower bound >= 0
        p2 = p2_raw
        interval_key = (round(p1, 4), round(p2, 4))
        if p1 >= p2 or interval_key in processed_intervals: continue
        processed_intervals.add(interval_key)

        try:
            y1 = payoff_func(p1)
            y2 = payoff_func(p2)

            # Check if payoff crosses zero within the interval
            if np.sign(y1) != np.sign(y2):
                found_be = None
                if SCIPY_AVAILABLE:
                    try:
                        # Use Brent's method (robust root finder)
                        be = brentq(payoff_func, p1, p2, xtol=1e-6, rtol=1e-6)
                        if be > 1e-6: # Check if BE is valid (positive price)
                             found_be = be
                             root_finder_used = "brentq"
                             logger.debug(f"[{func_name}] Found BE (brentq) in [{p1:.2f}, {p2:.2f}]: {be:.4f}")
                    except ValueError as brentq_err:
                        # Brentq can fail if function doesn't change sign or other issues
                        logger.debug(f"[{func_name}] Brentq failed for interval [{p1:.2f}, {p2:.2f}]: {brentq_err}. Trying interpolation.")
                    except Exception as brentq_gen_err:
                         logger.error(f"[{func_name}] Brentq unexpected error [{p1:.2f}, {p2:.2f}]: {brentq_gen_err}", exc_info=True)

                # Fallback: Linear Interpolation if SciPy fails or is unavailable
                if found_be is None and abs(y2 - y1) > 1e-9: # Avoid division by zero
                    be = p1 - y1 * (p2 - p1) / (y2 - y1)
                    # Check if interpolated point is within bounds and positive
                    if ((p1 <= be <= p2) or (p2 <= be <= p1)) and be > 1e-6:
                        found_be = be
                        root_finder_used = "interpolation"
                        logger.debug(f"[{func_name}] Found BE (interp) in [{p1:.2f}, {p2:.2f}]: {be:.4f}")

                if found_be is not None:
                     breakeven_points_found.append(found_be)

        except Exception as search_err:
            logger.error(f"[{func_name}] Error during BE search in interval [{p1:.2f}, {p2:.2f}]: {search_err}")

    # Check payoff AT strikes for exact zero touches (e.g., straddle at strike)
    zero_tolerance = 1e-4 # How close to zero counts
    for k in all_strikes_unique_sorted:
        payoff_at_k = payoff_func(k)
        if abs(payoff_at_k) < zero_tolerance:
             # Avoid adding duplicates extremely close to already found points
             is_close_to_existing = any(abs(k - be) < 0.01 for be in breakeven_points_found)
             if not is_close_to_existing:
                  breakeven_points_found.append(k)
                  logger.debug(f"[{func_name}] Found BE (strike touch): {k:.4f}")

    # Remove potential duplicates and ensure positive
    unique_be_points = sorted([p for p in list(set(round(p, 4) for p in breakeven_points_found)) if p >= 0])

    # --- Cluster Breakeven Points ---
    def cluster_points(points: List[float], tolerance_ratio: float, reference_price: float) -> List[float]:
        if not points: return []
        points.sort()
        # Use an absolute tolerance based on % of reference price, or a small minimum
        absolute_tolerance = max(reference_price * tolerance_ratio, 0.01)
        if not points: return []

        clustered_groups = [[points[0]]] # Start first group
        for p in points[1:]:
            # If current point is close to the *last point added to the current group*
            if abs(p - clustered_groups[-1][-1]) <= absolute_tolerance:
                clustered_groups[-1].append(p) # Add to current group
            else:
                clustered_groups.append([p]) # Start a new group
        # Return the average of each group, rounded
        return [round(np.mean(group), 2) for group in clustered_groups]

    breakeven_points_clustered = cluster_points(unique_be_points, BREAKEVEN_CLUSTER_GAP_PCT, spot_price)
    logger.debug(f"Breakeven raw: {unique_be_points}, clustered: {breakeven_points_clustered}, method: {root_finder_used}")


    # --- 6. Reward to Risk Ratio ---
    reward_to_risk_ratio = "N/A"; zero_threshold = 1e-9
    max_p_num = theo_max_profit # Use theoretical max profit
    max_l_num_abs = abs(theo_max_loss) if np.isfinite(theo_max_loss) else np.inf # Use absolute theoretical max loss

    if max_l_num_abs == np.inf: # Infinite risk
        reward_to_risk_ratio = "0.00" # R:R is effectively zero
    elif max_l_num_abs < zero_threshold: # Zero or negligible risk
        if max_p_num == np.inf or max_p_num > zero_threshold:
            reward_to_risk_ratio = "∞" # Infinite R:R
        else:
            reward_to_risk_ratio = "0.00" # No profit, no risk
    elif max_p_num == np.inf: # Infinite profit, finite risk
        reward_to_risk_ratio = "∞"
    else: # Both profit and loss are finite and loss > 0
        if max_p_num >= 0: # Only calculate if profit is non-negative
             reward_to_risk_ratio = round(max_p_num / max_l_num_abs, 2)
        else: # Max profit is negative (strategy always loses)
             reward_to_risk_ratio = "Loss" # Indicate guaranteed loss scenario


    # --- 7. Format Final Output ---
    max_profit_str = "∞" if max_profit_val == np.inf else format(max_profit_val, '.2f')
    max_loss_str = "-∞" if max_loss_val == -np.inf else format(max_loss_val, '.2f')
    reward_to_risk_str = str(reward_to_risk_ratio) # Already formatted or string like "∞", "Loss"

    logger.info(f"[{func_name}] Metrics calculated. MaxP: {max_profit_str}, MaxL: {max_loss_str}, BE(Clustered): {breakeven_points_clustered}, R:R: {reward_to_risk_str}")

    return {
        "calculation_inputs": { "asset": asset, "spot_price_used": round(spot_price, 2), "default_lot_size": default_lot_size, "num_legs_processed": processed_legs },
        "metrics": { "max_profit": max_profit_str, "max_loss": max_loss_str, "breakeven_points": breakeven_points_clustered, "reward_to_risk_ratio": reward_to_risk_str, "net_premium": round(total_net_premium, 2) },
        "cost_breakdown_per_leg": cost_breakdown # Premium cost/credit only
    }


# ===============================================================
# 4. Calculate Option Greeks (Keep as is - Assumes Correct)
# ===============================================================
def calculate_option_greeks(
    strategy_data: List[Dict[str, Any]],
    asset: str,
    interest_rate_pct: float = DEFAULT_INTEREST_RATE_PCT
) -> List[Dict[str, Any]]:
    """
    Calculates per-share option Greeks (scaled by 100) for each leg
    using the mibian Black-Scholes model. Requires 'iv' and 'days_to_expiry'
    in strategy_data for each leg.
    """
    func_name = "calculate_option_greeks"
    logger.info(f"Calculating SCALED PER-SHARE Greeks for {len(strategy_data)} legs, asset: {asset}, rate: {interest_rate_pct}%")
    greeks_result_list: List[Dict[str, Any]] = []

    # --- 1. Fetch Spot Price ---
    try:
        # Use consistent spot price source
        spot_price_info = get_latest_spot_price_from_db(asset)
        if spot_price_info: spot_price = spot_price_info['spot_price']
        else:
            cached_data = get_cached_option(asset)
            spot_price = _safe_get_float(cached_data.get("records", {}), "underlyingValue") if cached_data else None

        if spot_price is None or not isinstance(spot_price, (int, float)) or spot_price <= 0:
            raise ValueError(f"Spot price missing/invalid ({spot_price}) for greeks calculation")
        spot_price = float(spot_price)
        logger.debug(f"[{func_name}] Using spot price {spot_price} for asset {asset}")
    except Exception as spot_err:
        logger.error(f"[{func_name}] Error fetching spot price for {asset}: {spot_err}", exc_info=True)
        return [] # Return empty list if spot price fails

    # --- 2. Process Each Leg ---
    for i, leg_data in enumerate(strategy_data):
        try:
            # --- Validate and Extract Leg Data Safely ---
            if not isinstance(leg_data, dict):
                raise ValueError(f"Leg {i} data is not a dictionary.")

            strike_price = _safe_get_float(leg_data, 'strike')
            # DTE calculated during preparation step
            days_to_expiry = _safe_get_int(leg_data, 'days_to_expiry')
            # IV extracted during preparation step
            implied_vol_pct = _safe_get_float(leg_data, 'iv') # Expects IV as percentage (e.g., 25.5)
            option_type_flag = str(leg_data.get('op_type', '')).lower() # 'c' or 'p'
            transaction_type = str(leg_data.get('tr_type', '')).lower() # 'b' or 's'

            # --- Input Validation ---
            if strike_price is None or strike_price <= 0: raise ValueError("Missing/invalid 'strike'")
            if days_to_expiry is None or days_to_expiry < 0: raise ValueError("Missing/invalid 'days_to_expiry'")
            if implied_vol_pct is None or implied_vol_pct <= 0: raise ValueError("Missing/invalid 'iv'")
            if option_type_flag not in ['c', 'p']: raise ValueError("Invalid 'op_type'")
            if transaction_type not in ['b', 's']: raise ValueError("Invalid 'tr_type'")

            # Mibian calculation (handle near-zero DTE carefully)
            mibian_dte = max(days_to_expiry, 0.0001) # Avoid zero DTE for mibian
            mibian_inputs = [spot_price, strike_price, interest_rate_pct, mibian_dte]
            volatility_input = implied_vol_pct

            try:
                # Use specific model based on type for clarity
                if option_type_flag == 'c':
                    bs_model = mibian.BS(mibian_inputs, volatility=volatility_input)
                    delta = bs_model.callDelta
                    theta = bs_model.callTheta
                    rho = bs_model.callRho
                else: # Put
                    bs_model = mibian.BS(mibian_inputs, volatility=volatility_input)
                    delta = bs_model.putDelta
                    theta = bs_model.putTheta
                    rho = bs_model.putRho
                # Gamma and Vega are the same for calls and puts
                gamma = bs_model.gamma
                vega = bs_model.vega
            except OverflowError as math_err:
                 # Often happens with extreme inputs or very near expiry
                 logger.warning(f"[{func_name}] Mibian math error for leg {i} (asset {asset}, K={strike_price}, DTE={mibian_dte}, IV={iv}): {math_err}. Skipping greeks for this leg.")
                 continue # Skip this leg
            except Exception as mibian_err:
                 raise ValueError(f"Mibian calculation error: {mibian_err}") from mibian_err

            # --- Adjust Sign for Short Positions ---
            sign_multiplier = -1.0 if transaction_type == 's' else 1.0
            delta *= sign_multiplier
            gamma *= sign_multiplier
            theta *= sign_multiplier # Theta is usually negative (time decay cost), so selling makes it positive
            vega *= sign_multiplier
            rho *= sign_multiplier

            # --- Store PER-SHARE Greeks (NO SCALING HERE as per user logic intent) ---
            # Scaling was removed based on user's original provided `calculate_option_greeks`
            # If scaling *is* desired (e.g., delta * 100), add it back here.
            calculated_greeks = {
                'delta': round(delta, 4),
                'gamma': round(gamma, 4),
                'theta': round(theta, 4), # Theta per day
                'vega': round(vega, 4),   # Vega per 1% IV change
                'rho': round(rho, 4)    # Rho per 1% rate change
            }

            # Check for non-finite values AFTER calculations
            if any(not np.isfinite(v) for v in calculated_greeks.values()):
                logger.warning(f"[{func_name}] Skipping leg {i} for {asset} due to non-finite Greek result.")
                continue # Skip this leg

            # Append results for this leg
            greeks_result_list.append({
                'leg_index': i,
                'input_data': { # Log key inputs for debugging
                     'strike': strike_price, 'dte': days_to_expiry, 'iv': implied_vol_pct,
                     'op_type': option_type_flag, 'tr_type': transaction_type
                },
                'calculated_greeks_per_share': calculated_greeks # Unscaled, per-share
            })

        except (ValueError, KeyError, TypeError) as validation_err:
            logger.warning(f"[{func_name}] Skipping Greek calculation for leg {i} due to invalid input: {validation_err}. Leg data snapshot: strike={leg_data.get('strike')}, dte={leg_data.get('days_to_expiry')}, iv={leg_data.get('iv')}")
            continue # Skip to the next leg
        except Exception as e:
            logger.error(f"[{func_name}] Unexpected error calculating Greeks for leg {i}: {e}. Leg data: {leg_data}", exc_info=True)
            continue # Skip to the next leg

    logger.info(f"[{func_name}] Finished calculating PER-SHARE Greeks for {len(greeks_result_list)} valid legs.")
    return greeks_result_list


# ===============================================================
# 5. Fetch Stock Data (Robust yfinance handling - FIX for 404)
# ===============================================================
async def fetch_stock_data_async(stock_symbol: str) -> Optional[Dict[str, Any]]:
    """Fetches stock data using yfinance, handling indices and missing data more gracefully."""
    cache_key = f"stock_{stock_symbol}"
    cached = stock_data_cache.get(cache_key)
    if cached:
        logger.debug(f"Cache hit for stock data: {stock_symbol}")
        return cached

    logger.info(f"Fetching stock data for: {stock_symbol}")
    info = None
    h1d, h5d, h50d, h200d = None, None, None, None
    ticker_obj = None # ***** ADDED: Variable for ticker *****

    try:
        loop = asyncio.get_running_loop()
        # --- Step 1: Get Ticker object ---
        try:
             # Run synchronous yf.Ticker in executor
             ticker_obj = await loop.run_in_executor(None, yf.Ticker, stock_symbol)
             # ***** CHANGE: Check if Ticker object itself is valid *****
             # A common failure mode is getting a basic object with no real data if symbol is bad
             # We'll check later if we can get *any* price info. If not, ticker is likely invalid.
             if not ticker_obj: # Basic check
                 raise ValueError("yf.Ticker returned None or invalid object")
             logger.debug(f"yf.Ticker object created for {stock_symbol}")
        except Exception as ticker_err:
             logger.error(f"Failed to create or validate yf.Ticker object for {stock_symbol}: {ticker_err}", exc_info=False)
             return None # CRITICAL FAILURE: Cannot proceed without a valid ticker

        # --- Step 2: Fetch info (best effort) ---
        try:
            logger.debug(f"Attempting to fetch info for {stock_symbol}")
            # Run synchronous .info access in executor
            info = await loop.run_in_executor(None, getattr, ticker_obj, 'info')
            if not info:
                 logger.warning(f"Received empty 'info' dictionary for {stock_symbol}. Will rely on history.")
            else:
                 logger.debug(f"Successfully fetched 'info' dictionary for {stock_symbol}.")
        except json.JSONDecodeError as json_err:
             # This often indicates an invalid symbol or delisted stock. Consider this critical.
             logger.error(f"JSONDecodeError fetching info for {stock_symbol}: {json_err}. Symbol likely invalid or delisted.", exc_info=False)
             return None # CRITICAL FAILURE: Invalid symbol indicated by info fetch failure
        except Exception as info_err:
             logger.warning(f"Non-critical error fetching 'info' for {stock_symbol}: {info_err}", exc_info=False)
             info = {} # Ensure info is an empty dict if fetch fails, not None

        # --- Step 3: Fetch history concurrently (best effort) ---
        try:
            logger.debug(f"Attempting history fetch for {stock_symbol}")
            tasks = [
                loop.run_in_executor(None, ticker_obj.history, {"period":"1d", "interval":"1d"}),
                loop.run_in_executor(None, ticker_obj.history, {"period":"5d", "interval":"1d"}),
                loop.run_in_executor(None, ticker_obj.history, {"period":"50d", "interval":"1d"}),
                loop.run_in_executor(None, ticker_obj.history, {"period":"200d", "interval":"1d"})
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            h1d = results[0] if not isinstance(results[0], Exception) and not results[0].empty else None
            h5d = results[1] if not isinstance(results[1], Exception) and not results[1].empty else None
            h50d = results[2] if not isinstance(results[2], Exception) and not results[2].empty else None
            h200d = results[3] if not isinstance(results[3], Exception) and not results[3].empty else None
            if any(isinstance(r, Exception) for r in results):
                logger.warning(f"One or more history fetches failed for {stock_symbol}.")
            if h1d is None:
                logger.warning(f"Failed to fetch even 1d history for {stock_symbol}.")
        except Exception as hist_err:
             logger.warning(f"Unexpected error during history fetch for {stock_symbol}: {hist_err}", exc_info=False)

        # --- Step 4: Determine Price (CRITICAL) ---
        cp = None
        # Prioritize various fields from 'info'
        if isinstance(info, dict): # Ensure info is a dict before accessing
            cp = info.get("currentPrice") or info.get("regularMarketPrice") or \
                 info.get("bid") or info.get("ask") or \
                 info.get("previousClose") # Add previous close as fallback

        # Fallback to history if 'info' price missing
        if cp is None:
             if h1d is not None and 'Close' in h1d.columns:
                 cp = h1d["Close"].iloc[-1]
                 logger.debug(f"Using 1d history close price for {stock_symbol}: {cp}")
             elif h5d is not None and 'Close' in h5d.columns:
                 cp = h5d["Close"].iloc[-1]
                 logger.debug(f"Using 5d history close price for {stock_symbol}: {cp}")

        # ***** CHANGE: If NO price found after all attempts, consider it critical failure *****
        if cp is None:
             logger.error(f"CRITICAL: Could not determine any current/previous price for {stock_symbol} from info or history. Symbol might be invalid or data unavailable.")
             return None # CRITICAL FAILURE: Can't proceed without price

        # --- Step 5: Get other fields (handle missing gracefully) ---
        # Ensure info is a dict before using .get()
        safe_info = info if isinstance(info, dict) else {}
        vol = safe_info.get("volume")
        if vol is None and h1d is not None and 'Volume' in h1d.columns:
             vol = h1d["Volume"].iloc[-1]

        # Calculate MAs only if history is valid
        ma50 = h50d["Close"].mean() if h50d is not None and 'Close' in h50d.columns else None
        ma200 = h200d["Close"].mean() if h200d is not None and 'Close' in h200d.columns else None

        # Fundamental data - use None if missing
        mc = safe_info.get("marketCap")
        pe = safe_info.get("trailingPE")
        eps = safe_info.get("trailingEps")
        sec = safe_info.get("sector")
        ind = safe_info.get("industry")

        # --- Step 6: Construct final data dictionary ---
        # Include fields even if None, let the prompt builder handle N/A formatting
        data = {
            "current_price": cp, # Should always have a value if we reach here
            "volume": vol,
            "moving_avg_50": ma50,
            "moving_avg_200": ma200,
            "market_cap": mc,
            "pe_ratio": pe,
            "eps": eps,
            "sector": sec,
            "industry": ind
        }

        stock_data_cache[cache_key] = data
        logger.info(f"Successfully processed stock data for {stock_symbol}. Price: {cp}")
        return data # RETURN THE DICTIONARY, even with missing non-critical fields

    except Exception as e:
        # Catch any other unexpected errors during the overall process
        logger.error(f"Unexpected error in fetch_stock_data_async for {stock_symbol}: {e}", exc_info=True)
        return None

# ===============================================================
# 6. Fetch News (Robust Scraping - FIX for Reliability)
# ===============================================================
async def fetch_latest_news_async(asset: str) -> List[Dict[str, str]]:
    """Fetches latest news using feedparser from Google News RSS."""
    cache_key = f"news_{asset.upper()}"
    cached = news_cache.get(cache_key)
    if cached:
        logger.debug(f"Cache hit for news: {asset}")
        return cached

    logger.info(f"Fetching news for: {asset} using feedparser/Google News")
    # Construct Google News RSS URL (India, English)
    # Adding "stock" or "market" might help filter results
    search_term = f"{asset} stock market"
    # URL encoding is generally handled by libraries, but be mindful of special chars in asset if needed
    gnews_url = f"https://news.google.com/rss/search?q={search_term}&hl=en-IN&gl=IN&ceid=IN:en"

    news_list = []
    max_news = 5 # Fetch a few headlines

    try:
        loop = asyncio.get_running_loop()
        # feedparser is synchronous, run in executor
        logger.debug(f"Parsing Google News RSS feed: {gnews_url}")
        feed_data = await loop.run_in_executor(None, feedparser.parse, gnews_url)

        if feed_data.bozo: # Check if feedparser encountered errors
             # Log the specific error if available
             bozo_exception = feed_data.get('bozo_exception', 'Unknown parsing error')
             logger.warning(f"Feedparser encountered an error for {asset}: {bozo_exception}")
             # Return error state, but don't raise exception to break asset loading
             return [{"headline": f"Error parsing news feed for {asset}.", "summary": str(bozo_exception), "link": "#"}]

        if not feed_data.entries:
            logger.warning(f"No news entries found in Google News RSS feed for query: {search_term}")
            return [{"headline": f"No recent news found for {asset}.", "summary": "", "link": "#"}]

        # Process feed entries
        for entry in feed_data.entries[:max_news]:
            headline = entry.get('title', 'No Title')
            link = entry.get('link', '#')
            # Summary might be in 'summary' or 'description' field
            summary = entry.get('summary', entry.get('description', 'No summary available.'))
            # Basic cleaning of summary (remove potential HTML) - more robust parsing needed if required
            summary_soup = BeautifulSoup(summary, "html.parser")
            cleaned_summary = summary_soup.get_text(strip=True)

            # Filter out potential non-news items if possible (e.g., based on title)
            # Example: skip if title contains "is trading" or similar patterns? (can be brittle)
            # if " is trading " in headline.lower(): continue

            news_list.append({
                "headline": headline,
                "summary": cleaned_summary,
                "link": link
            })

        if not news_list: # Should be covered by 'if not feed_data.entries', but for safety
             return [{"headline": f"No relevant news found for {asset}.", "summary": "", "link": "#"}]

        news_cache[cache_key] = news_list # Cache the result
        logger.info(f"Successfully parsed {len(news_list)} news items for {asset} via feedparser.")
        return news_list

    except Exception as e:
        logger.error(f"Error fetching/parsing news feed for {asset} using feedparser: {e}", exc_info=True)
        # Return error state
        return [{"headline": f"Error fetching news for {asset}.", "summary": str(e), "link": "#"}]


# ===============================================================
# 7. Build Analysis Prompt (Keep as is - Relies on fixed data fetch)
# ===============================================================
def build_stock_analysis_prompt(stock_symbol: str, stock_data: dict, latest_news: list) -> str:
    """Generates structured prompt for LLM analysis, handling potentially missing data."""
    func_name = "build_stock_analysis_prompt"
    logger.debug(f"[{func_name}] Building structured prompt for {stock_symbol}")

    # --- Formatting Helper ---
    def fmt(v, p="₹", s="", pr=2, na="N/A"):
        if v is None or v == 'N/A': return na
        if isinstance(v,(int,float)):
            try:
                # Handle large numbers (Crores, Lakhs)
                if abs(v) >= 1e7: return f"{p}{v/1e7:.{pr}f} Cr{s}"
                if abs(v) >= 1e5: return f"{p}{v/1e5:.{pr}f} L{s}"
                # Standard formatting
                return f"{p}{v:,.{pr}f}{s}"
            except Exception: return str(v) # Fallback
        return str(v) # Return non-numeric as string

    # --- Prepare Data Sections ---
    price = stock_data.get('current_price') # Should always exist if fetch succeeded
    ma50 = stock_data.get('moving_avg_50')
    ma200 = stock_data.get('moving_avg_200')
    volume = stock_data.get('volume')
    market_cap = stock_data.get('market_cap')
    pe_ratio = stock_data.get('pe_ratio')
    eps = stock_data.get('eps')
    sector = stock_data.get('sector', 'N/A')
    industry = stock_data.get('industry', 'N/A')

    # Technical Context String (handles missing MAs)
    trend = "N/A"; support = "N/A"; resistance = "N/A"
    ma_available = ma50 is not None and ma200 is not None
    if price and ma_available:
        support_levels = sorted([lvl for lvl in [ma50, ma200] if lvl is not None and lvl < price], reverse=True)
        resistance_levels = sorted([lvl for lvl in [ma50, ma200] if lvl is not None and lvl >= price])
        support = " / ".join([fmt(lvl) for lvl in support_levels]) if support_levels else "Below Key MAs"
        resistance = " / ".join([fmt(lvl) for lvl in resistance_levels]) if resistance_levels else "Above Key MAs"
        # Trend description based on price vs MAs
        if price > ma50 > ma200: trend = "Strong Uptrend (Price > 50MA > 200MA)"
        elif price > ma50 and price > ma200: trend = "Uptrend (Price > Both MAs)"
        elif price < ma50 < ma200: trend = "Strong Downtrend (Price < 50MA < 200MA)"
        elif price < ma50 and price < ma200: trend = "Downtrend (Price < Both MAs)"
        elif ma50 > price > ma200 : trend = "Sideways/Consolidating (Between 200MA and 50MA)"
        elif ma200 > price > ma50 : trend = "Sideways/Consolidating (Between 50MA and 200MA)"
        elif ma50 > ma200: trend = "Sideways/Uptrend Context (Price vs 50MA: %s)" % ('Above' if price > ma50 else 'Below')
        else: trend = "Sideways/Downtrend Context (Price vs 50MA: %s)" % ('Above' if price > ma50 else 'Below')
    elif price and ma50: # Only 50MA available
        support = fmt(ma50) if price > ma50 else "N/A (Below 50MA)"
        resistance = fmt(ma50) if price <= ma50 else "N/A (Above 50MA)"
        trend = "Above 50MA" if price > ma50 else "Below 50MA"
    else: # No MAs available
        trend = "Trend Unknown (MA data unavailable)"

    tech_context = f"Price: {fmt(price)}, 50D MA: {fmt(ma50)}, 200D MA: {fmt(ma200)}, Trend Context: {trend}, Key Levels (from MAs): Support near {support}, Resistance near {resistance}. Volume (Approx 1d): {fmt(volume, p='', pr=0)}"

    # Fundamental Context String (handles missing data)
    fund_context = f"Market Cap: {fmt(market_cap, p='')}, P/E Ratio: {fmt(pe_ratio, p='', s='x')}, EPS: {fmt(eps)}, Sector: {fmt(sector, p='')}, Industry: {fmt(industry, p='')}"
    pe_comparison_note = ""
    if pe_ratio is not None:
        pe_comparison_note = f"Note: P/E ({fmt(pe_ratio, p='', s='x')}) should be compared to '{fmt(industry, p='')}' industry peers and historical averages for valuation context (peer data not provided)."
    elif market_cap is None and pe_ratio is None and eps is None: # Likely an index
         fund_context = "N/A (Index - Standard fundamental ratios like P/E, Market Cap, EPS do not apply directly)."
         pe_comparison_note = "N/A for index."
    else:
         pe_comparison_note = "Note: P/E data unavailable for comparison."


    # News Context String (handles potential errors)
    news_formatted = []
    if latest_news and not ("Error fetching news" in latest_news[0].get("headline", "") or "No recent news found" in latest_news[0].get("headline", "")):
        for n in latest_news[:3]: # Max 3 news items
            headline = n.get('headline','N/A')
            # Sanitize summary for prompt (optional, safer for LLM)
            summary = n.get('summary','N/A').replace('"', "'").replace('{', '(').replace('}', ')')
            link = n.get('link', '#')
            news_formatted.append(f'- [{headline}]({link}): {summary}')
        news_context = "\n".join(news_formatted)
    else: # Handle error case or no news case
        news_context = f"- {latest_news[0].get('headline', 'No recent news summaries found.')}" if latest_news else "- No recent news summaries found."


    # --- Construct the Structured Prompt ---
    # (Using the same well-structured prompt template)
    prompt = f"""
Analyze the stock/index **{stock_symbol}** based *only* on the provided data snapshots. Use clear headings and bullet points. Acknowledge missing data where applicable.

**Analysis Request:**

1.  **Executive Summary:**
    *   Provide a brief (2-3 sentence) overall takeaway combining technical posture (price vs. MAs), key fundamental indicators (if available), and recent news sentiment. State the implied short-term bias (e.g., Bullish, Bearish, Neutral, Cautious).

2.  **Technical Analysis:**
    *   **Trend & Momentum:** Describe the current trend based on the price vs. 50D and 200D Moving Averages (if available). Is the trend established or potentially changing? Comment on the provided volume figure (relative context might be missing).
        *   *Data:* {tech_context}
    *   **Support & Resistance:** Identify potential support and resistance levels based *only* on the 50D and 200D MAs provided. State if levels are unclear due to missing MA data.
    *   **Key Technical Observations:** Note any significant technical patterns *implied* by the data (e.g., price extended from MAs, consolidation near MAs, MA crossovers if evident).

3.  **Fundamental Snapshot:**
    *   **Company Size & Profitability (if applicable):** Based on Market Cap (if available), classify the company size. Comment on EPS (if available) as an indicator of profitability. Acknowledge if this section is not applicable (e.g., for an index).
    *   **Valuation (if applicable):** Discuss the P/E Ratio (if available). {pe_comparison_note}
        *   *Data:* {fund_context}
    *   **Sector/Industry Context:** State the sector and industry (if available). Briefly mention general characteristics if widely known, but prioritize provided data.

4.  **Recent News Sentiment:**
    *   **Sentiment Assessment:** Summarize the general sentiment (Positive, Negative, Neutral, Mixed) conveyed by the provided news headlines/summaries. State if news fetch failed.
    *   **Potential News Impact:** Briefly state how this news *might* influence near-term price action, considering the sentiment.
        *   *News Data:*
{news_context}

5.  **Outlook & Considerations:**
    *   **Consolidated Outlook:** Reiterate the short-term bias (Bullish/Bearish/Neutral/Uncertain) based on the synthesis of the above points. Qualify the outlook based on data availability/quality.
    *   **Key Factors to Monitor:** What are the most important technical levels (from MAs) or potential catalysts (from news/fundamentals provided) to watch?
    *   **Identified Risks (from Data):** What potential risks are directly suggested by the provided data (e.g., price below key MA, high P/E without context, negative news, missing data)?
    *   **General Option Strategy Angle:** Based *only* on the derived bias (Bullish/Bearish/Neutral) and acknowledging IV/risk tolerance are unknown, suggest *general types* of option strategies that align (e.g., Bullish -> Long Calls/Spreads; Bearish -> Long Puts/Spreads; Neutral -> Credit Spreads/Iron Condors). **Do NOT suggest specific strikes or expiries.** State if bias is too uncertain for a clear angle.

**Disclaimer:** This analysis is generated based on limited, potentially delayed data points and recent news summaries. It is NOT financial advice. Verify all information and conduct thorough independent research before making any trading or investment decisions. Market conditions change rapidly.
"""
    logger.debug(f"[{func_name}] Generated prompt for {stock_symbol}")
    return prompt


# ===============================================================
# Helper to get latest spot price from DB (used by multiple functions)
# ===============================================================
def get_latest_spot_price_from_db(asset: str, max_age_minutes: int = 5) -> Optional[Dict[str, Any]]:
    """Queries the DB for the latest spot price updated by the background thread."""
    sql = """
        SELECT last_spot_price, last_spot_update_time
        FROM option_data.assets
        WHERE asset_name = %s
          AND last_spot_update_time >= NOW() - INTERVAL %s MINUTE
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(sql, (asset, max_age_minutes))
                result = cursor.fetchone()
                if result and result.get("last_spot_price") is not None:
                    price = _safe_get_float(result, "last_spot_price")
                    ts = result.get("last_spot_update_time")
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts)
                    logger.debug(f"Found recent spot price in DB for {asset}: {price} at {ts_str}")
                    return {"spot_price": price, "timestamp": ts_str}
                else:
                    logger.debug(f"No recent spot price (<{max_age_minutes}min) found in DB for {asset}")
                    return None
    except (ConnectionError, mysql.connector.Error) as db_err:
        logger.error(f"DB Error fetching latest spot price for {asset}: {db_err}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching spot price from DB for {asset}: {e}", exc_info=True)
        return None


# ===============================================================
# API Endpoints (Modified Spot Price, Analysis)
# ===============================================================

@app.get("/health", tags=["Status"])
async def health_check():
     db_status = "unknown"
     try:
         with get_db_connection() as conn: # Use sync context manager
              with conn.cursor() as cursor:
                   cursor.execute("SELECT 1") # Sync execute
                   db_status = "connected" if cursor.fetchone() else "query_failed"
     except Exception as e:
         logger.error(f"Health check DB error: {e}", exc_info=False) # Less verbose log
         db_status = f"error: {type(e).__name__}"
     return {"status": "ok", "database": db_status, "nse_client": "initialized" if n else "failed"}


@app.get("/get_assets", tags=["Data"])
async def get_assets():
    logger.info("--- GET /get_assets Endpoint START ---")
    asset_names = []
    try:
        with get_db_connection() as conn: # Sync context manager
            with conn.cursor(dictionary=True) as cursor: # Use dictionary cursor
                sql = "SELECT asset_name FROM option_data.assets ORDER BY asset_name ASC"
                logger.debug(f"/get_assets: Executing SQL: {sql}")
                cursor.execute(sql) # Sync execute
                results = cursor.fetchall() # Sync fetchall
                if results:
                    asset_names = [row["asset_name"] for row in results if "asset_name" in row]
                    logger.info(f"/get_assets: Fetched {len(asset_names)} asset names.")
                else:
                    logger.warning("/get_assets: Query returned no results.")
        return {"assets": asset_names}
    except (ConnectionError, mysql.connector.Error) as db_err:
         logger.error(f"Database error in /get_assets: {db_err}", exc_info=True)
         raise HTTPException(status_code=500, detail="Database error fetching assets.")
    except Exception as e:
        logger.error(f"Unexpected error in /get_assets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected server error.")
    finally:
        logger.info("--- GET /get_assets Endpoint END ---")


@app.get("/expiry_dates", tags=["Data"])
async def get_expiry_dates(asset: str = Query(...)):
    logger.info(f"Request received for expiry dates: Asset={asset}")
    sql = """ SELECT DISTINCT DATE_FORMAT(e.expiry_date, '%Y-%m-%d') AS expiry_date_str
              FROM option_data.expiries e JOIN option_data.assets a ON e.asset_id = a.id
              WHERE a.asset_name = %s AND e.expiry_date >= CURDATE() ORDER BY expiry_date_str ASC; """
    expiries = []
    try:
         with get_db_connection() as conn: # Sync context manager
             with conn.cursor(dictionary=True) as cursor:
                 cursor.execute(sql, (asset,)) # Sync execute
                 results = cursor.fetchall() # Sync fetchall
                 expiries = [row["expiry_date_str"] for row in results if "expiry_date_str" in row]
    except (ConnectionError, mysql.connector.Error) as db_err:
        logger.error(f"DB Error fetching expiry dates for {asset}: {db_err}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error retrieving expiry data")
    except Exception as e:
         logger.error(f"Unexpected error fetching expiry dates for {asset}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Internal server error")

    if not expiries: logger.warning(f"No future expiry dates found for asset: {asset}")
    return {"expiry_dates": expiries}


@app.get("/get_news", tags=["Data"])
async def get_news_endpoint(asset: str = Query(...)):
    """Fetches the latest news headlines and summaries for a given asset."""
    endpoint_name = "get_news_endpoint"
    logger.info(f"[{endpoint_name}] Request received for news: Asset={asset}")
    asset_upper = asset.strip().upper()
    if not asset_upper:
        raise HTTPException(status_code=400, detail="Asset name required.")

    try:
        # Use the existing robust async news fetching function
        news_items = await fetch_latest_news_async(asset_upper)
        # Check if the fetch itself indicated an error internally
        if news_items and "Error fetching news" in news_items[0].get("headline", ""):
            logger.warning(f"[{endpoint_name}] News fetch for {asset} returned an error state.")
            # Return a slightly different structure or status? For now, return the error item.
            # return JSONResponse(status_code=503, content={"news": news_items})
            # Or just return the error message as success
            return {"news": news_items}
        elif not news_items:
             logger.warning(f"[{endpoint_name}] No news items found for {asset}.")
             # Return an empty list or a specific message
             return {"news": [{"headline": f"No recent news found for {asset}.", "summary": "", "link": "#"}]}
        else:
            logger.info(f"[{endpoint_name}] Successfully fetched {len(news_items)} news items for {asset}")
            return {"news": news_items}
    except Exception as e:
        logger.error(f"[{endpoint_name}] Unexpected error fetching news for {asset}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Server error fetching news for {asset}.")

        


@app.get("/get_option_chain", tags=["Data"])
async def get_option_chain(asset: str = Query(...), expiry: str = Query(...)):
    """
    API endpoint to fetch option chain data for a given asset and expiry
    from the database. Uses synchronous DB access with safe type conversion.
    """
    endpoint_name = "/get_option_chain" # For logging context
    logger.info(f"[{endpoint_name}] Request received: Asset={asset}, Expiry={expiry}")

    # --- Validate Input ---
    try:
        # Validate expiry date format
        datetime.strptime(expiry, '%Y-%m-%d')
    except ValueError:
        logger.warning(f"[{endpoint_name}] Invalid expiry date format received: {expiry}")
        raise HTTPException(status_code=400, detail="Invalid expiry date format. Use YYYY-MM-DD.")

    if not asset:
        logger.warning(f"[{endpoint_name}] Missing asset name in request.")
        raise HTTPException(status_code=400, detail="Asset name is required.")

    # --- Database Interaction ---
    sql = """
        SELECT
            oc.strike_price, oc.option_type, oc.open_interest, oc.change_in_oi,
            oc.implied_volatility, oc.last_price, oc.total_traded_volume,
            oc.bid_price, oc.bid_qty, oc.ask_price, oc.ask_qty
        FROM option_data.option_chain AS oc
        JOIN option_data.assets AS a ON oc.asset_id = a.id
        JOIN option_data.expiries AS e ON oc.expiry_id = e.id
        WHERE a.asset_name = %(asset_name)s AND e.expiry_date = %(expiry_date)s
        ORDER BY oc.strike_price ASC;
        """
    params = { "asset_name": asset, "expiry_date": expiry }
    rows = []
    conn = None # Initialize connection variable

    try:
        # Use SYNCHRONOUS DB Connection (ensure get_db_connection handles context)
        logger.debug(f"[{endpoint_name}] Attempting DB connection for {asset}/{expiry}.")
        with get_db_connection() as conn: # Uses the synchronous context manager from database.py
            if conn is None:
                # Log specific error if connection pool fails
                logger.error(f"[{endpoint_name}] Failed to get DB connection from pool.")
                raise ConnectionError("Failed to get DB connection for option chain.") # Raise specific error

            logger.debug(f"[{endpoint_name}] Got DB connection. Current DB: '{getattr(conn, 'database', 'N/A')}'")
            with conn.cursor(dictionary=True) as cursor:
                logger.debug(f"[{endpoint_name}] Executing SQL with params: {params}")
                cursor.execute(sql, params) # SYNC execute
                rows = cursor.fetchall() # SYNC fetchall
                logger.info(f"[{endpoint_name}] Fetched {len(rows)} rows from DB for {asset}/{expiry}.")
                if len(rows) > 0:
                    # Log only a few keys from the first row to keep logs concise
                    sample_row_keys = list(rows[0].keys())[:5] # Get first 5 keys
                    sample_row_preview = {k: rows[0].get(k) for k in sample_row_keys}
                    logger.debug(f"[{endpoint_name}] First row example preview: {sample_row_preview}")
                # No await needed for synchronous operations

    except (mysql.connector.Error, ConnectionError) as db_err: # Catch specific DB/Connection errors
        logger.error(f"[{endpoint_name}] Database/Connection error for {asset}/{expiry}: {db_err}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error fetching option chain.")
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"[{endpoint_name}] Unexpected error fetching DB data for {asset}/{expiry}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error fetching option chain.")
    # `with get_db_connection()` ensures connection is released/closed

    # --- Process Rows & Build Response ---
    if not rows:
         logger.warning(f"[{endpoint_name}] No option chain data found in DB for asset '{asset}' and expiry '{expiry}'. Returning empty structure.")
         # Return the expected structure but with an empty inner object
         return {"option_chain": {}}

    # Use defaultdict for easier handling of call/put assignment
    option_chain = defaultdict(lambda: {"call": None, "put": None})
    processed_row_count = 0
    skipped_row_count = 0

    for row in rows:
        # Basic validation of the row structure
        if not isinstance(row, dict):
             logger.warning(f"[{endpoint_name}] Skipping processing non-dictionary row: {row}")
             skipped_row_count += 1
             continue

        strike_raw = row.get("strike_price")
        opt_type = row.get("option_type")

        # Validate and convert strike price
        safe_strike = None
        if strike_raw is not None:
            try:
                safe_strike = float(strike_raw)
            except (ValueError, TypeError):
                # Log invalid strike and skip row
                logger.warning(f"[{endpoint_name}] Skipping row with invalid strike price format: {strike_raw}")
                skipped_row_count += 1
                continue

        # Validate essential fields
        if safe_strike is None or opt_type not in ("CE", "PE"):
            logger.warning(f"[{endpoint_name}] Skipping row with missing/invalid strike or option type. Strike Raw: {strike_raw}, Type: {opt_type}")
            skipped_row_count += 1
            continue

        # Use the safe helper functions for reliable type conversion and NULL handling
        # These helpers should return None if conversion fails or DB value is NULL
        data_for_type = {
            "last_price": _safe_get_float(row, "last_price"),
            "open_interest": _safe_get_int(row, "open_interest"),
            "change_in_oi": _safe_get_int(row, "change_in_oi"),
            "implied_volatility": _safe_get_float(row, "implied_volatility"),
            "volume": _safe_get_int(row, "total_traded_volume"), # Mapped from DB column name
            "bid_price": _safe_get_float(row, "bid_price"),
            "bid_qty": _safe_get_int(row, "bid_qty"),
            "ask_price": _safe_get_float(row, "ask_price"),
            "ask_qty": _safe_get_int(row, "ask_qty"),
        }
        # Optional: Remove keys with None values if frontend prefers cleaner objects
        # data_for_type = {k: v for k, v in data_for_type.items() if v is not None}

        # Assign the processed data to the correct structure
        if opt_type == "PE":
            option_chain[safe_strike]["put"] = data_for_type
        elif opt_type == "CE":
            option_chain[safe_strike]["call"] = data_for_type

        processed_row_count += 1 # Count successfully processed DB rows

    logger.info(f"[{endpoint_name}] Successfully processed {processed_row_count}/{len(rows)} fetched rows into chain structure (skipped: {skipped_row_count}).")

    # --- Final Logging and Return ---
    final_chain_dict = dict(option_chain) # Convert defaultdict to regular dict for JSON response

    if not final_chain_dict and len(rows) > 0:
         # This case means rows were fetched but ALL failed processing
         logger.warning(f"[{endpoint_name}] Processed {len(rows)} rows but resulted in empty option_chain dict for {asset}/{expiry}.")
         # Still return the empty structure
         return {"option_chain": {}}
    elif not final_chain_dict:
        # This case means no rows were fetched initially
         logger.debug(f"[{endpoint_name}] Returning empty option_chain dictionary (no rows fetched).")
         return {"option_chain": {}}
    else:
        # Log a sample before returning valid data
        log_sample_strikes = list(final_chain_dict.keys())[:2] # Log first 2 strikes
        sample_data_log = {k: final_chain_dict.get(k) for k in log_sample_strikes}
        logger.debug(f"[{endpoint_name}] Returning processed option_chain with {len(final_chain_dict)} strikes. Sample data: {sample_data_log}")
        return {"option_chain": final_chain_dict}


@app.get("/get_spot_price", response_model=SpotPriceResponse, tags=["Data"])
async def get_spot_price(asset: str = Query(...)):
    """
    Gets the spot price ONLY from the cached/live option chain data.
    (Database check removed).
    """
    logger.info(f"Received request for spot price (Cache/Live Only): Asset={asset}")
    spot = None
    timestamp = None
    source = "cache/live_fallback" # Source will always be this now

    # 1. Fetch data using the cache/live function
    logger.debug(f"Fetching spot price via get_cached_option for {asset}...")
    option_cache_data = get_cached_option(asset) # This uses its own cache/live fallback logic

    # 2. Extract spot price from the returned data
    if option_cache_data and isinstance(option_cache_data.get("records"), dict):
        # Use the safe helper to extract and convert the value
        spot = _safe_get_float(option_cache_data["records"], "underlyingValue")
        if spot is not None:
            # Use current time as timestamp since we don't have DB timestamp anymore
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Approximate time
            logger.info(f"Using spot price from Cache/Live Fallback for {asset}: {spot}")
        else:
            logger.warning(f"UnderlyingValue not found or invalid within cached data records for {asset}.")
            source = "cache/live_fallback (value missing)"
    else:
        logger.warning(f"Failed to get valid option chain data from get_cached_option for {asset}.")
        source = "cache/live_fallback (fetch failed)"


    # 3. Check if we got a valid price
    if spot is None:
        logger.error(f"Failed to retrieve spot price for {asset} from Cache/Live fetch.")
        # Keep returning 404 if the primary method fails
        raise HTTPException(status_code=404, detail=f"Spot price could not be determined for {asset} via cache/live.")

    # 4. Final validation of the retrieved price
    if not isinstance(spot, (int, float)) or spot <= 0:
         logger.error(f"Retrieved invalid spot price type or value for {asset}: {spot} (Type: {type(spot)})")
         # Raise 500 for invalid data format from the source
         raise HTTPException(status_code=500, detail=f"Invalid spot price data retrieved for {asset}.")

    # 5. Return the result
    logger.info(f"Returning spot price {spot} for {asset} (Source: {source})")
    # Round the float before returning
    return {"spot_price": round(float(spot), 2), "timestamp": timestamp}


# --- In-Memory Strategy Endpoints (Keep commented out - Dev Only) ---
# @app.post("/add_strategy", tags=["Strategy (In-Memory - Dev Only)"], include_in_schema=False)
# async def add_strategy_in_memory(position: PositionInput): ...
# @app.get("/get_strategies", tags=["Strategy (In-Memory - Dev Only)"], include_in_schema=False)
# async def get_strategies_in_memory(): ...


# --- Debug Endpoint for Background Task ---
@app.post("/debug/set_selected_asset", tags=["Debug"], include_in_schema=False)
async def set_selected_asset_endpoint(request: DebugAssetSelectRequest, background_tasks: BackgroundTasks):
    global selected_asset
    asset_name = request.asset.strip().upper()
    if not asset_name:
        raise HTTPException(status_code=400, detail="Asset name cannot be empty.")
    logger.warning(f"DEBUG: Setting globally selected asset for background DB updates to: {asset_name}")
    selected_asset = asset_name
    # Optionally trigger an immediate background update for this asset
    # Note: This runs fetch_and_update_single_asset_data in the background,
    # it doesn't wait for it to complete before returning the response.
    background_tasks.add_task(fetch_and_update_single_asset_data, asset_name)
    logger.warning(f"DEBUG: Added background task to immediately update DB for {asset_name}")
    return {
        "message": f"Global selected asset set to {asset_name}. Background task will target this asset for DB updates.",
        "task_added": f"Background task added to update {asset_name}."
    }


# --- Main Analysis & Payoff Endpoint ---
@app.post("/get_payoff_chart", tags=["Analysis & Payoff"])
async def get_payoff_chart_endpoint(request: PayoffRequest):
    """
    Calculates strategy metrics, Greeks, taxes, and generates a payoff chart.
    Handles data preparation and concurrent execution of calculations.
    """
    endpoint_name = "get_payoff_chart_endpoint"
    asset = request.asset; strategy_input = request.strategy # Use different name from loop var
    logger.info(f"[{endpoint_name}] Request received: Asset={asset}, Legs={len(strategy_input)}")
    if not asset or not strategy_input:
        raise HTTPException(status_code=400, detail="Missing asset or strategy legs")

    # --- Step 1: Fetch Initial Prerequisite Data (Spot Price, Lot Size) ---
    try:
        start_prereq_time = time.monotonic()
        # Get spot price (use the reliable /get_spot_price logic)
        spot_response = await get_spot_price(asset) # Call the fixed endpoint
        spot_price = spot_response.spot_price
        logger.info(f"[{endpoint_name}] Prerequisite Spot Price: {spot_price}")

        default_lot_size = get_lot_size(asset)
        if not default_lot_size or default_lot_size <= 0 :
            raise ValueError(f"Lot size missing or invalid ({default_lot_size}) for {asset}")
        logger.info(f"[{endpoint_name}] Prerequisite Lot Size: {default_lot_size}")
        prereq_duration = time.monotonic() - start_prereq_time
        logger.info(f"[{endpoint_name}] Prerequisites fetched in {prereq_duration:.3f}s")

    except HTTPException as http_err:
        logger.error(f"[{endpoint_name}] Failed prerequisite HTTP fetch for {asset}: {http_err.detail}")
        raise http_err # Re-raise FastAPI exception
    except ValueError as val_err:
         logger.error(f"[{endpoint_name}] Failed prerequisite data validation for {asset}: {val_err}")
         raise HTTPException(status_code=404, detail=str(val_err)) # 404 for missing essential data
    except Exception as e:
        logger.error(f"[{endpoint_name}] Unexpected server error fetching prerequisites for {asset}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Server error fetching initial data: {e}")

    # --- Step 2: Prepare Strategy Data (Validate, Add IV, DTE) ---
    prepared_strategy_data = []
    today = date.today()
    prep_errors = []
    for i, leg_input in enumerate(strategy_input):
        try:
            strike = _safe_get_float(leg_input, 'strike_price')
            premium = _safe_get_float(leg_input, 'option_price')
            lots_abs = _safe_get_int(leg_input, 'lots')
            opt_type_req = str(leg_input.option_type).upper() # CE/PE
            tr_type_req = str(leg_input.tr_type).lower() # b/s
            expiry_str = str(leg_input.expiry_date)

            # Validate core types and values
            if strike is None or strike <= 0: raise ValueError("Invalid strike price")
            if premium is None or premium < 0: raise ValueError("Invalid option price") # Allow 0 premium
            if lots_abs is None or lots_abs <= 0: raise ValueError("Invalid lots")
            if opt_type_req not in ("CE", "PE"): raise ValueError("Invalid option type")
            if tr_type_req not in ("b", "s"): raise ValueError("Invalid transaction type")

            # Validate and calculate Date-to-Expiry (DTE)
            try: expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            except ValueError: raise ValueError("Invalid expiry date format (use YYYY-MM-DD)")
            days_to_expiry = (expiry_dt - today).days
            # Allow DTE=0 for expiry day, but calculations might be sensitive
            if days_to_expiry < 0: raise ValueError("Expiry date cannot be in the past")

            # Determine Lot Size
            leg_lot_size = default_lot_size
            if leg_input.lot_size is not None:
                temp_ls = _safe_get_int(leg_input, 'lot_size')
                if temp_ls is not None and temp_ls > 0: leg_lot_size = temp_ls
                else: logger.warning(f"Ignoring invalid leg-specific lot size ({leg_input.lot_size}) for leg {i+1}, using default {default_lot_size}")

            # Extract Implied Volatility (IV) - Critical for Greeks
            # Use the expiry format required by extract_iv (YYYY-MM-DD)
            iv = extract_iv(asset, strike, expiry_str, opt_type_req)
            if iv is None or iv <= 0:
                # IV is essential for Greeks. If missing, we can still calc payoff/tax, but maybe warn/fail?
                # For now, let's add a placeholder and log a clear warning. Greeks will likely fail for this leg.
                logger.warning(f"[{endpoint_name}] Missing or invalid IV (<=0) for leg {i+1}: {asset} {expiry_str} {strike} {opt_type_req}. Greeks may be inaccurate.")
                # Decide: Use default IV, skip leg for greeks, or fail? Let's use 0 for now, Greeks func should handle it.
                iv = 0.0 # Placeholder - Greeks calc should ideally skip legs with IV<=0

            # Append fully prepared data for calculations
            prepared_strategy_data.append({
                "op_type": "c" if opt_type_req == "CE" else "p", # Use 'c'/'p' internally
                "strike": strike,
                "tr_type": tr_type_req, # 'b' or 's'
                "op_pr": premium,
                "lot": lots_abs,
                "lot_size": leg_lot_size,
                "iv": float(iv), # Store extracted/placeholder IV
                "days_to_expiry": days_to_expiry,
                "expiry_date_str": expiry_str, # Keep original string if needed
            })
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"[{endpoint_name}] Invalid data provided for leg {i+1}: {e}. Input: {leg_input.dict()}")
            prep_errors.append(f"Leg {i+1}: {e}")
        except Exception as e:
             logger.error(f"[{endpoint_name}] Unexpected error processing leg {i+1}: {e}", exc_info=True)
             prep_errors.append(f"Leg {i+1}: Unexpected error")

    if prep_errors: # If any legs failed preparation
        raise HTTPException(status_code=400, detail=f"Invalid strategy leg data: {'; '.join(prep_errors)}")
    if not prepared_strategy_data: # If input was empty or all legs failed
        raise HTTPException(status_code=400, detail="No valid strategy legs provided after preparation.")


    # --- Step 3: Perform Calculations Concurrently ---
    logger.info(f"[{endpoint_name}] Starting concurrent calculations for {len(prepared_strategy_data)} legs...")
    start_calc_time = time.monotonic()
    results = {} # Dictionary to store results

    try:
        # --- Run Calculations ---
        # Metrics calculation needs to run first as chart relies on its output
        logger.debug(f"[{endpoint_name}] Calculating metrics...")
        metrics_result = await asyncio.to_thread(calculate_strategy_metrics, prepared_strategy_data, asset)
        if metrics_result is None:
             logger.error(f"[{endpoint_name}] Metrics calculation failed. Cannot generate chart accurately.")
             # Decide whether to fail completely or proceed without metrics/chart adjustments
             raise ValueError("Strategy metrics calculation failed, cannot proceed.")
        results["metrics"] = metrics_result
        logger.debug(f"[{endpoint_name}] Metrics calculated.")

        # Now run chart, tax, and greeks concurrently
        logger.debug(f"[{endpoint_name}] Starting concurrent chart, tax, greeks...")
        tasks = {
            # Pass metrics result to chart function
            "chart": asyncio.to_thread(generate_payoff_chart_matplotlib, prepared_strategy_data, asset, metrics_result),
            "tax": asyncio.to_thread(calculate_option_taxes, prepared_strategy_data, asset),
            "greeks": asyncio.to_thread(calculate_option_greeks, prepared_strategy_data, asset)
        }

        # Await remaining tasks
        # Use return_exceptions=True to get results even if one task fails (e.g., tax fails but chart/greeks succeed)
        task_values = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Assign results, checking for exceptions
        results["chart"] = task_values[0] if not isinstance(task_values[0], Exception) else None
        results["tax"] = task_values[1] if not isinstance(task_values[1], Exception) else None
        results["greeks"] = task_values[2] if not isinstance(task_values[2], Exception) else [] # Default to empty list on error

        # Log any exceptions that occurred during gather
        for i, name in enumerate(tasks.keys()):
            if isinstance(task_values[i], Exception):
                 logger.error(f"[{endpoint_name}] Error during concurrent calculation of '{name}': {task_values[i]}", exc_info=task_values[i])

        # Check for critical failures (chart generation)
        if results["chart"] is None:
             logger.error(f"[{endpoint_name}] Payoff chart generation failed.")
             # Allow response without chart, but include error? Or raise 500?
             # Let's allow partial success for now, frontend can handle missing chart.
             # raise ValueError("Payoff chart generation failed.") # Option to fail hard

        # Log warnings for non-critical failures
        if results["tax"] is None: logger.warning(f"[{endpoint_name}] Tax calculation failed or returned None.")
        if not results["greeks"]: logger.warning(f"[{endpoint_name}] Greeks calculation failed or returned empty list.")


    except ValueError as calc_val_err: # Catch specific value errors like metrics failing
         logger.error(f"[{endpoint_name}] Calculation prerequisite failed for {asset}: {calc_val_err}")
         raise HTTPException(status_code=500, detail=f"Calculation Error: {calc_val_err}")
    except Exception as e: # Catch errors from gather or other unexpected issues
        logger.error(f"[{endpoint_name}] Error during calculation phase for {asset}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected Server Error during calculation.")

    calc_duration = time.monotonic() - start_calc_time
    logger.info(f"[{endpoint_name}] Calculations finished in {calc_duration:.3f}s")

    # --- Step 4: Return Results ---
    success_status = results["chart"] is not None # Define success based on chart generation
    logger.info(f"[{endpoint_name}] Successfully processed payoff request for asset: {asset}. Status: {'Success' if success_status else 'Partial Success (Chart Failed)'}")
    return {
        "image_base64": results["chart"], # Will be None if failed
        "success": success_status,
        "metrics": results.get("metrics"), # Metrics should exist if we got here
        "charges": results.get("tax"),     # Might be None if tax failed
        "greeks": results.get("greeks", []) # Greeks list, empty if failed
    }




# --- LLM Stock Analysis Endpoint (FIXED 404/Data Handling) ---
@app.post("/get_stock_analysis", tags=["Analysis & Payoff"])
async def get_stock_analysis_endpoint(request: StockRequest):
    """
    Fetches stock data and news, generates an LLM analysis.C
    Handles missing data more gracefully and provides 404 only if essential data is unavailable.
    """
    asset = request.asset.strip().upper()
    if not asset: raise HTTPException(status_code=400, detail="Asset name required.")

    analysis_cache_key = f"analysis_{asset}"
    cached_analysis = analysis_cache.get(analysis_cache_key)
    if cached_analysis:
        logger.info(f"Cache hit analysis: {asset}")
        return {"analysis": cached_analysis}

    # Map asset to Yahoo symbol (handle common indices explicitly)
    if asset == "NIFTY": stock_symbol = "^NSEI"
    elif asset == "BANKNIFTY": stock_symbol = "^NSEBANK"
    elif asset == "FINNIFTY": stock_symbol = "NIFTY_FIN_SERVICE.NS" # Verify this ticker
    # Add other index mappings if needed
    else: stock_symbol = f"{asset}.NS" # Assume equity

    stock_data = None
    latest_news = []

    logger.info(f"Analysis request for {asset}, using symbol {stock_symbol}")

    try:
        # Fetch stock data FIRST - it's the most critical part
        logger.debug(f"Fetching stock data for {asset} ({stock_symbol})")
        stock_data = await fetch_stock_data_async(stock_symbol) # Use updated function

        # ***** CHANGE: Check if stock_data is None (critical failure) *****
        if stock_data is None:
            # fetch_stock_data_async now returns None only on critical failure (no price/invalid ticker)
            logger.error(f"Essential stock data (e.g., price) not found for {asset} ({stock_symbol}) after fetch attempt.")
            # This is the appropriate case for 404 Not Found.
            raise HTTPException(status_code=404, detail=f"Essential stock data not found for: {stock_symbol}. Symbol might be invalid or delisted.")
        else:
            logger.info(f"Successfully fetched stock data (may have missing fields) for {asset} ({stock_symbol}).")

        # Now fetch news concurrently (or sequentially if preferred)
        logger.debug(f"Fetching news for {asset}")
        latest_news = await fetch_latest_news_async(asset) # Use original asset name
        if not latest_news or "Error fetching news" in latest_news[0].get("headline",""):
            logger.warning(f"News fetch failed or returned error for {asset}. Proceeding without news.")
            # Use a placeholder if fetch failed
            latest_news = [{"headline": f"News data unavailable for {asset}.", "summary": "", "link": "#"}]
        else:
             logger.info(f"Successfully fetched news for {asset}.")

    except HTTPException as http_err:
         # Re-raise the specific 404 or 503 from stock data check
         raise http_err
    except Exception as e:
        # Catch other unexpected errors during data fetching orchestration
        logger.error(f"Unexpected error during concurrent data fetch for analysis ({asset}): {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Server error during data fetching for analysis.")

    # --- Generate Analysis using LLM ---
    # Proceed only if stock_data is valid (checked above)
    logger.debug(f"Building analysis prompt for {asset}...")
    prompt = build_stock_analysis_prompt(asset, stock_data, latest_news) # Pass asset name and symbol
    try:
        # Ensure Gemini API was configured
        if not GEMINI_API_KEY:
             raise RuntimeError("Gemini API Key not configured.")

        model = genai.GenerativeModel("gemini-1.5-flash-latest") # Or your preferred model
        logger.info(f"Generating Gemini analysis for {asset}...")
        start_llm = time.monotonic()
        # Use generate_content_async for non-blocking call
        response = await model.generate_content_async(prompt)
        llm_duration = time.monotonic() - start_llm
        logger.info(f"Gemini response received for {asset} in {llm_duration:.3f}s.")

        # Enhanced error checking for Gemini response
        analysis_text = None
        try:
            analysis_cache[analysis_cache_key] = analysis_text
            logger.info(f"Successfully generated and cached analysis for {asset}")
            return {"analysis": analysis_text}
        except (RuntimeError, ValueError) as gen_err:
            # This can happen if the response is blocked due to safety settings etc.
             logger.error(f"Gemini response error for {asset}: {resp_err}. Potentially blocked content.")
             logger.error(f"Gemini Prompt Feedback: {response.prompt_feedback}")
             raise ValueError(f"Analysis generation failed (response error/blocked). Feedback: {response.prompt_feedback}")

        if not analysis_text:
             logger.error(f"Gemini response missing text for {asset}. Parts: {response.parts}, Feedback: {response.prompt_feedback}")
             raise ValueError(f"Analysis generation failed (empty response). Feedback: {response.prompt_feedback}")

        analysis_cache[analysis_cache_key] = analysis_text # Cache successful analysis
        logger.info(f"Successfully generated and cached analysis for {asset}")
        return {"analysis": analysis_text}

    except (RuntimeError, ValueError) as gen_err: # Catch specific generation errors
        logger.error(f"Analysis generation error for {asset}: {gen_err}")
        raise HTTPException(status_code=503, detail=f"Analysis Generation Error: {gen_err}")
    except Exception as e:
        logger.error(f"Gemini API call or processing error for {asset}: {e}", exc_info=True)
        # Check for specific API errors if the library provides them
        # Example: if isinstance(e, google.api_core.exceptions.GoogleAPIError): ...
        raise HTTPException(status_code=503, detail=f"Analysis generation failed due to API or processing error: {type(e).__name__}")


# ===============================================================
# Main Execution Block (Keep as is)
# ===============================================================
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0") # Bind to 0.0.0.0 for external access (like Render)
    port = int(os.getenv("PORT", 8000)) # Render typically sets PORT env var
    reload = os.getenv("RELOAD", "false").lower() == "true"

    # Set log level based on environment?
    log_level = "debug" if reload else "info"

    logger.info(f"Starting Uvicorn server on http://{host}:{port} (Reload: {reload}, LogLevel: {log_level})")
    uvicorn.run(
        "app:app", # Point to the FastAPI app instance
        host=host,
        port=port,
        reload=reload, # Enable auto-reload for local development if needed
        log_level=log_level
        # Consider adding reload_dirs=["."] if reload=True and you have other modules
    )