
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
from contextlib import asynccontextmanager
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
# import plotly.graph_objects as go
# Removed pandas import if not used elsewhere
# import pandas as pd
# from scipy.stats import norm # Not currently used
import math
import mibian
# Opstrat not used for plotting now
# import opstrat
# *** Ensure Matplotlib Imports are Correct ***
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
# Set some optimized Matplotlib parameters globally (optional)
# plt.style.use('fast') # potentially faster style, less visually complex
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 0.6
plt.rcParams['agg.path.chunksize'] = 10000 # Process paths in chunks

# --- Caching ---
from cachetools import TTLCache

# --- AI/LLM ---
import google.generativeai as genai

# --- Database ---
# Assuming database.py provides initialize_database_pool and get_db_connection
try:
    from database import initialize_database_pool, get_db_connection
except ImportError:
    # Provide dummy functions if database.py is missing, allowing startup but failing DB ops
    print("WARNING: database.py not found or failed to import. Using dummy DB functions.")
    logger = logging.getLogger(__name__) # Need logger early
    logger.error("FATAL: database.py not found. Database operations will fail.")
    # Dummy DB pool init function
    def initialize_database_pool():
        logger.warning("Using dummy initialize_database_pool.")
        pass
    # Dummy DB connection context manager
    @asynccontextmanager # Corrected decorator for async context manager if database.py is async
    async def get_db_connection(): # Assuming async connection based on FastAPI context
        logger.error("Attempted to use dummy get_db_connection. DB operations will fail.")
        raise ConnectionError("Database module not found or connection failed.")
        yield None # Needs to yield something for 'with' statement structure

import mysql.connector # Keep for catching specific DB errors if needed


# ===============================================================
# Initial Setup (Keep as is)
# ===============================================================

# --- Append Project Root (If necessary) ---
sys.path.append(str(Path(__file__).resolve().parent))

# --- Load Environment Variables ---
load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================================================
# Configuration & Constants (Keep as is, but ensure GEMINI_API_KEY is secure)
# ===============================================================
# --- API Base ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# --- Caching ---
option_chain_cache = TTLCache(maxsize=50, ttl=3)
option_chain_lock = threading.Lock()
stock_data_cache = TTLCache(maxsize=100, ttl=600)
news_cache = TTLCache(maxsize=100, ttl=900)
analysis_cache = TTLCache(maxsize=50, ttl=1800)

# --- Background Update Thread Config ---
try:
    # *** SET DEFAULT INTERVAL TO 3 SECONDS ***
    LIVE_UPDATE_INTERVAL_SECONDS = int(os.getenv("LIVE_UPDATE_INTERVAL", 3))
    if LIVE_UPDATE_INTERVAL_SECONDS <= 0: raise ValueError()
except:
    LIVE_UPDATE_INTERVAL_SECONDS = 3 # Default to 3 seconds
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
PAYOFF_LOWER_BOUND_FACTOR = 0.50
PAYOFF_UPPER_BOUND_FACTOR = 1.50
PAYOFF_POINTS = 300
BREAKEVEN_CLUSTER_GAP_PCT = 0.005

# --- LLM Configuration ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # <-- LOAD FROM ENVIRONMENT
    # Fallback for testing if not in env (REMOVE THIS IN PRODUCTION)
    if not GEMINI_API_KEY:
         GEMINI_API_KEY = "AIzaSyDd_UVZ_1OeLahVrJ0A-hbazQcr1FOpgPE" # Your hardcoded key
         logger.warning("!!! Loaded Gemini API Key from hardcoded value - NOT FOR PRODUCTION !!!")

    if not GEMINI_API_KEY:
        raise ValueError("Gemini API Key is missing in environment variables and fallback.")
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured.")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}. Analysis endpoint will likely fail.")
    # Depending on criticality, you might want to raise SystemExit here
    # raise SystemExit(f"Gemini API Key configuration failed: {e}") from e


# ===============================================================
# Global State (Keep as is)
# ===============================================================
selected_asset: Optional[str] = None
strategy_positions: List[dict] = []
shutdown_event = threading.Event()
background_thread_instance: Optional[threading.Thread] = None
n: Optional[NSELive] = None
#PLOTLY_KALEIDO_AVAILABLE = False

# ===============================================================
# Helper Functions (Keep as is)
# ===============================================================
def _safe_get_float(data: Dict, key: str, default: Optional[float] = None) -> Optional[float]:
    # ... (no changes needed)
    value = data.get(key)
    if value is None: return default
    try: return float(value)
    except (TypeError, ValueError):
        if default is not None:
            logger.debug(f"Invalid float value '{value}' for key '{key}'. Using default {default}.")
        return default

def _safe_get_int(data: Dict, key: str, default: Optional[int] = None) -> Optional[int]:
    # ... (no changes needed)
    value = data.get(key)
    if value is None: return default
    try: return int(float(value)) # Allow float conversion first
    except (TypeError, ValueError):
        if default is not None:
            logger.debug(f"Invalid int value '{value}' for key '{key}'. Using default {default}.")
        return default

def get_cached_option(asset: str) -> Optional[dict]:
    # This function remains the same - it's used by API endpoints,
    # but WILL NOT be used by the background thread anymore.
    global n; now = time.time(); cache_key = f"option_chain_{asset}"
    with option_chain_lock: cached_data = option_chain_cache.get(cache_key)
    if cached_data: return cached_data
    logger.info(f"Cache miss (API). Fetching live option chain for: {asset}")
    try:
        if not n: raise RuntimeError("NSELive client not initialized.")
        asset_upper = asset.upper()
        option_data = n.index_option_chain(asset_upper) if asset_upper in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"] else n.equities_option_chain(asset_upper)
        if option_data:
            with option_chain_lock: option_chain_cache[cache_key] = option_data
            logger.info(f"Successfully fetched/cached option chain for API: {asset}")
            return option_data
        else: logger.warning(f"Received empty data from NSELive for {asset}"); return None
    except Exception as e: logger.error(f"Error fetching option chain from NSELive for {asset}: {e}", exc_info=False); return None

def fetch_and_update_single_asset_data(asset_name: str):
    """
    Fetches LIVE data directly from NSELive and updates the database for ONE asset.
    BYPASSES get_cached_option to ensure fresh fetch attempts.
    Called by the background thread every LIVE_UPDATE_INTERVAL_SECONDS.
    WARNING: Frequent calls can lead to rate limiting or performance issues.
    MUST REMAIN SYNCHRONOUS.
    """
    global n # Need access to the global NSELive client instance
    func_name = "fetch_and_update_single_asset_data"
    logger.info(f"[{func_name}] Starting live fetch & DB update for: {asset_name}")
    start_time = datetime.now()
    conn_obj = None
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

        if not option_source_data:
            logger.warning(f"[{func_name}] Received empty data directly from NSELive for {asset_name}.")
            return # Exit if no data received
        logger.info(f"[{func_name}] Successfully fetched live data for {asset_name} directly.")

    except RuntimeError as rte:
         logger.error(f"[{func_name}] Runtime Error during live fetch for {asset_name}: {rte}")
         return # Exit if client not ready
    except Exception as fetch_err:
        logger.error(f"[{func_name}] Error during direct NSELive fetch for {asset_name}: {fetch_err}", exc_info=False)
        return # Exit if fetch fails

    # --- Step 2: Process and Update Database (Same logic as before) ---
    try:
        # Check structure again just in case
        if not isinstance(option_source_data, dict) or "records" not in option_source_data or "data" not in option_source_data["records"]:
            logger.error(f"[{func_name}] Invalid live data structure for {asset_name} after fetch.")
            return
        option_data_list = option_source_data["records"]["data"]
        if not option_data_list:
            logger.warning(f"[{func_name}] No option data found in live source for {asset_name}.")
            return

        # Database interaction (remains synchronous)
        with get_db_connection() as conn:
            if conn is None: raise ConnectionError("Failed to get DB connection.")
            conn_obj = conn
            with conn.cursor(dictionary=True) as cursor:
                # Get asset_id (sync)
                cursor.execute("SELECT id FROM option_data.assets WHERE asset_name = %s", (asset_name,))
                result = cursor.fetchone()
                if not result: logger.error(f"[Updater] Asset '{asset_name}' not found in DB."); return
                asset_id = result["id"]

                # Process Expiries (sync)
                expiry_dates_formatted = set()
                for item in option_data_list:
                    try: expiry_dates_formatted.add(datetime.strptime(item.get("expiryDate"), "%d-%b-%Y").strftime("%Y-%m-%d"))
                    except Exception: pass

                # Delete Old Expiries (sync)
                today_str = date.today().strftime("%Y-%m-%d"); cursor.execute("DELETE FROM option_data.expiries WHERE asset_id = %s AND expiry_date < %s", (asset_id, today_str))

                # Upsert Current Expiries & Fetch IDs (sync)
                expiry_id_map = {}
                if expiry_dates_formatted:
                    ins_data=[(asset_id,e) for e in expiry_dates_formatted]; cursor.executemany("INSERT INTO option_data.expiries (asset_id, expiry_date) VALUES (%s, %s) ON DUPLICATE KEY UPDATE expiry_date = VALUES(expiry_date)", ins_data)
                    placeholders=', '.join(['%s']*len(expiry_dates_formatted)); cursor.execute(f"SELECT id, expiry_date FROM option_data.expiries WHERE asset_id = %s AND expiry_date IN ({placeholders})", (asset_id, *expiry_dates_formatted))
                    for row in cursor.fetchall(): expiry_id_map[row["expiry_date"].strftime("%Y-%m-%d")] = row["id"]

                # Prepare Option Chain Data (sync)
                option_chain_data_to_upsert = []; skipped = 0; processed = 0
                for item in option_data_list:
                    try: # Process row safely
                        strike=float(item['strikePrice']); raw_expiry=item['expiryDate']; expiry_date_str=datetime.strptime(raw_expiry,"%d-%b-%Y").strftime("%Y-%m-%d"); expiry_id=expiry_id_map.get(expiry_date_str)
                        if expiry_id is None: skipped+=1; continue
                        for opt_type in ["CE", "PE"]:
                            details = item.get(opt_type);
                            if isinstance(details, dict):
                                processed+=1; idf=details.get("identifier", f"{asset_name}_{expiry_date_str}_{strike}_{opt_type}")
                                row_data=(asset_id,expiry_id,strike,opt_type,idf,_safe_get_int(details,"openInterest",0),_safe_get_int(details,"changeinOpenInterest",0),_safe_get_int(details,"totalTradedVolume",0),_safe_get_float(details,"impliedVolatility",0.0),_safe_get_float(details,"lastPrice",0.0),_safe_get_int(details,"bidQty",0),_safe_get_float(details,"bidprice",0.0),_safe_get_int(details,"askQty",0),_safe_get_float(details,"askPrice",0.0),_safe_get_int(details,"totalBuyQuantity",0),_safe_get_int(details,"totalSellQuantity",0))
                                option_chain_data_to_upsert.append(row_data)
                    except Exception as row_err: skipped+=1; logger.debug(f"Error processing row: {row_err}"); continue
                logger.debug(f"Prepared {len(option_chain_data_to_upsert)} option rows for {asset_name} ({processed} options, {skipped} rows skipped).")

                # Upsert Option Chain Data (sync)
                if option_chain_data_to_upsert:
                    sql = "INSERT INTO option_data.option_chain (asset_id, expiry_id, strike_price, option_type, identifier, open_interest, change_in_oi, total_traded_volume, implied_volatility, last_price, bid_qty, bid_price, ask_qty, ask_price, total_buy_qty, total_sell_qty) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE identifier=VALUES(identifier), open_interest=VALUES(open_interest), change_in_oi=VALUES(change_in_oi), total_traded_volume=VALUES(total_traded_volume), implied_volatility=VALUES(implied_volatility), last_price=VALUES(last_price), bid_qty=VALUES(bid_qty), bid_price=VALUES(bid_price), ask_qty=VALUES(ask_qty), ask_price=VALUES(ask_price), total_buy_qty=VALUES(total_buy_qty), total_sell_qty=VALUES(total_sell_qty)"
                    cursor.executemany(sql, option_chain_data_to_upsert)
                    logger.debug(f"Upserted option chain for {asset_name} (Affected: {cursor.rowcount}).")

                # Commit (sync)
                conn.commit()
                logger.info(f"[{func_name}] Successfully committed DB update for {asset_name}.")

    # Error handling for DB part
    except (mysql.connector.Error, ConnectionError) as db_err:
        logger.error(f"[{func_name}] DB/Connection error during update for {asset_name}: {db_err}")
        try: conn_obj.rollback(); logger.info("Rollback attempted.")
        except Exception as rb_err: logger.error(f"Rollback failed: {rb_err}")
    except Exception as e:
        logger.error(f"[{func_name}] Unexpected error during DB update phase for {asset_name}: {e}", exc_info=True)
        try: conn_obj.rollback(); logger.info("Rollback attempted.")
        except Exception as rb_err: logger.error(f"Rollback failed: {rb_err}")
    finally:
        duration = datetime.now() - start_time
        logger.info(f"[{func_name}] Finished task for asset: {asset_name}. Duration: {duration}")



def live_update_runner():
    """ Background thread target function. Calls the DIRECT fetch/update function. """
    global selected_asset
    thread_name = threading.current_thread().name
    logger.info(f"Background update thread '{thread_name}' started. Interval: {LIVE_UPDATE_INTERVAL_SECONDS}s.")
    while not shutdown_event.is_set():
        asset_to_update = selected_asset
        if asset_to_update and isinstance(asset_to_update, str) and asset_to_update.strip():
            logger.info(f"[{thread_name}] Updating data for selected asset: {asset_to_update}")
            start_time = time.monotonic()
            try:
                # *** Calls the modified function that fetches live data directly ***
                fetch_and_update_single_asset_data(asset_to_update)
                duration = time.monotonic() - start_time
                logger.info(f"[{thread_name}] Finished update cycle for {asset_to_update}. Duration: {duration:.3f}s")
            except Exception as e:
                duration = time.monotonic() - start_time
                logger.error(f"[{thread_name}] Error in update cycle for {asset_to_update} after {duration:.3f}s: {e}", exc_info=True)
        else:
            #Optional: Add a small log if no asset is selected to show the thread is alive
            logger.debug(f"[{thread_name}] No asset selected. Waiting...")
            pass

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
    # Removed Plotly/Kaleido dependency check
    try: initialize_database_pool(); logger.info("Database pool initialized.")
    except Exception as db_err: logger.exception("CRITICAL: DB Pool Init Failed.")
    try: n = NSELive(); logger.info("NSELive client initialized.")
    except Exception as nse_err: logger.error(f"Failed to initialize NSELive client: {nse_err}."); n = None
    logger.warning("Starting flawed background update thread...")
    shutdown_event.clear()
    background_thread_instance = threading.Thread(target=live_update_runner, name="FlawedLiveUpdateThread", daemon=True)
    background_thread_instance.start()
    yield
    logger.info("Application shutting down...") # Shutdown sequence...
    shutdown_event.set()
    if background_thread_instance and background_thread_instance.is_alive():
        background_thread_instance.join(timeout=LIVE_UPDATE_INTERVAL_SECONDS + 1)
        if background_thread_instance.is_alive(): logger.warning("Background thread did not stop gracefully.")
    logger.info("Application shutdown complete.")


app = FastAPI(
    title="Option Strategy Analyzer API",
    description="API for fetching option data, calculating strategies, and performing analysis.",
    version="1.1.0", # Incremented version (Plotly change)
    lifespan=lifespan
)

# --- CORS Middleware ---
ALLOWED_ORIGINS = [
    "http://localhost", "http://localhost:3000", "http://127.0.0.1:8000", # Added 127.0.0.1
    "https://option-strategy-vaqz.onrender.com", "https://option-strategy.onrender.com"
]
logger.info(f"Configuring CORS for origins: {ALLOWED_ORIGINS}")
app.add_middleware( CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ===============================================================
# Pydantic Models (Keep as is)
# ===============================================================
class AssetUpdateRequest(BaseModel): asset: str
class SpotPriceResponse(BaseModel): spot_price: float
class StockRequest(BaseModel): asset: str
class PositionInput(BaseModel): symbol: str; strike: float; type: str = Field(pattern="^(CE|PE)$"); quantity: int; price: float
class StrategyLegInputPayoff(BaseModel): option_type: str = Field(pattern="^(CE|PE)$"); strike_price: Union[float, str]; tr_type: str = Field(pattern="^(b|s)$"); option_price: Union[float, str]; expiry_date: str; lots: Union[int, str]; lot_size: Optional[Union[int, str]] = None
class PayoffRequest(BaseModel): asset: str; strategy: List[StrategyLegInputPayoff]
class DebugAssetSelectRequest(BaseModel): asset: str


# ===============================================================
# Calculation Functions (Modify news fetch)
# ===============================================================
def get_lot_size(asset_name: str) -> int | None:
    # ... (no changes needed)
    logger.debug(f"Fetching lot size for asset: {asset_name}")
    sql = "SELECT lot_size FROM option_data.assets WHERE asset_name = %s"
    lot_size = None
    try:
        # Assuming sync connection for this part as well
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (asset_name,))
                result = cursor.fetchone()
                if result and result[0] is not None:
                    try: lot_size = int(result[0]); logger.debug(f"Found lot size for {asset_name}: {lot_size}")
                    except (ValueError, TypeError): logger.error(f"Invalid non-integer lot size '{result[0]}' in DB for {asset_name}.")
                elif not result: logger.warning(f"No asset found with name: {asset_name} in assets table.")
                else: logger.warning(f"Lot size is NULL in DB for {asset_name}.")
    except ConnectionError as e: logger.error(f"DB Connection Error fetching lot size for {asset_name}: {e}", exc_info=True)
    except mysql.connector.Error as e: logger.error(f"DB Query Error fetching lot size for {asset_name}: {e}", exc_info=True)
    except Exception as e: logger.error(f"Unexpected error fetching lot size for {asset_name}: {e}", exc_info=True)
    return lot_size


def extract_iv(asset_name: str, strike_price: float, expiry_date: str, option_type: str) -> float | None:
    # ... (no changes needed)
    logger.debug(f"Attempting to extract IV for {asset_name} {expiry_date} {strike_price} {option_type}")
    try: target_expiry = datetime.strptime(expiry_date, "%Y-%m-%d").strftime("%d-%b-%Y")
    except ValueError: logger.error(f"Invalid expiry date format: {expiry_date}"); return None
    try:
        option_data = get_cached_option(asset_name)
        if not isinstance(option_data, dict): logger.warning(f"Cached data not dict for {asset_name}"); return None
        records = option_data.get("records"); data_list = records.get("data") if isinstance(records, dict) else None
        if not isinstance(data_list, list): logger.warning(f"Records.data not list for {asset_name}"); return None
        option_key = option_type.upper()
        for item in data_list:
            if isinstance(item, dict):
                 item_strike = _safe_get_float(item, "strikePrice")
                 item_expiry = item.get("expiryDate")
                 # Check type and value equality carefully
                 if item_strike is not None and abs(item_strike - strike_price) < 0.01 and item_expiry == target_expiry: # Use tolerance for float comparison
                    option_details = item.get(option_key)
                    if isinstance(option_details, dict):
                        iv = option_details.get("impliedVolatility")
                        if iv is not None and isinstance(iv, (int, float)) and iv > 0:
                            logger.debug(f"Found IV {iv} for {asset_name} {target_expiry} {strike_price} {option_key}")
                            return float(iv)
                        else: logger.debug(f"IV missing or invalid<=0 ({iv}) for {asset_name} {strike_price} {option_key}")
        logger.warning(f"No matching contract/valid IV found for {asset_name} {strike_price}@{target_expiry} {option_key}")
        return None
    except Exception as e: logger.error(f"Error extracting IV for {asset_name}: {e}", exc_info=True); return None


# ===============================================================
# 1. Calculate Option Taxes (Keep as corrected previously)
# ===============================================================
def calculate_option_taxes(strategy_data: List[Dict[str, Any]], asset: str) -> Optional[Dict[str, Any]]:
    # ... (Keep the robust version from previous corrections) ...
    func_name = "calculate_option_taxes"
    logger.info(f"[{func_name}] Calculating for {len(strategy_data)} leg(s), asset: {asset}")
    # ... (Fetch prerequisites, Initialize totals, Process each leg with validation, Finalize) ...
    # ... (Return results dict or None) ...
    try:
        # --- Fetch Prerequisites ---
        cached_data = get_cached_option(asset)
        if not cached_data or "records" not in cached_data:
            raise ValueError("Missing or invalid market data from cache")
        spot_price = _safe_get_float(cached_data.get("records", {}), "underlyingValue")
        if spot_price is None or spot_price <= 0:
            raise ValueError(f"Spot price missing or invalid ({spot_price}) in cache")
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
            op_type = str(leg.get('op_type', '')).lower()
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
            elif premium is None or premium < 0: error_msg = f"invalid premium ({premium})"
            elif lots is None or lots <= 0: error_msg = f"invalid lots ({lots})"
            elif leg_lot_size <= 0: error_msg = f"invalid lot_size ({leg_lot_size})"

            if error_msg:
                 raise ValueError(f"Leg {i}: {error_msg}. Data: {leg}")

            # --- Calculate Charges for the Leg ---
            quantity = lots * leg_lot_size
            turnover = premium * quantity
            stt_val = 0.0
            stt_note = ""

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

                # STT applied only if ITM and intrinsic > 0
                if is_itm and intrinsic > 0:
                    stt_val = intrinsic * STT_EXERCISE_RATE
                    stt_note = f"{STT_EXERCISE_RATE*100:.4f}% STT (Exercise ITM)"
                else:
                    stt_note = "No STT (Buy OTM/ATM)"

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
                "option_type": op_type.upper(),
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
            # Log and re-raise to stop calculation for the whole strategy
            logger.error(f"[{func_name}] Validation Error processing leg {i} for {asset}: {val_err}")
            raise ValueError(f"Invalid data in tax leg {i+1}: {val_err}") from val_err
        except Exception as leg_err:
            # Log and re-raise unexpected errors
            logger.error(f"[{func_name}] Unexpected Error processing tax leg {i} for {asset}: {leg_err}", exc_info=True)
            raise ValueError(f"Unexpected error processing tax leg {i+1}") from leg_err

    # --- Finalize and Return ---
    final_charges_summary = {k: round(v, 2) for k, v in totals.items()}
    final_total_cost = round(sum(totals.values()), 2)

    logger.info(f"[{func_name}] Calculation complete for {asset}. Total estimated charges: {final_total_cost:.2f}")

    # Include rate info used in calculations
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
# 2. Generate Payoff Chart (using Matplotlib - Enhanced)
# ===============================================================
def generate_payoff_chart_matplotlib(strategy_data: List[Dict[str, Any]], asset: str) -> Optional[str]:
    # ... (Prerequisite fetching and Payoff Calculation logic remains the same) ...
    # ... (Ensure PAYOFF_POINTS is set, e.g., 350) ...
    func_name = "generate_payoff_chart_matplotlib"
    logger.info(f"[{func_name}] Generating chart for {len(strategy_data)} leg(s), asset: {asset}")
    start_time = time.monotonic()

    fig = None
    try:
        # 1. Fetch Prerequisites (Same as before)
        # ...
        cached_data = get_cached_option(asset); spot_price = _safe_get_float(cached_data.get("records", {}), "underlyingValue"); default_lot_size = get_lot_size(asset)
        if not spot_price or spot_price <= 0 or not default_lot_size or default_lot_size <=0: raise ValueError("Invalid spot/lot size")

        # 2. Calculate Payoff Data (Same as before)
        # ...
        lower_bound = max(spot_price*PAYOFF_LOWER_BOUND_FACTOR*0.9, 0.1); upper_bound = spot_price*PAYOFF_UPPER_BOUND_FACTOR*1.1
        price_range = np.linspace(lower_bound, upper_bound, PAYOFF_POINTS); total_payoff = np.zeros_like(price_range)
        unique_strikes = set()
        processed_legs_count = 0
        for i, leg in enumerate(strategy_data): # Loop through legs
            try: # Extract/validate leg
                tr_type=leg['tr_type'].lower(); op_type=leg['op_type'].lower(); strike=float(leg['strike']); premium=float(leg['op_pr']); lots=int(leg['lot']); leg_lot_size=int(leg.get('lot_size',default_lot_size))
                if not(tr_type in ('b','s') and op_type in ('c','p') and strike>0 and premium>=0 and lots>0 and leg_lot_size>0): raise ValueError("Invalid params")
                unique_strikes.add(strike)
            except Exception as e: raise ValueError(f"Leg {i+1}: {e}") from e
            quantity=lots*leg_lot_size; leg_prem_tot=premium*quantity # Calculate payoff
            intrinsic = np.maximum(price_range - strike, 0) if op_type == 'c' else np.maximum(strike - price_range, 0)
            leg_payoff = (intrinsic*quantity - leg_prem_tot) if tr_type == 'b' else (leg_prem_tot - intrinsic*quantity)
            total_payoff += leg_payoff; processed_legs_count += 1
        if processed_legs_count == 0: logger.warning(f"No valid legs for {asset}."); return None

        # 3. --- Create Matplotlib Figure ---
        plt.close('all')
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=95) # Slightly larger figure

        # 4. --- Plot and Style (with Larger Fonts) ---
        ax.plot(price_range, total_payoff, color='mediumblue', linewidth=2.0, label="Strategy Payoff", zorder=10) # Thicker line
        ax.axhline(0, color='black', linewidth=1.0, linestyle='-', alpha=0.9, zorder=1) # Slightly thicker zero line
        ax.axvline(spot_price, color='dimgrey', linestyle='--', linewidth=1.2, label=f'Spot {spot_price:.2f}', zorder=1) # Slightly thicker spot line

        strike_line_color = 'darkorange'; text_y_offset_factor = 0.09 # Adjust Y offset
        y_min_plot, y_max_plot = np.min(total_payoff), np.max(total_payoff); y_range_plot = y_max_plot - y_min_plot if y_max_plot > y_min_plot else 1.0
        text_y_pos = y_min_plot - y_range_plot * text_y_offset_factor

        for k in sorted(list(unique_strikes)):
             ax.axvline(k, color=strike_line_color, linestyle=':', linewidth=1.0, alpha=0.75, zorder=1) # Slightly thicker strike lines
             ax.text(k, text_y_pos, f'{k:g}', color=strike_line_color, ha='center', va='top', fontsize=9, alpha=0.95, weight='medium') # Larger strike text

        ax.fill_between(price_range, total_payoff, 0, where=total_payoff >= 0, facecolor='palegreen', alpha=0.5, interpolate=True, zorder=0) # Slightly more opaque fill
        ax.fill_between(price_range, total_payoff, 0, where=total_payoff < 0, facecolor='lightcoral', alpha=0.5, interpolate=True, zorder=0)

        # Increase font sizes
        ax.set_title(f"{asset} Strategy Payoff", fontsize=15, weight='bold') # Larger title
        ax.set_xlabel("Underlying Price at Expiry", fontsize=11) # Larger X label
        ax.set_ylabel("Profit / Loss", fontsize=11) # Larger Y label
        ax.tick_params(axis='both', which='major', labelsize=10) # Larger tick labels
        ax.legend(fontsize=10) # Larger legend
        ax.grid(True, which='major', linestyle=':', linewidth=0.6, alpha=0.7)

        # Adjust axis limits (same logic)
        x_padding = (upper_bound - lower_bound) * 0.02
        ax.set_xlim(lower_bound - x_padding, upper_bound + x_padding)
        y_padding = (y_max_plot - y_min_plot) * 0.05 if y_range_plot > 0 else abs(y_min_plot * 0.1) if y_min_plot != 0 else 10
        ax.set_ylim(y_min_plot - y_padding - abs(text_y_pos - y_min_plot), y_max_plot + y_padding)

        fig.tight_layout(pad=1.0)

        # 5. --- Save to Buffer (Same logic) ---
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=fig.dpi)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode("utf-8")

        duration = time.monotonic() - start_time
        logger.info(f"[{func_name}] Successfully generated Matplotlib chart for {asset} in {duration:.3f}s")
        return img_base64

    # ... (Error handling and finally block remain the same) ...
    except ValueError as val_err: logger.error(f"[{func_name}] Error: {val_err}"); return None
    except Exception as e: logger.error(f"[{func_name}] Unexpected error for {asset}: {e}", exc_info=True); return None
    finally:
        if fig is not None:
            try: plt.close(fig)
            except Exception: pass

# ===============================================================
# 3. Calculate Strategy Metrics (REVISED Breakeven Calculation)
# ===============================================================
def calculate_strategy_metrics(strategy_data: List[Dict[str, Any]], asset: str) -> Optional[Dict[str, Any]]:
    """
    Calculates Profit & Loss metrics for a multi-leg options strategy,
    using theoretical P/L limits and refined breakeven calculation.
    """
    func_name = "calculate_strategy_metrics"
    logger.info(f"[{func_name}] Calculating for {len(strategy_data)} leg(s), asset: {asset}")

    # --- 1. Fetch Essential Data ---
    try:
        cached_data = get_cached_option(asset)
        if not isinstance(cached_data, dict) or "records" not in cached_data: raise ValueError("Invalid cache structure")
        records = cached_data.get("records", {})
        spot_price = records.get("underlyingValue")
        if spot_price is None or not isinstance(spot_price, (int, float)) or spot_price <= 0: raise ValueError("Spot price missing/invalid")
        spot_price = float(spot_price)
        default_lot_size = get_lot_size(asset)
        if default_lot_size is None or default_lot_size <= 0: raise ValueError("Default lot size missing/invalid")
    except Exception as e:
        logger.error(f"[{func_name}] Failed prerequisite data fetch for {asset}: {e}")
        return None

    # --- 2. Process Strategy Legs & Store Details ---
    # (This part remains the same as the previous version - validating legs and storing details)
    total_net_premium = 0.0; cost_breakdown = []; processed_legs = 0
    net_long_call_qty = 0; net_short_call_qty = 0; net_long_put_qty = 0; net_short_put_qty = 0
    payoff_at_S_equals_0 = 0.0; legs_for_payoff_calc = []
    all_strikes_list = [] # Collect all strikes

    for i, leg in enumerate(strategy_data):
        try:
            tr_type = str(leg.get('tr_type', '')).lower(); option_type = str(leg.get('op_type', '')).lower()
            strike = _safe_get_float(leg, 'strike'); premium = _safe_get_float(leg, 'op_pr')
            lots = _safe_get_int(leg, 'lot'); leg_lot_size = _safe_get_int(leg, 'lot_size', default_lot_size)
            if tr_type not in ('b','s') or option_type not in ('c','p') or strike is None or strike <= 0 or premium is None or premium < 0 or lots is None or lots <= 0 or leg_lot_size <= 0:
                raise ValueError("Invalid leg parameters")
            quantity = lots * leg_lot_size; leg_premium_total = premium * quantity; action_verb = ""
            all_strikes_list.append(strike) # Add strike to list

            if tr_type == 'b': # Buy
                total_net_premium -= leg_premium_total; action_verb = "Paid"
                if option_type == 'c': net_long_call_qty += quantity
                else: net_long_put_qty += quantity
            else: # Sell
                total_net_premium += leg_premium_total; action_verb = "Received"
                if option_type == 'c': net_short_call_qty += quantity
                else: net_short_put_qty += quantity

            if option_type == 'p': # Payoff at S=0
                intrinsic_at_zero = strike
                payoff_at_S_equals_0 += (intrinsic_at_zero * quantity - leg_premium_total) if tr_type == 'b' else (leg_premium_total - intrinsic_at_zero * quantity)
            else: payoff_at_S_equals_0 += -leg_premium_total if tr_type == 'b' else leg_premium_total

            cost_breakdown.append({ "leg_index": i, "action": tr_type.upper(), "type": option_type.upper(), "strike": strike, "premium_per_share": premium, "lots": lots, "lot_size": leg_lot_size, "quantity": quantity, "total_premium": round(leg_premium_total if tr_type=='s' else -leg_premium_total, 2), "effect": action_verb })
            legs_for_payoff_calc.append({'tr_type': tr_type, 'op_type': option_type, 'strike': strike, 'premium': premium, 'quantity': quantity})
            processed_legs += 1
        except (ValueError, KeyError, TypeError) as leg_err: logger.error(f"[{func_name}] Error processing leg {i}: {leg_err}. Data: {leg}"); return None
    if processed_legs == 0: logger.error(f"[{func_name}] No valid legs processed."); return None

    # --- 3. Define Payoff Function ---
    def _calculate_payoff_at_price(S: float, legs: List[Dict]) -> float:
        # (Same payoff calculation helper as before)
        total_pnl = 0.0
        for leg in legs:
            intrinsic = 0.0; premium = leg['premium']; strike = leg['strike']
            quantity = leg['quantity']; op_type = leg['op_type']; tr_type = leg['tr_type']
            leg_prem_tot = premium * quantity
            if op_type == 'c': intrinsic = max(S - strike, 0)
            else: intrinsic = max(strike - S, 0)
            pnl = (intrinsic * quantity - leg_prem_tot) if tr_type == 'b' else (leg_prem_tot - intrinsic * quantity)
            total_pnl += pnl
        return total_pnl

    # --- 4. Determine Theoretical Max Profit / Loss (Same as before) ---
    net_calls = net_long_call_qty - net_short_call_qty
    all_strikes_unique_sorted = sorted(list(set(all_strikes_list))) # Unique sorted strikes
    max_profit_val = np.inf if net_calls > 0 else payoff_at_S_equals_0
    max_loss_val = -np.inf if net_calls < 0 else payoff_at_S_equals_0
    if net_calls <= 0: # Bounded Profit: Check payoff at strikes
        for k in all_strikes_unique_sorted: max_profit_val = max(max_profit_val, _calculate_payoff_at_price(k, legs_for_payoff_calc))
    if net_calls >= 0: # Bounded Loss: Check payoff at strikes
        for k in all_strikes_unique_sorted: max_loss_val = min(max_loss_val, _calculate_payoff_at_price(k, legs_for_payoff_calc))


    # --- 5. Breakeven Points (REVISED SEARCH LOGIC) ---
    logger.debug(f"[{func_name}] Starting breakeven search...")
    breakeven_points_found = []
    unique_strikes_and_zero = sorted(list(set(all_strikes_list + [0.0]))) # Include 0 and unique strikes

    # Define search intervals based on strikes and bounds
    search_intervals = []
    # Interval from 0 up to the first strike (or slightly beyond if only one strike)
    first_strike = unique_strikes_and_zero[1] if len(unique_strikes_and_zero) > 1 else spot_price
    search_intervals.append((max(0, first_strike * 0.5), first_strike * 1.1)) # Search below and slightly above first strike

    # Intervals between strikes
    for i in range(len(all_strikes_unique_sorted) - 1):
        # Add interval slightly padded around the strikes
        search_intervals.append((all_strikes_unique_sorted[i] * 0.99, all_strikes_unique_sorted[i+1] * 1.01))

    # Interval beyond the last strike
    last_strike = all_strikes_unique_sorted[-1] if all_strikes_unique_sorted else spot_price
    search_intervals.append((last_strike * 0.99, max(last_strike * 1.5, spot_price * PAYOFF_UPPER_BOUND_FACTOR * 1.1)))

    # Define the target function for the root finder (payoff should be zero)
    payoff_func = lambda s: _calculate_payoff_at_price(s, legs_for_payoff_calc)

    processed_intervals = set() # Avoid redundant searches

    for p1_raw, p2_raw in search_intervals:
        p1 = max(0.0, p1_raw) # Ensure lower bound is not negative
        p2 = p2_raw
        interval_key = (round(p1, 4), round(p2, 4)) # Key for processed set
        if p1 >= p2 or interval_key in processed_intervals: continue # Skip invalid or already processed intervals
        processed_intervals.add(interval_key)

        try:
            y1 = payoff_func(p1)
            y2 = payoff_func(p2)

            # Check if the signs are different (indicates a crossing)
            if np.sign(y1) != np.sign(y2):
                try:
                    from scipy.optimize import brentq
                    # Attempt to find the root using Brent's method
                    be = brentq(payoff_func, p1, p2, xtol=1e-6, rtol=1e-6)
                    # Check if BE is valid (positive price)
                    if be > 1e-6: # Use small tolerance above 0
                        breakeven_points_found.append(be)
                        logger.debug(f"[{func_name}] Found BE (brentq) in [{p1:.2f}, {p2:.2f}]: {be:.4f}")
                except ImportError:
                    # Fallback: Linear Interpolation
                    if abs(y2 - y1) > 1e-9:
                        be = p1 - y1 * (p2 - p1) / (y2 - y1)
                        if ((p1 <= be <= p2) or (p2 <= be <= p1)) and be > 1e-6:
                            breakeven_points_found.append(be)
                            logger.debug(f"[{func_name}] Found BE (interp) in [{p1:.2f}, {p2:.2f}]: {be:.4f}")
                except ValueError as brentq_err:
                    # Brentq raises ValueError if signs are the same or other issues
                    logger.debug(f"[{func_name}] Brentq failed for interval [{p1:.2f}, {p2:.2f}]: {brentq_err}")
                    # Could try simpler bisection here if needed as another fallback
        except Exception as search_err:
            logger.error(f"[{func_name}] Error during BE search in interval [{p1:.2f}, {p2:.2f}]: {search_err}")


    # Check payoff AT strikes for exact zero touches (common in straddles/strangles)
    zero_tolerance = 1e-4 # How close to zero counts
    for k in all_strikes_unique_sorted:
        payoff_at_k = payoff_func(k)
        if abs(payoff_at_k) < zero_tolerance:
             # Avoid adding duplicates extremely close to already found points
             is_close_to_existing = any(abs(k - be) < 0.01 for be in breakeven_points_found)
             if not is_close_to_existing:
                  breakeven_points_found.append(k)
                  logger.debug(f"[{func_name}] Found BE (strike touch): {k:.4f}")

    # Remove potential duplicates resulting from different methods finding near-identical points
    unique_be_points = sorted(list(set(round(p, 4) for p in breakeven_points_found if p >= 0))) # Ensure non-negative

    # --- Cluster Breakeven Points ---
    def cluster_points(points: List[float], tolerance_ratio: float, reference_price: float) -> List[float]:
        # (Same clustering function as before)
        if not points: return []
        points.sort(); absolute_tolerance = max(reference_price * tolerance_ratio, 0.01)
        if not points: return []
        clustered_groups = [[points[0]]]
        for p in points[1:]:
            if abs(p - clustered_groups[-1][-1]) <= absolute_tolerance: clustered_groups[-1].append(p)
            else: clustered_groups.append([p])
        return [round(np.mean(group), 2) for group in clustered_groups]

    breakeven_points_clustered = cluster_points(unique_be_points, BREAKEVEN_CLUSTER_GAP_PCT, spot_price)

    # --- 6. Reward to Risk Ratio (Keep logic from previous fix) ---
    # ... (R:R calculation logic based on theoretical max_profit_val, max_loss_val) ...
    reward_to_risk_ratio = "N/A"; zero_threshold = 1e-9
    max_p_num = max_profit_val if isinstance(max_profit_val, (int, float)) else None
    max_l_num = max_loss_val if isinstance(max_loss_val, (int, float)) else None
    if max_l_num is not None:
        if abs(max_l_num) < zero_threshold: reward_to_risk_ratio = "∞" if max_p_num is not None and max_p_num > zero_threshold else "0.00"
        elif max_p_num is not None: reward_to_risk_ratio = round(max_p_num / abs(max_l_num), 2) if max_p_num >= 0 else "Loss"
        elif max_profit_val == np.inf: reward_to_risk_ratio = "∞"
    elif max_loss_val == -np.inf: reward_to_risk_ratio = "0.00" if max_profit_val != np.inf else "Undefined"
    elif max_profit_val == np.inf: reward_to_risk_ratio = "∞"

    # --- 7. Format Final Output ---
    max_profit_str = "∞" if max_profit_val == np.inf else format(max_profit_val, '.2f')
    max_loss_str = "-∞" if max_loss_val == -np.inf else format(max_loss_val, '.2f')
    reward_to_risk_str = str(reward_to_risk_ratio) if not isinstance(reward_to_risk_ratio, (int, float)) else format(reward_to_risk_ratio, '.2f')

    logger.info(f"[{func_name}] Metrics calculated. MaxP: {max_profit_str}, MaxL: {max_loss_str}, BE: {breakeven_points_clustered}, R:R: {reward_to_risk_str}")

    return {
        "calculation_inputs": { "asset": asset, "spot_price_used": round(spot_price, 2), "default_lot_size": default_lot_size, "num_legs_processed": processed_legs },
        "metrics": { "max_profit": max_profit_str, "max_loss": max_loss_str, "breakeven_points": breakeven_points_clustered, "reward_to_risk_ratio": reward_to_risk_str, "net_premium": round(total_net_premium, 2) },
        "cost_breakdown_per_leg": cost_breakdown
    }


# ===============================================================
# 4. Calculate Option Greeks (Integrate User Provided Logic - Scaled Per Share)
# ===============================================================
def calculate_option_greeks(
    strategy_data: List[Dict[str, Any]],
    asset: str,
    interest_rate_pct: float = DEFAULT_INTEREST_RATE_PCT
) -> List[Dict[str, Any]]:
    """
    Calculates per-share option Greeks (scaled by 100) for each leg
    of a strategy using the mibian Black-Scholes model.
    (Based on user-provided logic)

    Args:
        strategy_data: A list of strategy leg dictionaries. Expected keys per leg:
                       'strike' (float/str), 'days_to_expiry' (int/str),
                       'iv' (float/str - *percentage*, e.g., 25.5 for 25.5%),
                       'op_type' (c/p), 'tr_type' (b/s).
                       'lot' and 'lot_size' keys are ignored for calculation based on intent.
        asset: The underlying asset name (e.g., "NIFTY").
        interest_rate_pct: Risk-free interest rate in percentage (e.g., 6.5 for 6.5%).

    Returns:
        A list of dictionaries, each containing the calculated scaled, per-share
        Greeks for a valid leg, or an empty list if fetching spot price fails or
        no legs are valid.
    """
    func_name = "calculate_option_greeks" # For logging
    logger.info(f"Calculating SCALED PER-SHARE Greeks for {len(strategy_data)} legs, asset: {asset}, rate: {interest_rate_pct}%")

    greeks_result_list: List[Dict[str, Any]] = []

    # --- 1. Fetch Spot Price ---
    try:
        cached_data = get_cached_option(asset)
        if not isinstance(cached_data, dict) or "records" not in cached_data:
            logger.error(f"[{func_name}] Invalid cache structure for asset {asset}: 'records' key missing.")
            return [] # Cannot proceed
        records = cached_data.get("records", {})
        spot_price = records.get("underlyingValue")
        if spot_price is None or not isinstance(spot_price, (int, float)):
            logger.error(f"[{func_name}] Spot price ('underlyingValue') missing or invalid in cache for {asset}")
            return [] # Cannot proceed
        spot_price = float(spot_price)
        logger.debug(f"[{func_name}] Using spot price {spot_price} for asset {asset}")
    except Exception as cache_err:
        logger.error(f"[{func_name}] Error fetching spot price from cache for {asset}: {cache_err}", exc_info=True)
        return [] # Return empty list on failure

    # --- 2. Process Each Leg ---
    for i, leg_data in enumerate(strategy_data):
        try:
            # --- Validate and Extract Leg Data Safely ---
            if not isinstance(leg_data, dict):
                raise ValueError(f"Leg {i} data is not a dictionary.")

            # Use safe getters
            strike_price = _safe_get_float(leg_data, 'strike', default=None)
            days_to_expiry = _safe_get_int(leg_data, 'days_to_expiry', default=None)
            implied_vol_pct = _safe_get_float(leg_data, 'iv', default=None) # Input IV is percentage
            option_type_flag = str(leg_data.get('op_type', '')).lower()
            transaction_type = str(leg_data.get('tr_type', '')).lower()
            # Lots/LotSize are present in leg_data but ignored for per-share greek calc

            # --- Input Validation ---
            if strike_price is None or strike_price <= 0: raise ValueError("Missing or invalid 'strike' (must be > 0)")
            if days_to_expiry is None or days_to_expiry < 0: raise ValueError("Missing or invalid 'days_to_expiry' (must be >= 0)")
            if implied_vol_pct is None or implied_vol_pct <= 0: raise ValueError("Missing or invalid 'iv' (must be > 0)")
            if option_type_flag not in ['c', 'p']: raise ValueError("Invalid 'op_type' (must be 'c' or 'p')")
            if transaction_type not in ['b', 's']: raise ValueError("Invalid 'tr_type' (must be 'b' or 's')")

            # Prepare inputs for mibian ([spot, strike, interest_rate_%, days_to_expiry])
            mibian_inputs = [spot_price, strike_price, interest_rate_pct, days_to_expiry]
            volatility_input = implied_vol_pct # Mibian expects IV as percentage points

            # --- Calculate PER-SHARE Greeks using mibian ---
            try:
                bs_model = mibian.BS(mibian_inputs, volatility=volatility_input)
            except OverflowError as math_err:
                 raise ValueError(f"Mibian calculation failed due to math error: {math_err}") from math_err
            except Exception as mibian_err:
                 raise ValueError(f"Mibian calculation error: {mibian_err}") from mibian_err

            # Extract per-share Greeks
            delta = bs_model.callDelta if option_type_flag == 'c' else bs_model.putDelta
            theta = bs_model.callTheta if option_type_flag == 'c' else bs_model.putTheta # Mibian theta is per day
            rho = bs_model.callRho if option_type_flag == 'c' else bs_model.putRho
            gamma = bs_model.gamma
            vega = bs_model.vega # Per 1% change in IV

            # --- Adjust Sign for Short Positions ---
            if transaction_type == 's':
                delta *= -1; gamma *= -1; theta *= -1; vega *= -1; rho *= -1

            # --- Apply Scaling (Per-share Greek * 100) ---
            scaling_factor = 100.0
            calculated_greeks = {
                'delta': round(delta * scaling_factor, 2),
                'gamma': round(gamma * scaling_factor, 4), # Gamma often needs more precision
                'theta': round(theta * scaling_factor, 2), # Scale Daily Theta
                'vega': round(vega * scaling_factor, 2),
                'rho': round(rho * scaling_factor, 2)
            }

            # Check for non-finite values AFTER scaling/rounding
            if any(not np.isfinite(v) for v in calculated_greeks.values()):
                logger.warning(f"[{func_name}] Skipping leg {i} for {asset} due to non-finite Greek result after scaling.")
                continue # Skip this leg

            # Append results for this leg
            greeks_result_list.append({
                'leg_index': i,
                'input_data': leg_data, # Include original input for context
                'calculated_greeks': calculated_greeks # Store the SCALED PER-SHARE greeks
            })

        except (ValueError, KeyError, TypeError) as validation_err:
            logger.warning(f"[{func_name}] Skipping Greek calculation for leg {i} due to invalid input: {validation_err}. Leg data: {leg_data}")
            continue # Skip to the next leg
        except Exception as e:
            logger.error(f"[{func_name}] Unexpected error calculating Greeks for leg {i}: {e}. Leg data: {leg_data}", exc_info=True)
            continue # Skip to the next leg

    logger.info(f"[{func_name}] Finished calculating SCALED PER-SHARE Greeks for {len(greeks_result_list)} valid legs.")
    return greeks_result_list


async def fetch_stock_data_async(stock_symbol: str) -> Optional[Dict[str, Any]]:
    # ... (no changes needed)
    cache_key = f"stock_{stock_symbol}"
    cached = stock_data_cache.get(cache_key)
    if cached: return cached
    logger.info(f"Fetching stock data for: {stock_symbol}")
    try:
        loop = asyncio.get_running_loop(); stock = await loop.run_in_executor(None, yf.Ticker, stock_symbol)
        # Fetch history in parallel
        t1=loop.run_in_executor(None, stock.history, "1d");
        t2=loop.run_in_executor(None, stock.history, "50d")
        t3=loop.run_in_executor(None, stock.history, "200d");
        t4=loop.run_in_executor(None, getattr, stock, 'info') # Fetch info concurrently
        h1d,h50d,h200d,info = await asyncio.gather(t1,t2,t3,t4)

        if h1d.empty:
             logger.warning(f"No 1-day history found for {stock_symbol}"); return None
        cp = h1d["Close"].iloc[-1] if not h1d["Close"].empty else info.get("currentPrice") # Fallback to info
        if cp is None: cp = info.get("previousClose") # Further fallback
        vol=h1d["Volume"].iloc[-1] if not h1d["Volume"].empty else info.get("volume", 0)

        ma50=h50d["Close"].mean() if not h50d.empty else None;
        ma200=h200d["Close"].mean() if not h200d.empty else None
        mc=info.get("marketCap"); pe=info.get("trailingPE"); eps=info.get("trailingEps"); sec=info.get("sector","N/A"); ind=info.get("industry","N/A")

        if cp is None: logger.warning(f"Could not determine current price for {stock_symbol}"); return None

        data = {"current_price":cp,"volume":vol,"moving_avg_50":ma50,"moving_avg_200":ma200,"market_cap":mc,"pe_ratio":pe,"eps":eps,"sector":sec,"industry":ind}
        stock_data_cache[cache_key] = data;
        return data
    except Exception as e: logger.error(f"Error fetching stock data for {stock_symbol}: {e}", exc_info=True); return None


async def fetch_latest_news_async(asset: str) -> List[Dict[str, str]]:
    """Fetches latest news headlines and summaries from Yahoo Finance."""
    cache_key = f"news_{asset.upper()}"
    cached = news_cache.get(cache_key)
    if cached: return cached

    logger.info(f"Fetching news for: {asset}")
    # Use ^NSEI for NIFTY index, otherwise append .NS for NSE stocks
    symbol = "^NSEI" if asset.upper() == "NIFTY" else f"{asset.upper()}.NS"
    url = f"https://finance.yahoo.com/quote/{symbol}/news"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    news_list = []
    try:
        loop = asyncio.get_running_loop()
        # Correctly run requests.get with timeout in executor
        response = await loop.run_in_executor(
            None,
            functools.partial(requests.get, url, headers=headers, timeout=10) # Use functools.partial
        )
        response.raise_for_status() # Check for HTTP errors

        soup = BeautifulSoup(response.text, "html.parser")
        # Target the list items containing news links (selector might need adjustment if Yahoo changes layout)
        news_items = soup.select('li.js-stream-content div') # Adjusted selector based on potential structure

        count = 0
        processed_links = set() # Avoid duplicate links if structure nests oddly

        for item in news_items:
            if count >= 3: break # Limit to 3 news items

            link_tag = item.find('a', href=True)
            title_tag = item.find('h3')
            summary_tag = item.find('p')

            if link_tag and title_tag:
                link = link_tag['href']
                # Ensure link is absolute
                if link.startswith('/'):
                    link = f"https://finance.yahoo.com{link}"

                # Check if we've already processed this link
                if link in processed_links:
                    continue

                headline = title_tag.get_text(strip=True)
                summary = summary_tag.get_text(strip=True) if summary_tag else "No summary available."

                if headline: # Ensure headline is not empty
                    news_list.append({"headline": headline, "summary": summary, "link": link})
                    processed_links.add(link)
                    count += 1

        if not news_list:
            logger.warning(f"No news items found for {asset} using selector 'li.js-stream-content div'.")
            news_list.append({"headline": "No recent news found.", "summary": "", "link": "#"})

        news_cache[cache_key] = news_list
        return news_list

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Error fetching news URL for {asset}: {req_err}")
        return [{"headline": "Error fetching news (Network).", "summary": "", "link": "#"}]
    except Exception as e:
        logger.error(f"Error fetching/parsing news for {asset}: {e}", exc_info=True)
        return [{"headline": "Error fetching news (Parsing).", "summary": "", "link": "#"}]


def build_stock_analysis_prompt(stock_symbol: str, stock_data: dict, latest_news: list) -> str:
    """
    Generates a more structured and detailed prompt for LLM stock analysis,
    focusing on available data and contextual comparison needs.
    """
    func_name = "build_stock_analysis_prompt"
    logger.debug(f"[{func_name}] Building structured prompt for {stock_symbol}")

    # --- Formatting Helper ---
    def fmt(v, p="₹", s="", pr=2, na="N/A"):
        # (Formatting function remains the same)
        if v is None: return na
        if isinstance(v,(int,float)):
            try:
                if abs(v) >= 1e7: return f"{p}{v/1e7:.{pr}f} Cr{s}"
                if abs(v) >= 1e5: return f"{p}{v/1e5:.{pr}f} L{s}"
                return f"{p}{v:,.{pr}f}{s}"
            except: return str(v)
        return str(v)

    # --- Prepare Data Sections for Prompt ---
    price = stock_data.get('current_price'); ma50 = stock_data.get('moving_avg_50'); ma200 = stock_data.get('moving_avg_200')
    volume = stock_data.get('volume'); market_cap = stock_data.get('market_cap'); pe_ratio = stock_data.get('pe_ratio')
    eps = stock_data.get('eps'); sector = stock_data.get('sector', 'N/A'); industry = stock_data.get('industry', 'N/A')

    # Technical Context String (same as before)
    # ... (Trend, Support, Resistance calculation) ...
    trend = "N/A"; support = "N/A"; resistance = "N/A"
    if price and ma50 and ma200:
        support_levels = sorted([lvl for lvl in [ma50, ma200] if lvl is not None and lvl < price], reverse=True)
        resistance_levels = sorted([lvl for lvl in [ma50, ma200] if lvl is not None and lvl >= price])
        support = " / ".join([fmt(lvl) for lvl in support_levels]) if support_levels else "Below Key MAs"
        resistance = " / ".join([fmt(lvl) for lvl in resistance_levels]) if resistance_levels else "Above Key MAs"
        if price > ma50 > ma200: trend = "Strong Uptrend (Price > 50MA > 200MA)"
        elif price > ma50 and price > ma200: trend = "Uptrend (Price > MAs)"
        elif price < ma50 < ma200: trend = "Strong Downtrend (Price < 50MA < 200MA)"
        elif price < ma50 and price < ma200: trend = "Downtrend (Price < MAs)"
        elif ma50 > ma200: trend = "Sideways/Uptrend Context (Price vs 50MA: %s)" % ('Above' if price > ma50 else 'Below')
        else: trend = "Sideways/Downtrend Context (Price vs 50MA: %s)" % ('Above' if price > ma50 else 'Below')
    elif price and ma50:
        support = fmt(ma50) if price > ma50 else "N/A (Below 50MA)"
        resistance = fmt(ma50) if price <= ma50 else "N/A (Above 50MA)"
        trend = "Above 50MA" if price > ma50 else "Below 50MA"
    tech_context = f"Price: {fmt(price)}, 50D MA: {fmt(ma50)}, 200D MA: {fmt(ma200)}, Trend Context: {trend}, Key Levels (from MAs): Support near {support}, Resistance near {resistance}. Volume (1d): {fmt(volume, p='', pr=0)}"


    # Fundamental Context String (same as before)
    fund_context = f"Market Cap: {fmt(market_cap, p='')}, P/E Ratio: {fmt(pe_ratio, p='', s='x')}, EPS: {fmt(eps)}, Sector: {sector}, Industry: {industry}"
    pe_comparison_note = f"Note: P/E ({fmt(pe_ratio, p='', s='x')}) should be compared to '{industry}' industry and historical averages for proper valuation context (peer data not provided)." if pe_ratio else "Note: P/E data unavailable for comparison."

    # News Context String - *** Corrected Quote Handling ***
    news_formatted = []
    for n in latest_news[:3]:
        headline = n.get('headline','N/A')
        # Use triple quotes for summary or replace both quote types if needed
        summary = n.get('summary','N/A').replace('"', "'").replace("'", "`") # Replace " with ' and ' with `
        link = n.get('link', '#')
        # Use single quotes for the outer f-string
        news_formatted.append(f'- [{headline}]({link}): {summary}')
    news_context = "\n".join(news_formatted) if news_formatted else "- No recent news summaries found."


    # --- Construct the Structured Prompt (same as before) ---
    prompt = f"""
Analyze the stock **{stock_symbol}** based *only* on the provided data snapshots. Use clear headings and bullet points.

**Analysis Request:**

1.  **Executive Summary:**
    *   Provide a brief (2-3 sentence) overall takeaway combining technical posture, basic fundamental indicators, and recent news sentiment. State the implied short-term bias (e.g., Bullish, Bearish, Neutral, Cautious).

2.  **Technical Analysis:**
    *   **Trend & Momentum:** Describe the current trend based on the price vs. 50D and 200D Moving Averages. Is the trend established or potentially changing? Comment on the provided volume figure (is it high/low relative to what might be typical? Acknowledge if comparison data is missing).
        *   *Data:* {tech_context}
    *   **Support & Resistance:** Identify immediate support and resistance levels based *only* on the 50D and 200D MAs provided.
    *   **Key Technical Observations:** Note any significant technical patterns implied by the data (e.g., price significantly extended from MAs, price consolidating near MAs).

3.  **Fundamental Snapshot:**
    *   **Company Size & Profitability:** Classify the company based on its Market Cap. Comment on the provided EPS figure as an indicator of recent profitability.
    *   **Valuation:** Discuss the P/E Ratio. {pe_comparison_note}
        *   *Data:* {fund_context}
    *   **Sector/Industry Context:** Briefly state the sector and industry. Mention any generally known characteristics or recent performance trends *if widely known* (but prioritize provided data).

4.  **Recent News Sentiment:**
    *   **Sentiment Assessment:** Summarize the general sentiment (Positive, Negative, Neutral, Mixed) conveyed by the recent news headlines/summaries provided.
    *   **Potential News Impact:** Briefly state how this news *might* influence near-term price action.
        *   *News Data:*
{news_context}

5.  **Outlook & Considerations:**
    *   **Consolidated Outlook:** Reiterate the short-term bias based on the synthesis of the above points.
    *   **Key Factors to Monitor:** What are the most important technical levels or potential catalysts (from news/fundamentals provided) to watch?
    *   **Identified Risks (from Data):** What potential risks are directly suggested by the provided data (e.g., price below key MA, high P/E without context, negative news)?
    *   **General Option Strategy Angle:** Based ONLY on the derived bias (Bullish/Bearish/Neutral) and acknowledging IV is unknown, suggest *general types* of option strategies that align (e.g., Bullish -> Long Calls/Spreads; Bearish -> Long Puts/Spreads; Neutral -> Credit Spreads/Iron Condors). **Do NOT suggest specific strikes or expiries.**

**Disclaimer:** This analysis is based on limited data points. It is not financial advice. Verify information and conduct thorough research before making any trading decisions.
"""
    return prompt



# ===============================================================
# API Endpoints (Modify Analysis Endpoint)
# ===============================================================

# ===============================================================
# API Endpoints (Fix async DB access)
# ===============================================================

@app.get("/health", tags=["Status"])
async def health_check(): # Already async
     db_status = "unknown"
     try:
          # *** USE async with and await ***
          async with get_db_connection() as conn:
               # Assuming the cursor from an async connection is also async
               async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    db_status = "connected" if await cursor.fetchone() else "query_failed"
     except Exception as e:
         logger.error(f"Health check DB error: {e}")
         db_status = f"error: {type(e).__name__}"
     return {"status": "ok", "database": db_status}

@app.get("/get_assets", tags=["Data"])
async def get_assets(): # Already async
    logger.info("Request received for asset list.")
    asset_names = []
    try:
        # *** USE async with and await ***
        async with get_db_connection() as conn:
            # Assuming async cursor
            async with conn.cursor(dictionary=True) as cursor:
                await cursor.execute("SELECT asset_name FROM option_data.assets ORDER BY asset_name ASC")
                results = await cursor.fetchall() # await fetchall
                asset_names = [row["asset_name"] for row in results]

        if not asset_names: logger.warning("No assets found in DB.")
        return {"assets": asset_names}
    except (ConnectionError, mysql.connector.Error) as db_err:
        logger.error(f"DB Error fetching assets: {db_err}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error fetching assets")
    except Exception as e:
        logger.error(f"Unexpected error fetching assets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error fetching assets")

@app.get("/expiry_dates", tags=["Data"])
async def get_expiry_dates(asset: str = Query(..., description="Asset name (e.g., NIFTY)")): # Already async
    """ Fetches unique, sorted expiry dates (YYYY-MM-DD) for a given asset. """
    logger.info(f"Request received for expiry dates: Asset={asset}")
    sql = """
        SELECT DISTINCT DATE_FORMAT(e.expiry_date, '%Y-%m-%d') AS expiry_date_str
        FROM option_data.expiries e
        JOIN option_data.assets a ON e.asset_id = a.id
        WHERE a.asset_name = %s AND e.expiry_date >= CURDATE()
        ORDER BY expiry_date_str ASC;
    """
    expiries = []
    try:
        # *** USE async with and await ***
        async with get_db_connection() as conn:
            async with conn.cursor(dictionary=True) as cursor:
                await cursor.execute(sql, (asset,))
                results = await cursor.fetchall() # await fetchall
                expiries = [row["expiry_date_str"] for row in results]

    except (ConnectionError, mysql.connector.Error) as db_err:
        logger.error(f"DB Query Error fetching expiry dates for {asset}: {db_err}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error retrieving expiry data")
    except Exception as e:
         logger.error(f"Unexpected error fetching expiry dates for {asset}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Internal server error")

    if not expiries:
        logger.warning(f"No future expiry dates found for asset: {asset}")

    logger.info(f"Returning {len(expiries)} expiry dates for {asset}.")
    return {"expiry_dates": expiries}

@app.get("/get_option_chain", tags=["Data"])
async def get_option_chain(asset: str = Query(...), expiry: str = Query(...)): # Already async
    logger.info(f"Request received for option chain: Asset={asset}, Expiry={expiry}")
    try: datetime.strptime(expiry, '%Y-%m-%d')
    except ValueError: raise HTTPException(status_code=400, detail="Invalid expiry date format. Use YYYY-MM-DD.")

    sql = """ SELECT oc.strike_price, oc.option_type, oc.open_interest, oc.change_in_oi,
                     oc.implied_volatility, oc.last_price, oc.total_traded_volume,
                     oc.bid_price, oc.bid_qty, oc.ask_price, oc.ask_qty
              FROM option_data.option_chain AS oc JOIN option_data.assets AS a ON oc.asset_id = a.id
              JOIN option_data.expiries AS e ON oc.expiry_id = e.id
              WHERE a.asset_name = %s AND e.expiry_date = %s ORDER BY oc.strike_price ASC; """
    rows = []
    try:
        # *** USE async with and await ***
        async with get_db_connection() as conn:
            async with conn.cursor(dictionary=True) as cursor:
                 await cursor.execute(sql, (asset, expiry))
                 rows = await cursor.fetchall() # await fetchall
    except (ConnectionError, mysql.connector.Error) as db_err:
        logger.error(f"Error fetching option chain for {asset} {expiry}: {db_err}", exc_info=True);
        raise HTTPException(status_code=500, detail="Database error fetching option chain.")
    except Exception as e:
        logger.error(f"Unexpected error fetching option chain for {asset} {expiry}: {e}", exc_info=True);
        raise HTTPException(status_code=500, detail="Internal server error fetching option chain.")


    if not rows:
         logger.warning(f"No option chain data found for asset '{asset}' and expiry '{expiry}'.")
         return {"option_chain": {}}

    # Processing rows remains synchronous
    option_chain = defaultdict(lambda: {"call": None, "put": None})
    for row in rows:
        strike = row["strike_price"]; opt_type = row.get("option_type")
        data_for_type = { k: row.get(k) for k in ["last_price", "open_interest", "change_in_oi", "implied_volatility", "volume", "bid_price", "bid_qty", "ask_price", "ask_qty"] }
        if opt_type == "PE": option_chain[strike]["put"] = data_for_type
        elif opt_type == "CE": option_chain[strike]["call"] = data_for_type

    return {"option_chain": dict(option_chain)}


@app.get("/get_spot_price", response_model=SpotPriceResponse, tags=["Data"])
async def get_spot_price(asset: str = Query(...)): # Make async
    logger.info(f"Received request for spot price: Asset={asset}")
    try:
        # get_cached_option is sync, no await needed here
        option_cache_data = get_cached_option(asset)
        if not option_cache_data or not isinstance(option_cache_data.get("records"), dict):
            # Attempt fetch if cache miss/invalid and needed immediately
            logger.warning(f"Spot price cache miss/invalid for {asset}, attempting live fetch.")
            # This requires the NSELive client 'n' to be initialized
            if n:
                asset_upper = asset.upper()
                live_data = None
                if asset_upper in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
                    live_data = n.index_quote(asset_upper)
                else:
                    live_data = n.stock_quote(asset_upper) # Check if this function exists/works as expected

                if live_data and isinstance(live_data.get("underlyingValue"), (int, float)):
                    spot = live_data["underlyingValue"]
                    # Optionally update cache here if desired, though get_cached_option handles it
                else:
                     raise HTTPException(status_code=404, detail=f"Spot price could not be fetched live for {asset}")
            else:
                 raise HTTPException(status_code=503, detail=f"NSELive client not available to fetch spot price for {asset}")
        else:
            # Get from cache if valid
            spot = option_cache_data["records"].get("underlyingValue")
            if spot is None: raise HTTPException(status_code=404, detail=f"Spot price not available in cache for {asset}")

        if not isinstance(spot, (int, float)): raise ValueError("Invalid spot price type")
        return {"spot_price": round(float(spot), 2)}

    except HTTPException as e: raise e
    except Exception as e:
        logger.error(f"Error fetching spot price for {asset}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching spot price for {asset}.")


# --- In-Memory Strategy Endpoints (Keep as is - Development/Testing Only) ---
@app.post("/add_strategy", tags=["Strategy (In-Memory - Dev Only)"], include_in_schema=False)
async def add_strategy_in_memory(position: PositionInput): # Make async
    global strategy_positions; pos_dict = position.dict()
    logger.info(f"Adding in-memory position: {pos_dict}")
    strategy_positions.append(pos_dict)
    return {"status": "added", "position": pos_dict, "total_positions": len(strategy_positions)}

@app.get("/get_strategies", tags=["Strategy (In-Memory - Dev Only)"], include_in_schema=False)
async def get_strategies_in_memory(): # Make async
    global strategy_positions
    logger.info(f"Returning {len(strategy_positions)} in-memory positions")
    return strategy_positions


# --- Debug Endpoint for Flawed Background Task ---
# This endpoint expects a POST with JSON body: {"asset": "ASSET_NAME"}
@app.post("/debug/set_selected_asset", tags=["Debug"], include_in_schema=False)
async def set_selected_asset_endpoint(request: DebugAssetSelectRequest): # Expects Pydantic model correctly
    global selected_asset
    # Validate asset if necessary (e.g., check against DB assets)
    asset_name = request.asset.strip().upper()
    logger.warning(f"Setting globally selected asset to: {asset_name} (via debug endpoint)")
    selected_asset = asset_name
    # Add background task to immediately trigger an update for this asset
    # BackgroundTasks requires FastAPI BackgroundTasks parameter in the function signature
    # background_tasks.add_task(fetch_and_update_single_asset_data, asset_name)
    # Note: Adding BackgroundTasks might change how the endpoint is called or structured.
    # For simplicity of the flawed design, we just set the global variable.
    return {"message": f"Global selected asset set to {asset_name}. Background task targets this.",
            "warning": "Global state approach NOT suitable for production."}


# --- Main Analysis & Payoff Endpoint ---
@app.post("/get_payoff_chart", tags=["Analysis & Payoff"])
async def get_payoff_chart_endpoint(request: PayoffRequest):
    endpoint_name = "get_payoff_chart_endpoint"
    asset = request.asset; strategy = request.strategy
    logger.info(f"[{endpoint_name}] Request: {asset}, {len(strategy)} legs")
    if not asset or not strategy: raise HTTPException(400, "Missing asset or strategy")

    # --- Step 1: Fetch Initial Data (Sequential - needed for others) ---
    try:
        # (Fetch spot price, lot size - same as before)
        default_lot_size = get_lot_size(asset); cached_data = get_cached_option(asset); spot_price = None
        if cached_data and "records" in cached_data: spot_price = _safe_get_float(cached_data.get("records", {}), "underlyingValue")
        if spot_price is None:
             try: spot_response = await get_spot_price(asset); spot_price = spot_response.spot_price
             except Exception as e: raise HTTPException(503, f"Spot price missing: {e}") from e
        if not default_lot_size or default_lot_size <= 0 : raise HTTPException(404, f"Lot size missing/invalid for {asset}")
        if not spot_price or spot_price <= 0: raise HTTPException(503, f"Spot price missing/invalid for {asset}")
    except HTTPException as e: raise e
    except Exception as e: raise HTTPException(503, f"Server error fetching initial data: {e}") from e

    # --- Step 2: Prepare Strategy Data (Sequential - needed for others) ---
    # (Prepare data, including iv, days_to_expiry - same as before)
    prepared_strategy_data = []
    today = date.today()
    for i, leg_input in enumerate(request.strategy):
        try: # Simplified validation
            strike = float(leg_input.strike_price); premium = float(leg_input.option_price); lots_abs = int(leg_input.lots)
            opt_type = leg_input.option_type.upper(); tr_type = leg_input.tr_type.lower()
            if strike <= 0 or premium < 0 or lots_abs <= 0: raise ValueError("Invalid numeric")
            expiry_dt = datetime.strptime(leg_input.expiry_date, "%Y-%m-%d").date(); days_to_expiry = (expiry_dt - today).days
            if days_to_expiry < 0: raise ValueError("Expiry date past")
            leg_lot_size = default_lot_size
            if leg_input.lot_size is not None:
                try: temp_ls = int(leg_input.lot_size); leg_lot_size = temp_ls if temp_ls > 0 else default_lot_size
                except: pass
            iv = extract_iv(asset, strike, leg_input.expiry_date, opt_type)
            if iv is None or iv <= 0: raise ValueError(f"Missing IV for {opt_type}@{strike}")
            prepared_strategy_data.append({ "op_type": "c" if opt_type=="CE" else "p", "strike": strike, "tr_type": tr_type, "op_pr": premium, "lot": lots_abs, "lot_size": leg_lot_size, "iv": float(iv), "days_to_expiry": days_to_expiry, "expiry_date_str": leg_input.expiry_date,})
        except (ValueError, KeyError, TypeError) as e: raise HTTPException(400, f"Invalid leg {i+1}: {e}") from e
        except Exception as e: raise HTTPException(400, f"Error processing leg {i+1}: {e}") from e
    if not prepared_strategy_data: raise HTTPException(400, "No valid strategy legs provided.")


    # --- Step 3: Perform Calculations Concurrently ---
    logger.debug(f"[{endpoint_name}] Starting concurrent calculations...")
    start_calc_time = time.monotonic()
    results = {} # Dictionary to store results

    try:
        # Run potentially blocking/CPU-bound sync functions in threads
        # Use asyncio.to_thread if available (Python 3.9+)
        tasks = {
            "chart": asyncio.to_thread(generate_payoff_chart_matplotlib, prepared_strategy_data, asset),
            "metrics": asyncio.to_thread(calculate_strategy_metrics, prepared_strategy_data, asset),
            "tax": asyncio.to_thread(calculate_option_taxes, prepared_strategy_data, asset),
            "greeks": asyncio.to_thread(calculate_option_greeks, prepared_strategy_data, asset)
        }
        # If asyncio.to_thread is not available (Python < 3.9) use loop.run_in_executor:
        # loop = asyncio.get_running_loop()
        # executor = ThreadPoolExecutor() # Or reuse one if created elsewhere
        # tasks = {
        #     "chart": loop.run_in_executor(executor, generate_payoff_chart_matplotlib, prepared_strategy_data, asset),
        #     "metrics": loop.run_in_executor(executor, calculate_strategy_metrics, prepared_strategy_data, asset),
        #     "tax": loop.run_in_executor(executor, calculate_option_taxes, prepared_strategy_data, asset),
        #     "greeks": loop.run_in_executor(executor, calculate_option_greeks, prepared_strategy_data, asset)
        # }


        # Await all tasks concurrently, allow exceptions to propagate
        # Use return_exceptions=True if you want partial results even if one task fails
        task_results = await asyncio.gather(*tasks.values(), return_exceptions=False)

        # Assign results back to variables
        results["chart"], results["metrics"], results["tax"], results["greeks"] = task_results

        # Check for critical failures (chart must succeed)
        if results["chart"] is None:
             logger.error(f"[{endpoint_name}] Chart generation failed within concurrent execution.")
             raise ValueError("Payoff chart generation failed.")
        if results["metrics"] is None:
             logger.warning(f"[{endpoint_name}] Metrics calculation failed concurrently.")
        if results["tax"] is None:
             logger.warning(f"[{endpoint_name}] Tax calculation failed concurrently.")
        if not results["greeks"]: # Check if empty list was returned
             logger.warning(f"[{endpoint_name}] Greeks calculation returned empty list concurrently.")


    except Exception as e: # Catch errors from gather or the tasks themselves
        logger.error(f"[{endpoint_name}] Error during concurrent calculation phase for {asset}: {e}", exc_info=True)
        # Provide a more specific error if possible
        error_detail = f"Calculation Error: {e}" if isinstance(e, ValueError) else f"Unexpected Server Error during calculation."
        raise HTTPException(status_code=500, detail=error_detail)

    calc_duration = time.monotonic() - start_calc_time
    logger.info(f"[{endpoint_name}] Concurrent calculations finished in {calc_duration:.3f}s")

    # --- Step 4: Return Results ---
    logger.info(f"[{endpoint_name}] Successfully processed payoff request for asset: {asset}")
    return {
        "image_base64": results["chart"],
        "success": True,
        "metrics": results["metrics"], # Might be None
        "charges": results["tax"],     # Might be None
        "greeks": results["greeks"] or [] # Ensure it's a list
    }


# --- LLM Stock Analysis Endpoint ---
@app.post("/get_stock_analysis", tags=["Analysis & Payoff"])
async def get_stock_analysis_endpoint(request: StockRequest):
    asset = request.asset.upper(); analysis_cache_key = f"analysis_{asset}"
    cached_analysis = analysis_cache.get(analysis_cache_key)
    if cached_analysis: logger.info(f"Cache hit analysis: {asset}"); return {"analysis": cached_analysis}

    stock_symbol = "^NSEI" if asset == "NIFTY" else f"{asset}.NS"
    try: stock_data, latest_news = await asyncio.gather(fetch_stock_data_async(stock_symbol), fetch_latest_news_async(asset))
    except Exception as e: raise HTTPException(status_code=503, detail=f"Data fetch error for analysis: {e}") from e
    if not stock_data: raise HTTPException(status_code=404, detail=f"Stock data not found for: {stock_symbol}")

    prompt = build_stock_analysis_prompt(asset, stock_data, latest_news) # Calls the REFINED prompt func
    try:
        # *** Using gemini-1.5-flash-latest - confirmed as advanced free tier ***
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        logger.info(f"Generating Gemini analysis for {asset} using {model.model_name}...")

        # Generation safety settings (Optional - Adjust thresholds if needed)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        # Setting generation config (Optional - e.g., control randomness)
        # generation_config = genai.types.GenerationConfig(temperature=0.7) # Example

        response = await model.generate_content_async(
            prompt,
            safety_settings=safety_settings
            # generation_config=generation_config
        )

        # More robust check for valid response content
        analysis_text = ""
        if response.parts:
             # Sometimes text is within parts, concatenate if needed
             analysis_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))

        if not analysis_text:
             # Check for blocking reason if text is empty
             block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
             logger.error(f"Gemini response blocked or empty for {asset}. Reason: {block_reason}. Feedback: {response.prompt_feedback}")
             raise ValueError(f"Content blocked or generation failed. Reason: {block_reason}")

        analysis_cache[analysis_cache_key] = analysis_text
        logger.info(f"Successfully generated analysis for {asset}")
        return {"analysis": analysis_text}

    except ValueError as ve: # Catch specific value errors (like blocking)
        logger.error(f"ValueError during Gemini Generation for {asset}: {ve}")
        raise HTTPException(status_code=503, detail=f"Analysis generation failed: {ve}")
    except Exception as e:
        logger.error(f"Gemini API or processing error for {asset}: {e}", exc_info=True)
        # Provide a slightly more informative generic error
        raise HTTPException(status_code=503, detail=f"Analysis generation failed (API/Processing Error).")


# ===============================================================
# Main Execution Block (Keep as is)
# ===============================================================
if __name__ == "__main__":
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true" # Enable reload via env var if needed

    logger.info(f"Starting Uvicorn server on http://{host}:{port} (Reload: {reload})")
    uvicorn.run(
        "__main__:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )








