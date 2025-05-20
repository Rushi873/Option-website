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
import aiohttp
import functools # Import functools
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Union, Optional, Tuple, Set
from contextlib2 import asynccontextmanager
from collections import defaultdict
from functools import partial

# --- Environment & Config ---
from dotenv import load_dotenv

# --- FastAPI & Web ---
from fastapi import FastAPI, HTTPException, Query, Body, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# --- Pydantic Models ---
from pydantic import BaseModel, Field, field_validator 

# --- Data Sources ---
from jugaad_data.nse import NSELive # For option chain source
import yfinance as yf              # For stock data
import requests                    # For news scraping
from bs4 import BeautifulSoup       # For news scraping
import feedparser
from urllib.parse import quote_plus

# --- Calculation & Plotting ---
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import math

# *** Ensure Matplotlib Imports are Correct ***
# import matplotlib
# matplotlib.use('Agg') # Use non-interactive backend BEFORE importing pyplot
# import matplotlib.pyplot as plt
# # Set some optimized Matplotlib parameters globally (optional)
# # plt.style.use('fast') # potentially faster style, less visually complex
# plt.rcParams['path.simplify'] = True
# plt.rcParams['path.simplify_threshold'] = 0.6
# plt.rcParams['agg.path.chunksize'] = 10000 # Process paths in chunks
# Import SciPy optimize if needed for breakeven (ensure it's installed)
try:
    from scipy import optimize # Import the optimize module
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger = logging.getLogger(__name__) # Need logger early for this message
    logger.warning("SciPy not found. Breakeven calculation will use linear interpolation fallback.")

try:
    import mibian
    MIBIAN_AVAILABLE = True
except ImportError:
    mibian = None
    MIBIAN_AVAILABLE = False
    print("WARNING: Mibian library not found. Greeks calculation will be skipped.")


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
news_cache = TTLCache(maxsize=100, ttl=60*15)
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
PAYOFF_LOWER_FACTOR = 0.90
PAYOFF_UPPER_FACTOR = 1.10
PAYOFF_POINTS = 300
# --- Y-Axis Fitting Constants ---
Y_AXIS_PADDING_FACTOR = 0.12 # How much padding relative to the range (12%)
MIN_Y_AXIS_ABS_RANGE = 100   # Minimum absolute P/L range shown (e.g., ₹100)
# --- strategy metric ---
PAYOFF_UPPER_BOUND_FACTOR = 1.5 # Factor for BE search upper limit
BREAKEVEN_CLUSTER_GAP_PCT = 0.005 # Clustering threshold
PAYOFF_CHECK_EPSILON = 0.01 # Small value around strikes for max/min check
ZERO_TOLERANCE = 1e-6 # Tolerance for checking if payoff is zero at BE/strike

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

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")
if not RAPIDAPI_KEY or RAPIDAPI_KEY == "YOUR_FALLBACK_KEY_IF_NEEDED_FOR_TESTING_ONLY":
    logger.error("RAPIDAPI_KEY environment variable not set or is a placeholder.")
RAPIDAPI_HEADERS = {
    'x-rapidapi-key': RAPIDAPI_KEY,
    'x-rapidapi-host': RAPIDAPI_HOST
}
RAPIDAPI_BASE_URL = f"https://{RAPIDAPI_HOST}"

# ===============================================================
# Global State (Keep as is)
# ===============================================================
selected_asset: Optional[str] = "NIFTY"
strategy_positions: List[dict] = [] # Removed, as related endpoints were commented out
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


def generate_price_sample_points(
    strikes: List[Union[int, float]], # List of strike prices
    spot: float,                      # Current spot price
    more_samples: bool = True         # Flag to add extra points
) -> Set[float]:                      # Return a set for automatic uniqueness
    """
    Generate a comprehensive set of price points around strikes and spot
    for payoff calculations, ensuring non-negative values.
    """
    func_name = "generate_price_sample_points"
    logger.debug(f"[{func_name}] Generating sample points. Strikes: {strikes}, Spot: {spot}")

    # Start with essential points: 0 and current spot
    # Use a set for automatic handling of duplicates
    points: Set[float] = {0.0, spot}

    # Handle empty strikes list
    if not strikes:
        logger.debug(f"[{func_name}] No strikes provided, generating points around spot.")
        points.add(max(0.0, spot * 0.5)) # Ensure >= 0
        points.add(spot * 1.5)
        points.add(spot * 2.0)
        return {p for p in points if p >= 0} # Final filter just in case

    # Add all unique strikes
    for strike in strikes:
        if isinstance(strike, (int, float)) and strike > 0:
            points.add(float(strike))

    # Add points slightly above/below each strike for better BE detection
    if more_samples:
        for strike in strikes:
             if isinstance(strike, (int, float)) and strike > 0:
                 # Points very close to strikes
                 points.add(max(0.0, strike - 0.01))
                 points.add(strike + 0.01)
                 # Points moderately close (percentage based)
                 points.add(max(0.0, strike * 0.99))
                 points.add(strike * 1.01)

    # Add points to span the range beyond min/max strikes
    min_strike = min(s for s in strikes if isinstance(s, (int, float)) and s > 0)
    max_strike = max(s for s in strikes if isinstance(s, (int, float)) and s > 0)

    # Points significantly below lowest strike (but not negative)
    points.add(max(0.0, min_strike * 0.7))
    # Points significantly above highest strike
    points.add(max_strike * 1.3)

    # Add more intermediate points if requested and range exists
    if more_samples and max_strike > min_strike:
        try:
            # Generate points within the min/max strike range
            step = (max_strike - min_strike) / 8.0 # 7 intermediate points
            if step > 1e-6: # Avoid issues if min/max are too close
                for i in range(1, 8):
                    points.add(min_strike + step * i)
        except Exception as e:
            logger.warning(f"[{func_name}] Error generating intermediate points: {e}")

    # Final filter to ensure all points are non-negative and return the set
    final_points = {p for p in points if p >= 0}
    logger.debug(f"[{func_name}] Generated {len(final_points)} unique, non-negative sample points.")
    return final_points
    


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
    Optimized for faster upserts and handles integer strike price conversion.
    DEPENDS ON: assets table having last_spot_price, last_spot_update_time columns.
    REMAINS SYNCHRONOUS for background thread compatibility.
    """
    global n # Need access to the global NSELive client instance
    func_name = "fetch_and_update_single_asset_data"
    logger.info(f"[{func_name}] Starting live fetch & DB update for: {asset_name}")
    start_time = datetime.now()
    conn_obj = None # Initialize conn_obj for rollback handling
    option_source_data = None

    # --- Step 1: Directly Fetch Live Data ---
    try:
        if not n:
            raise RuntimeError("NSELive client ('n') is not initialized. Cannot fetch live data.")

        asset_upper = asset_name.upper()
        logger.debug(f"[{func_name}] Calling NSELive directly for {asset_upper}...")

        # Consider adding a timeout if NSELive library supports it? (Optional enhancement)
        if asset_upper in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
            option_source_data = n.index_option_chain(asset_upper)
        else:
            option_source_data = n.equities_option_chain(asset_upper)

        if not option_source_data or not isinstance(option_source_data, dict) or \
           "records" not in option_source_data or "data" not in option_source_data["records"]:
            logger.warning(f"[{func_name}] Received empty or invalid data structure from NSELive for {asset_name}.")
            # Attempt spot price update even if chain data is bad/missing
            # Get potential spot price from whatever data *was* received
            live_spot_partial = _safe_get_float(option_source_data.get("records", {}), "underlyingValue") if isinstance(option_source_data, dict) else None
            if live_spot_partial is not None:
                logger.info(f"[{func_name}] Attempting spot price update only for {asset_name} due to missing chain data. Spot: {live_spot_partial}")
                try:
                    # Use a separate connection context for this specific update
                    with get_db_connection() as spot_conn:
                         if spot_conn is None: raise ConnectionError("Failed to get DB connection for spot update.")
                         with spot_conn.cursor(dictionary=True) as spot_cursor:
                              # Fetch asset_id first
                              spot_cursor.execute("SELECT id FROM option_data.assets WHERE asset_name = %s", (asset_name,))
                              result = spot_cursor.fetchone()
                              if result:
                                   asset_id = result["id"]
                                   spot_cursor.execute(
                                        "UPDATE option_data.assets SET last_spot_price = %s, last_spot_update_time = NOW() WHERE id = %s",
                                        (live_spot_partial, asset_id)
                                   )
                                   spot_conn.commit()
                                   logger.info(f"[{func_name}] Committed spot price update only ({live_spot_partial}) for asset ID {asset_id}")
                              else: logger.error(f"[{func_name}] Asset '{asset_name}' not found in DB for spot update.")
                except Exception as spot_upd_err:
                     logger.error(f"[{func_name}] FAILED spot price only update for {asset_name}: {spot_upd_err}")
            return # Exit function after attempting spot update

        # If we reach here, basic structure seems okay
        option_data_list = option_source_data["records"]["data"]
        live_spot = option_source_data.get("records", {}).get("underlyingValue") # Already checked records exists
        logger.info(f"[{func_name}] Successfully fetched live data structure for {asset_name}. Live Spot: {live_spot}. Processing {len(option_data_list) if option_data_list else 0} data rows.")

    except RuntimeError as rte:
         logger.error(f"[{func_name}] Runtime Error during live fetch for {asset_name}: {rte}")
         return # Exit if client not ready
    except Exception as fetch_err:
        logger.error(f"[{func_name}] Error during direct NSELive fetch for {asset_name}: {fetch_err}", exc_info=True) # Log full traceback
        return # Exit if fetch fails

    # --- Step 2: Process and Update Database ---
    try:
        # Get spot price again safely for DB update (might be float)
        db_live_spot = _safe_get_float(option_source_data.get("records", {}), "underlyingValue")

        # Database interaction remains synchronous
        with get_db_connection() as conn:
            if conn is None: raise ConnectionError("Failed to get DB connection.")
            conn_obj = conn # Assign conn to conn_obj for potential rollback
            with conn.cursor(dictionary=True) as cursor:
                # Get asset_id
                cursor.execute("SELECT id FROM option_data.assets WHERE asset_name = %s", (asset_name,))
                result = cursor.fetchone()
                if not result:
                    logger.error(f"[{func_name}] Asset '{asset_name}' not found in DB. Aborting update.")
                    return
                asset_id = result["id"]

                # --- Update Spot Price in Assets Table ---
                # if db_live_spot is not None:
                #     try:
                #         cursor.execute(
                #             "UPDATE option_data.assets SET last_spot_price = %s, last_spot_update_time = NOW() WHERE id = %s",
                #             (db_live_spot, asset_id)
                #         )
                #         if cursor.rowcount > 0:
                #              logger.info(f"[{func_name}] Updated spot price ({db_live_spot}) in DB for asset ID {asset_id}")
                #         # No warning if 0 rows affected, might be same price
                #     except mysql.connector.Error as spot_upd_err:
                #          # Log as warning, but continue processing chain data if possible
                #          logger.warning(f"[{func_name}] FAILED to update spot price in DB for {asset_name} (Check columns exist): {spot_upd_err}")
                # else:
                #     logger.warning(f"[{func_name}] Could not extract spot price from live data for DB update ({asset_name}).")

                # --- Process Option Chain Data (if available) ---
                if not option_data_list:
                    logger.warning(f"[{func_name}] No option data found in live source for {asset_name}. Only spot price updated (if successful).")
                    conn.commit() # Commit the spot price update
                    logger.info(f"[{func_name}] Committed spot price update only for {asset_name}.")
                    return

                # --- Process Expiries ---
                expiry_dates_formatted = set()
                unique_expiry_parse_errors = set() # Track unique errors
                for item in option_data_list:
                    raw_expiry = item.get("expiryDate")
                    if raw_expiry:
                        try:
                            expiry_dates_formatted.add(datetime.strptime(raw_expiry, "%d-%b-%Y").strftime("%Y-%m-%d"))
                        except (ValueError, TypeError):
                             if raw_expiry not in unique_expiry_parse_errors:
                                 logger.warning(f"[{func_name}] Skipping invalid expiry format encountered: {raw_expiry}")
                                 unique_expiry_parse_errors.add(raw_expiry)
                             pass # Skip invalid formats silently after first log

                # Delete Old Expiries
                today_str = date.today().strftime("%Y-%m-%d")
                cursor.execute("DELETE FROM option_data.expiries WHERE asset_id = %s AND expiry_date < %s", (asset_id, today_str))
                logger.debug(f"[{func_name}] Deleted expiries before {today_str} for asset ID {asset_id} (Affected: {cursor.rowcount}).")

                # Upsert Current Expiries & Fetch IDs
                expiry_id_map = {}
                if expiry_dates_formatted:
                    ins_data = [(asset_id, e) for e in expiry_dates_formatted]
                    upsert_expiry_sql = "INSERT INTO option_data.expiries (asset_id, expiry_date) VALUES (%s, %s) ON DUPLICATE KEY UPDATE expiry_date = VALUES(expiry_date)"
                    cursor.executemany(upsert_expiry_sql, ins_data)
                    logger.debug(f"[{func_name}] Upserted {len(ins_data)} expiries for asset ID {asset_id} (Affected: {cursor.rowcount}).")

                    # Fetch IDs for the relevant expiries
                    placeholders = ', '.join(['%s'] * len(expiry_dates_formatted))
                    select_expiry_sql = f"SELECT id, expiry_date FROM option_data.expiries WHERE asset_id = %s AND expiry_date IN ({placeholders})"
                    cursor.execute(select_expiry_sql, (asset_id, *expiry_dates_formatted))
                    fetched_rows = cursor.fetchall()
                    for row in fetched_rows:
                         expiry_key = row["expiry_date"].strftime("%Y-%m-%d") if isinstance(row["expiry_date"], date) else str(row["expiry_date"])
                         expiry_id_map[expiry_key] = row["id"]
                    logger.debug(f"[{func_name}] Fetched {len(expiry_id_map)} expiry IDs for mapping from {len(fetched_rows)} rows.")
                else:
                    logger.warning(f"[{func_name}] No valid expiry dates extracted from live data for {asset_name}.")
                    # Decide if we should stop? If no expiries, can't process chain. Let's commit spot and exit.
                    conn.commit()
                    logger.info(f"[{func_name}] Committed spot price update and exiting due to no valid expiries found.")
                    return


                # --- Prepare Option Chain Data ---
                option_chain_data_to_upsert = []
                skipped_rows = 0
                processed_options = 0
                unique_processing_errors = defaultdict(int) # Count errors by type/message

                for item in option_data_list:
                    row_processed = False
                    try:
                        raw_strike = item.get('strikePrice')
                        raw_expiry = item.get('expiryDate')
                        if raw_strike is None or raw_expiry is None:
                            unique_processing_errors["Missing strike/expiry"] += 1
                            skipped_rows += 1; continue

                        # --- *** HANDLE STRIKE PRICE AS INT *** ---
                        strike_float = float(raw_strike)
                        # Round to nearest int. Add warning if significant difference?
                        strike_int = int(round(strike_float))
                        if abs(strike_float - strike_int) > 0.01: # Log if it wasn't close to an integer
                             logger.warning(f"[{func_name}] Strike price {strike_float} rounded to {strike_int} for DB insertion.")
                        # Use strike_int for DB operations now
                        # --- *********************************** ---

                        expiry_date_str = datetime.strptime(raw_expiry, "%d-%b-%Y").strftime("%Y-%m-%d")
                        expiry_id = expiry_id_map.get(expiry_date_str)

                        if expiry_id is None:
                             # This can happen if the expiry date was invalid earlier but still present in data
                             if expiry_date_str not in unique_processing_errors: # Log only once per expiry
                                logger.warning(f"[{func_name}] Skipping row: Expiry ID not found for date '{expiry_date_str}' (possibly invalid format earlier?). Strike: {strike_int}")
                             unique_processing_errors[expiry_date_str] += 1
                             skipped_rows += 1; continue

                        # Process CE and PE
                        for opt_type in ["CE", "PE"]:
                            details = item.get(opt_type)
                            if isinstance(details, dict) and details: # Check if details dict is not empty
                                processed_options += 1
                                # Use a robust identifier generation
                                idf = details.get("identifier")
                                if not idf:
                                     # Generate a fallback identifier if missing, ensure consistency
                                     idf = f"{asset_name.upper()}_{expiry_date_str}_{strike_int}_{opt_type}"
                                     # logger.debug(f"[{func_name}] Generated fallback identifier: {idf}")

                                row_data = (
                                    asset_id, expiry_id, strike_int, opt_type, idf, # Use strike_int
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
                                row_processed = True # Mark that at least one option (CE/PE) was processed for this strike/expiry row
                            # else: # Log only if needed, can be noisy
                                # logger.debug(f"[{func_name}] Missing or invalid details dict for {opt_type} at strike {strike_int}, expiry {expiry_date_str}.")


                    except (ValueError, TypeError, KeyError) as row_err:
                         err_key = f"Row Processing Error: {type(row_err).__name__}"
                         if unique_processing_errors[err_key] < 5: # Log first 5 occurrences
                            logger.warning(f"[{func_name}] {err_key}: {row_err}. Row snapshot: Strike={item.get('strikePrice')}, Expiry={item.get('expiryDate')}")
                         unique_processing_errors[err_key] += 1
                         skipped_rows += 1
                    except Exception as general_row_err:
                         err_key = f"Unexpected Row Error: {type(general_row_err).__name__}"
                         if unique_processing_errors[err_key] < 5:
                             logger.error(f"[{func_name}] {err_key} processing live data row: {general_row_err}", exc_info=False)
                         unique_processing_errors[err_key] += 1
                         skipped_rows += 1
                    # if not row_processed: # Increment skipped count if neither CE nor PE was processed for a valid strike/expiry row
                    #     skipped_rows += 1 # This might double count if initial checks fail

                # Log summary of processing errors if any occurred
                if unique_processing_errors:
                     logger.warning(f"[{func_name}] Summary of data processing issues: {dict(unique_processing_errors)}")

                logger.debug(f"[{func_name}] Prepared {len(option_chain_data_to_upsert)} option rows ({processed_options} options total) for {asset_name} DB upsert. Skipped {skipped_rows} input rows/options.")

                # --- Upsert Option Chain Data (Optimized) ---
                if option_chain_data_to_upsert:
                    # OPTIMIZED: Only update frequently changing data fields
                    upsert_chain_sql = """
                        INSERT INTO option_data.option_chain (
                            asset_id, expiry_id, strike_price, option_type, identifier,
                            open_interest, change_in_oi, total_traded_volume, implied_volatility,
                            last_price, bid_qty, bid_price, ask_qty, ask_price,
                            total_buy_qty, total_sell_qty
                            -- last_updated handled by DB trigger
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON DUPLICATE KEY UPDATE
                            -- Only update volatile data fields
                            open_interest=VALUES(open_interest),
                            change_in_oi=VALUES(change_in_oi),
                            total_traded_volume=VALUES(total_traded_volume),
                            implied_volatility=VALUES(implied_volatility),
                            last_price=VALUES(last_price),
                            bid_qty=VALUES(bid_qty),
                            bid_price=VALUES(bid_price),
                            ask_qty=VALUES(ask_qty),
                            ask_price=VALUES(ask_price),
                            total_buy_qty=VALUES(total_buy_qty),
                            total_sell_qty=VALUES(total_sell_qty)
                            -- No need to update identifier=VALUES(identifier) (unique key)
                            -- No need to update last_updated=NOW() (done by DB)
                    """
                    try:
                        start_upsert_time = time.monotonic()
                        # Execute using executemany for efficiency
                        cursor.executemany(upsert_chain_sql, option_chain_data_to_upsert)
                        rows_affected = cursor.rowcount
                        upsert_duration = time.monotonic() - start_upsert_time
                        # rowcount for ON DUPLICATE KEY UPDATE: 1 for each insert, 2 for each update
                        logger.info(f"[{func_name}] Upserted {len(option_chain_data_to_upsert)} rows into option_chain for {asset_name}. DB Rows Affected: {rows_affected}. Duration: {upsert_duration:.3f}s")
                    except mysql.connector.Error as upsert_err:
                         logger.error(f"[{func_name}] FAILED during option_chain upsert for {asset_name}: {upsert_err}", exc_info=True)
                         # Raise the error to trigger rollback
                         raise upsert_err
                else:
                     logger.info(f"[{func_name}] No valid option chain data prepared for DB upsert for {asset_name}.")

                # --- Commit Transaction ---
                conn.commit()
                logger.info(f"[{func_name}] Successfully committed DB updates for {asset_name}.")

    # --- Error Handling for DB/Processing Phase ---
    except (mysql.connector.Error, ConnectionError) as db_err:
        logger.error(f"[{func_name}] DB/Connection error during update phase for {asset_name}: {db_err}", exc_info=True)
        try:
            if conn_obj and conn_obj.is_connected():
                 conn_obj.rollback()
                 logger.info(f"[{func_name}] Rollback attempted due to DB error.")
        except Exception as rb_err:
            logger.error(f"[{func_name}] Rollback failed: {rb_err}")
    except Exception as e:
        logger.error(f"[{func_name}] Unexpected error during DB update phase for {asset_name}: {e}", exc_info=True)
        try:
            if conn_obj and conn_obj.is_connected():
                 conn_obj.rollback()
                 logger.info(f"[{func_name}] Rollback attempted due to unexpected error.")
        except Exception as rb_err:
            logger.error(f"[{func_name}] Rollback failed: {rb_err}")
    # `with get_db_connection()` ensures connection is released/closed

    duration = datetime.now() - start_time
    logger.info(f"[{func_name}] Finished task for asset: {asset_name}. Total Duration: {duration}")


def prepare_strategy_data(
    strategy_dicts: List[Dict[str, Any]], # Expects dicts post-Pydantic validation
    asset: str,
    spot_price: float
) -> List[Dict[str, Any]]:
    """
    Validates strategy leg dicts, calculates DTE, extracts IV, determines lot size,
    and formats for calculations. v5 Fix: Correctly handles _safe_get_float call for option_price.
    Expects input dicts with keys: 'op_type'(c/p), 'strike'(str), 'tr_type'(b/s),
    'op_pr'(str), 'lot'(str), etc.
    Outputs dicts with standard keys and types.
    """
    func_name = "prepare_strategy_data_v5" # Version tracking
    logger.info(f"[{func_name}] Preparing data for {len(strategy_dicts)} legs for asset {asset} (Spot: {spot_price}).")
    prepared_data: List[Dict[str, Any]] = []
    today = date.today()

    # --- Get Default Lot Size ---
    default_lot_size = None
    try:
        default_lot_size = get_lot_size(asset)
        if default_lot_size is None or not isinstance(default_lot_size, int) or default_lot_size <= 0:
             raise ValueError(f"Invalid default lot size ({default_lot_size}) retrieved")
        logger.debug(f"[{func_name}] Using default lot size: {default_lot_size}")
    except Exception as lot_err:
         logger.error(f"[{func_name}] Failed default lot size fetch: {lot_err}. Cannot prepare.", exc_info=True)
         return []

    # --- Process Each Leg Dictionary ---
    for i, leg_input in enumerate(strategy_dicts):
        leg_desc = f"Leg {i+1}"
        error_msg = None
        prepared_leg_base = leg_input.copy() # Keep basic info for potential storage

        try:
            logger.debug(f"[{func_name}] Processing dict {leg_desc}: {leg_input}")

            # --- Extract Raw Inputs ---
            op_type_in = str(leg_input.get('op_type', '')).strip().lower()
            strike_in = leg_input.get('strike')
            tr_type_in = str(leg_input.get('tr_type', '')).strip().lower()
            op_pr_in = leg_input.get('op_pr')
            lot_in = leg_input.get('lot')
            lot_size_in = leg_input.get('lot_size')
            iv_in = leg_input.get('iv')
            dte_in = leg_input.get('days_to_expiry')
            expiry_in = leg_input.get('expiry_date')

            # --- Basic Validation & Conversion ---

            # Validate Op Type first
            if op_type_in not in ('c', 'p'):
                error_msg = f"Invalid 'op_type' received: '{leg_input.get('op_type')}' (Expected 'c' or 'p')"
            # Validate Tr Type
            elif tr_type_in not in ('b', 's'):
                 error_msg = f"Invalid 'tr_type' ({leg_input.get('tr_type')})"

            # Convert Strike (only if no prior error)
            strike_price = None
            if error_msg is None:
                 strike_price = _safe_get_float({'sp': strike_in}, 'sp')
                 if strike_price is None or strike_price <= 0: error_msg = f"Invalid 'strike' ({strike_in})"

            # Convert Lots (only if no prior error)
            lots = None
            if error_msg is None:
                 lots = _safe_get_int({'l': lot_in}, 'l')
                 if lots is None or lots <= 0: error_msg = f"Invalid 'lot' ({lot_in})"

            # --- CONVERT AND HANDLE OPTION PRICE (Corrected) ---
            option_price = 0.0 # Default value
            if error_msg is None: # Only process if no prior critical errors
                # Call _safe_get_float WITHOUT the default argument
                option_price_raw = _safe_get_float({'op': op_pr_in}, 'op')

                # Handle the default value *after* the call
                if option_price_raw is not None: # If conversion was successful
                    if option_price_raw < 0:
                        logger.warning(f"[{func_name}] Negative premium '{op_pr_in}' for {leg_desc}, using 0.0.")
                        option_price = 0.0 # Keep as 0.0
                    else:
                        option_price = option_price_raw # Use the valid, non-negative converted value
                elif op_pr_in is not None: # Log if input existed but conversion failed
                     logger.warning(f"[{func_name}] Invalid premium value '{op_pr_in}' for {leg_desc}. Using 0.0.")
                # If option_price_raw is None and op_pr_in was None, price remains 0.0
            # --- END OF CORRECTED PRICE HANDLING ---


            # --- Determine/Validate Days to Expiry (only if no prior error) ---
            days_to_expiry = None
            if error_msg is None:
                if isinstance(dte_in, int) and dte_in >= 0:
                     days_to_expiry = dte_in
                elif isinstance(expiry_in, str) and expiry_in:
                     try:
                          expiry_dt = datetime.strptime(expiry_in, "%Y-%m-%d").date()
                          calculated_dte = (expiry_dt - today).days
                          if calculated_dte < 0: error_msg = f"Expiry date '{expiry_in}' is in the past"
                          else: days_to_expiry = calculated_dte
                     except ValueError: error_msg = f"Invalid 'expiry_date' format ({expiry_in})"
                if days_to_expiry is None and error_msg is None:
                     error_msg = "Missing valid 'days_to_expiry' or 'expiry_date'"

            # --- If any critical error occurred up to this point, skip leg ---
            if error_msg:
                logger.warning(f"[{func_name}] Skipping {leg_desc} due to error(s): {error_msg}. Input: {leg_input}")
                continue # Skip to next leg in the loop

            # --- Determine/Validate IV (Now safe to proceed) ---
            iv_float = None
            if isinstance(iv_in, (int, float)) and iv_in >= 0:
                iv_float = float(iv_in)
            else:
                 logger.debug(f"[{func_name}] {leg_desc} Provided IV invalid/missing ({iv_in}). Attempting extraction...")
                 op_type_req_for_iv = 'CE' if op_type_in == 'c' else 'PE'
                 iv_extracted = extract_iv(asset, strike_price, expiry_in, op_type_req_for_iv)
                 if iv_extracted is None or not isinstance(iv_extracted, (int, float)) or iv_extracted < 0:
                      logger.warning(f"[{func_name}] IV extraction failed/invalid for {leg_desc}. Using 0.0 placeholder.")
                      iv_float = 0.0
                 else:
                      iv_float = float(iv_extracted)

            iv_float = max(0.0, iv_float) # Ensure non-negative
            if iv_float <= 1e-6: logger.warning(f"[{func_name}] IV for {leg_desc} is zero/near-zero ({iv_float}).");

            # --- Determine Final Lot Size ---
            leg_specific_lot_size = _safe_get_int({'ls': lot_size_in}, 'ls')
            final_lot_size = default_lot_size
            if leg_specific_lot_size is not None and leg_specific_lot_size > 0:
                final_lot_size = leg_specific_lot_size
            elif lot_size_in is not None: # Log only if an invalid value was provided
                logger.warning(f"[{func_name}] Invalid leg 'lot_size' '{lot_size_in}'. Using default: {default_lot_size}")

            # --- Assemble Prepared Leg Data ---
            prepared_leg = {
                "op_type": op_type_in,        # Use 'c' or 'p' directly
                "strike": strike_price,       # float
                "tr_type": tr_type_in,        # 'b' or 's'
                "op_pr": option_price,        # float (>= 0)
                "lot": lots,                  # int (> 0)
                "lot_size": final_lot_size,   # int (> 0)
                "iv": iv_float,               # float (>= 0)
                "days_to_expiry": days_to_expiry, # int (>= 0)
                "expiry_date_str": expiry_in, # Keep original expiry for reference if needed
                # Add calculated quantity for convenience in later functions
                "quantity": lots * final_lot_size
            }
            prepared_data.append(prepared_leg)
            logger.debug(f"[{func_name}] Successfully prepared {leg_desc}: {prepared_leg}")

        except Exception as unexpected_err: # Catch any other unexpected error during processing
             logger.error(f"[{func_name}] UNEXPECTED Error preparing {leg_desc}: {unexpected_err}. Skipping leg. Input: {leg_input}", exc_info=True)
             continue # Skip leg on unexpected errors too

    # --- Final Log ---
    logger.info(f"[{func_name}] Finished preparation. Prepared {len(prepared_data)} valid legs out of {len(strategy_dicts)} input legs for asset {asset}.")
    return prepared_data
    

def format_greek(value: Optional[float], precision: int = 4) -> str:
    """Formats a float nicely for display, handles None/NaN/Inf."""
    if value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
        return "N/A"
    # Add '+' sign for positive numbers for clarity in prompt
    sign = '+' if value > 0 else ''
    return f"{sign}{value:.{precision}f}"



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


def calculate_payoff_at_expiration(strategy_data: List[Dict[str, Any]], price_at_expiry: float) -> float:
    """Calculate the payoff (excluding initial premium) of the strategy at a given expiration price."""
    total_payoff = 0.0
    price_at_expiry = max(0.0, price_at_expiry) # Ensure price >= 0

    for leg in strategy_data:
        try:
            # Expect numeric types from prepared data
            strike = leg['strike']
            quantity = leg['quantity']
            op_type = leg['op_type'] # 'c' or 'p'
            tr_type = leg['tr_type'] # 'b' or 's'

            # Calculate intrinsic value
            if op_type == "c":  # Call option
                intrinsic_value = max(0.0, price_at_expiry - strike)
            else:  # Put option
                intrinsic_value = max(0.0, strike - price_at_expiry)

            # Calculate leg's contribution to payoff (ignoring premium here)
            if tr_type == "b":  # Buy
                leg_payoff = intrinsic_value * quantity
            else:  # Sell
                leg_payoff = -intrinsic_value * quantity

            if not np.isfinite(leg_payoff):
                 logger.warning(f"Non-finite leg payoff calc: {leg}, Price: {price_at_expiry}")
                 return np.nan # Signal error if any part fails

            total_payoff += leg_payoff

        except KeyError as ke:
             logger.error(f"Missing key '{ke}' in leg data during payoff calculation: {leg}")
             return np.nan # Signal error
        except Exception as e:
             logger.error(f"Error calculating leg payoff for {leg} at price {price_at_expiry}: {e}")
             return np.nan # Signal error

    return total_payoff


def find_breakeven_points(
    strategy_data: List[Dict[str, Any]],
    net_premium: float,
    spot_price: float
) -> List[float]: # Returns list of numeric BE points
    """Find breakeven points using numerical methods (based on provided logic)."""
    func_name = "find_breakeven_points_v_user"
    if not strategy_data: return []
    if not SCIPY_AVAILABLE or optimize is None:
        logger.warning(f"[{func_name}] Scipy not available. Breakeven accuracy reduced.")
        # Could implement a pure interpolation fallback here if needed

    # Define the total P/L function (payoff at expiration + net premium)
    def pnl_function(price: float) -> float:
        payoff_at_exp = calculate_payoff_at_expiration(strategy_data, price)
        if np.isnan(payoff_at_exp): return np.nan
        return payoff_at_exp + net_premium

    breakeven_candidates = []
    try:
        # Get all strikes, ensure valid numbers
        strikes = [float(leg["strike"]) for leg in strategy_data if isinstance(leg.get("strike"), (int, float)) and leg["strike"] > 0]
        if not strikes:
            logger.warning(f"[{func_name}] No valid strikes in strategy. Using spot for range.")
            min_strike = spot_price * 0.8
            max_strike = spot_price * 1.2
        else:
            min_strike = min(strikes)
            max_strike = max(strikes)

        # Define search range
        price_range_width = max(max_strike - min_strike, spot_price * 0.5)
        lower_bound = max(0.01, min_strike - price_range_width) # Adjusted slightly from 0.1
        upper_bound = max_strike + price_range_width
        logger.debug(f"[{func_name}] BE Search Range: [{lower_bound:.2f}, {upper_bound:.2f}]")

        # Create dense grid
        num_points = 10000 # Keep density
        prices = np.linspace(lower_bound, upper_bound, num_points)

        # Calculate P/L at each price point
        payoffs = [pnl_function(price) for price in prices]

        # Find intervals where P/L crosses zero
        processed_roots = set() # Track roots found to avoid duplicates from nearby intervals

        for i in range(len(payoffs) - 1):
            y1, y2 = payoffs[i], payoffs[i+1]
            p1, p2 = prices[i], prices[i+1]

            if not (np.isfinite(y1) and np.isfinite(y2)): continue # Skip if calc failed

            # Check for sign change or zero crossing
            if (y1 * y2 <= 0): # Simpler check for crossing or touching zero
                root_found_in_interval = False
                # Try brentq first if signs strictly different
                if (y1 * y2 < 0) and SCIPY_AVAILABLE and optimize:
                    try:
                        if p2 > p1 + 1e-9: # Ensure valid interval for brentq
                            root = optimize.brentq(pnl_function, p1, p2, xtol=1e-6, rtol=1e-6, maxiter=100)
                            if root > 1e-6: # Avoid zero price
                                rounded_root = round(root, 2)
                                if rounded_root not in processed_roots:
                                    breakeven_candidates.append(root) # Add precise root
                                    processed_roots.add(rounded_root)
                                    root_found_in_interval = True
                                    logger.debug(f"[{func_name}] Found BE (brentq): {root:.4f} in [{p1:.2f}, {p2:.2f}]")
                    except ValueError: pass # brentq fails if signs are same
                    except Exception as e: logger.warning(f"[{func_name}] Brentq error in [{p1:.2f}, {p2:.2f}]: {e}")

                # Fallback/Alternative: Linear interpolation if brentq failed/skipped
                # Also check endpoints if they are zero
                if not root_found_in_interval:
                    if abs(y1) < ZERO_TOLERANCE and p1 > 1e-6:
                         rounded_root = round(p1, 2)
                         if rounded_root not in processed_roots: breakeven_candidates.append(p1); processed_roots.add(rounded_root)
                    elif abs(y2) < ZERO_TOLERANCE and p2 > 1e-6:
                         rounded_root = round(p2, 2)
                         if rounded_root not in processed_roots: breakeven_candidates.append(p2); processed_roots.add(rounded_root)
                    elif abs(y1 - y2) > 1e-9 and (y1 * y2 < 0) : # Check diff and sign change for interp
                        try:
                            breakeven = p1 + (p2 - p1) * (0 - y1) / (y2 - y1)
                            if p1 <= breakeven <= p2 and breakeven > 1e-6:
                                # Optional: Verify payoff at interpolated point
                                payoff_at_be = pnl_function(breakeven)
                                if np.isfinite(payoff_at_be) and abs(payoff_at_be) < 0.1: # Looser tolerance
                                     rounded_root = round(breakeven, 2)
                                     if rounded_root not in processed_roots:
                                         breakeven_candidates.append(breakeven)
                                         processed_roots.add(rounded_root)
                                         logger.debug(f"[{func_name}] Found BE (interp): {breakeven:.4f} in [{p1:.2f}, {p2:.2f}]")
                        except Exception as e: logger.warning(f"[{func_name}] Interpolation error in [{p1:.2f}, {p2:.2f}]: {e}")

    except Exception as e:
        logger.error(f"[{func_name}] Error finding breakeven points: {e}", exc_info=True)
        return []

    # Return sorted, unique numeric points (caller will format)
    final_points = sorted(list(set(round(be, 2) for be in breakeven_candidates))) # Ensure unique and rounded
    logger.info(f"[{func_name}] Found final breakeven points (numeric): {final_points}")
    return final_points



def calculate_max_profit_loss(
    strategy_data: List[Dict[str, Any]],
    net_premium: float
) -> tuple[Union[float, np.inf], Union[float, np.inf]]: # Return numeric/inf
    """
    Calculate maximum profit and maximum loss (based on provided logic).
    Returns tuple (max_profit, max_loss) using float('inf')/-float('inf').
    Note: User provided logic returns float('inf') for max_loss, which is unusual.
          This implementation returns -float('inf') for consistency.
    """
    func_name = "calculate_max_profit_loss_v_user"
    max_profit: Union[float, np.inf] = -np.inf # Start assuming worst profit
    max_loss: Union[float, np.inf] = np.inf   # Start assuming worst loss (positive value initially)

    try:
        # Get all strikes, ensure valid numbers
        strikes = sorted([float(leg["strike"]) for leg in strategy_data if isinstance(leg.get("strike"), (int, float)) and leg["strike"] > 0])

        # Define check points: 0, strikes, point far above highest strike
        prices_to_check = {0.0}
        if strikes:
            prices_to_check.update(strikes)
            # Check slightly around strikes
            for k in strikes:
                 prices_to_check.add(max(0.0, k - 0.01))
                 prices_to_check.add(k + 0.01)
            # Add a point far above the highest strike
            prices_to_check.add(strikes[-1] * 5) # Check further out than 10x maybe? Or use SD approx?
        else:
             prices_to_check.add(spot_price * 2) # Check around spot if no strikes

        logger.debug(f"[{func_name}] Prices to check for Max P/L: {sorted(list(prices_to_check))}")

        # Calculate total P/L (payoff + net_premium) at each point
        profits = []
        last_finite_p = 0.0
        last_finite_profit = np.nan

        for price in sorted(list(prices_to_check)):
            pnl = calculate_payoff_at_expiration(strategy_data, price) + net_premium
            if np.isfinite(pnl):
                profits.append(pnl)
                last_finite_p = price
                last_finite_profit = pnl
            else:
                logger.warning(f"[{func_name}] Non-finite P/L ({pnl}) calculated at price {price:.2f}")

        if not profits: # Handle case where no points yielded finite profit
            logger.error(f"[{func_name}] No finite profit values calculated.")
            return (np.nan, np.nan) # Signal error

        # --- Determine Max P/L based on Slope Heuristic (from provided logic) ---
        # Need at least two points to calculate slope reliably
        # Recalculate points to get a clear pair for slope check
        p_low = last_finite_p # Last point where profit was finite
        p_high = p_low * 1.5 # A point significantly higher

        profit_low = last_finite_profit
        profit_high = calculate_payoff_at_expiration(strategy_data, p_high) + net_premium

        if np.isfinite(profit_low) and np.isfinite(profit_high) and (p_high > p_low + 1e-6) :
             high_slope = (profit_high - profit_low) / (p_high - p_low)
             logger.debug(f"[{func_name}] Slope check: p_low={p_low:.2f}, p_high={p_high:.2f}, profit_low={profit_low:.2f}, profit_high={profit_high:.2f}, slope={high_slope:.4f}")

             if high_slope > 0.01:  # Profit trending upwards
                 max_profit = np.inf
                 # Max loss is the minimum observed finite P/L (or 0 if always positive)
                 max_loss_numeric = min(0.0, min(profits)) # Cap loss at 0
                 max_loss = max_loss_numeric # Keep as number
             elif high_slope < -0.01: # Profit trending downwards (Loss trending upwards)
                 max_profit = max(0.0, max(profits)) # Max profit is capped
                 max_loss = -np.inf # Loss is potentially infinite (Corrected from provided logic)
             else:  # Profit is flat
                 max_profit = max(0.0, max(profits))
                 max_loss_numeric = min(0.0, min(profits))
                 max_loss = max_loss_numeric
        else:
             # Fallback if slope check failed (e.g., only one finite point)
             logger.warning(f"[{func_name}] Could not perform slope check. Determining P/L from calculated points only.")
             max_profit = max(0.0, max(profits))
             max_loss_numeric = min(0.0, min(profits))
             max_loss = max_loss_numeric
             # Cannot determine if unbounded in this fallback case

    except Exception as e:
        logger.error(f"[{func_name}] Error calculating Max P/L: {e}", exc_info=True)
        return (np.nan, np.nan) # Signal error

    # Return numeric values (or +/- inf)
    return max_profit, max_loss


# --- R:R Calculation (Based on provided logic - outputs dict or None) ---
def calculate_risk_reward_ratio(
    max_profit: Union[float, np.inf],
    max_loss: Union[float, np.inf] # Expects numeric loss (can be -inf)
) -> Optional[Dict[str, Any]]: # Returns None if infinite, else dict
    """
    Calculate the risk-reward ratio based on user-provided logic structure.
    Handles infinite values by returning None.
    """
    func_name = "calculate_risk_reward_ratio_v_user"
    logger.debug(f"[{func_name}] Calculating R:R. MaxP={max_profit}, MaxL={max_loss}")

    # Check for calculation errors first
    if not (np.isfinite(max_profit) or max_profit == np.inf):
        logger.warning(f"[{func_name}] Invalid max_profit input: {max_profit}")
        return None # Or return specific error dict? {"description": "N/A (Max Profit Error)"}
    if not (np.isfinite(max_loss) or max_loss == -np.inf): # Check for -inf for loss
        logger.warning(f"[{func_name}] Invalid max_loss input: {max_loss}")
        return None # Or return specific error dict? {"description": "N/A (Max Loss Error)"}


    # Handle infinite risk or reward -> ratio is undefined/infinite
    if max_profit == np.inf or max_loss == -np.inf:
        desc = "Infinite risk or reward"
        if max_profit == np.inf and max_loss == -np.inf: desc = "Infinite profit and loss potential"
        elif max_profit == np.inf: desc = "Infinite profit potential"
        elif max_loss == -np.inf: desc = "Infinite loss potential"
        logger.info(f"[{func_name}] R:R is undefined due to infinite component.")
        # Returning None as per original function's intent for infinite cases
        # Alternatively, could return this dict:
        # return {"ratio": None, "raw": "N/A", "normalized": "N/A", "description": desc}
        return None # Following provided function's logic

    # Handle zero loss or zero profit cases (finite P/L)
    max_loss_abs = abs(max_loss)
    if max_loss_abs < ZERO_TOLERANCE: # Effectively zero loss
        if max_profit <= ZERO_TOLERANCE: # No profit, no loss
             desc = "Zero risk, zero reward"
             ratio_decimal = 0.0
             ratio_str = "0.00:0.00"
             normalized_ratio = "1:0.00" # Or N/A?
        else: # Positive profit, zero loss
             desc = "No risk, positive reward"
             ratio_decimal = np.inf # Technically infinite ratio
             ratio_str = "0.00:∞"
             normalized_ratio = "1:∞"
    elif max_profit <= ZERO_TOLERANCE: # Non-zero loss, but no profit
        desc = "No reward, positive risk"
        ratio_decimal = 0.0
        ratio_str = f"{max_loss_abs:.2f}:0.00"
        normalized_ratio = "1:0.00"
    # Finite profit and finite non-zero loss
    else:
        try:
            ratio_decimal = abs(max_profit / max_loss)
            ratio_str = f"{max_loss_abs:.2f}:{max_profit:.2f}"
            normalized_ratio = f"1:{ratio_decimal:.2f}"
            desc = "Limited risk and reward"
        except Exception as e:
             logger.error(f"[{func_name}] Error calculating finite R:R: {e}")
             return {"ratio": None, "raw": "N/A", "normalized": "N/A", "description": "Calculation Error"}

    result = {
        "ratio": ratio_decimal if np.isfinite(ratio_decimal) else None, # Store number or None for Inf
        "raw": ratio_str,
        "normalized": normalized_ratio,
        "description": desc
    }
    logger.info(f"[{func_name}] R:R result: {result}")
    return result


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
    "https://option-strategy-website.onrender.com",
    "http://localhost:8080",   # Common alternative dev server port - Add if needed
    "http://127.0.0.1:8000"
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
class StrategyLegInput(BaseModel): # Ensure this name is used below
    # These field names MUST match the keys sent by the frontend JS
    op_type: str
    strike: str  # Receive as string
    tr_type: str
    op_pr: str   # Receive premium as string
    lot: str     # Receive lots as string
    lot_size: Optional[str] = None # Receive as string or null
    iv: Optional[float] = None # Allow float or null
    days_to_expiry: Optional[int] = None # Allow int or null
    expiry_date: Optional[str] = None # Optional, if used
    
class PayoffRequest(BaseModel):
    asset: str
    strategy: List[StrategyLegInput] 
    
class DebugAssetSelectRequest(BaseModel): asset: str

class GreeksData(BaseModel):
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

class GreeksAnalysisRequest(BaseModel):
    asset_symbol: str = Field(..., min_length=1, description="Underlying asset symbol")
    portfolio_greeks: GreeksData = Field(..., description="Dictionary of total portfolio Greeks")




# ===============================================================
# Calculation Functions 
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
# 1. Calculate Option Taxes (with Debugging)
# ===============================================================
def calculate_option_taxes(strategy_data: List[Dict[str, Any]],  spot_price: float, asset: str) -> Optional[Dict[str, Any]]:
    """Calculates estimated option taxes based on strategy legs."""
    func_name = "calculate_option_taxes"
    logger.info(f"[{func_name}] Calculating for {len(strategy_data)} leg(s), asset: {asset}")
    logger.debug(f"[{func_name}] Input strategy_data: {strategy_data}")

    # --- Fetch Prerequisites ---
    try:
        logger.debug(f"[{func_name}] Fetching prerequisites...")
        spot_price_info = get_latest_spot_price_from_db(asset) # Use new helper
        if spot_price_info and 'spot_price' in spot_price_info:
            spot_price = _safe_get_float(spot_price_info, 'spot_price') # Use safe get
            logger.debug(f"[{func_name}] Using Spot Price from DB: {spot_price}")
        else:
             logger.debug(f"[{func_name}] Spot price from DB failed or missing key, trying cache...")
             cached_data = get_cached_option(asset)
             if not cached_data or "records" not in cached_data:
                 raise ValueError("Missing or invalid market data (cache/live) for tax calc")
             spot_price = _safe_get_float(cached_data.get("records", {}), "underlyingValue")
             logger.debug(f"[{func_name}] Using Spot Price from Cache/Live: {spot_price}")

        if spot_price is None or spot_price <= 0:
            raise ValueError(f"Spot price missing or invalid ({spot_price})")

        default_lot_size = get_lot_size(asset)
        if default_lot_size is None or default_lot_size <= 0:
            raise ValueError(f"Default lot size missing or invalid ({default_lot_size})")
        logger.debug(f"[{func_name}] Prerequisites OK - Spot: {spot_price}, Default Lot Size: {default_lot_size}")

    except ValueError as val_err:
        logger.error(f"[{func_name}] Failed initial data fetch for {asset}: {val_err}")
        return None
    except Exception as e:
        logger.error(f"[{func_name}] Unexpected error fetching initial data for {asset}: {e}", exc_info=True)
        return None

    # --- Initialize Totals ---
    totals = defaultdict(float)
    breakdown = []
    logger.debug(f"[{func_name}] Initialized totals and breakdown.")

    # --- Process Each Leg ---
    for i, leg in enumerate(strategy_data):
        leg_desc = f"Leg {i+1}"
        try:
            logger.debug(f"[{func_name}] Processing {leg_desc}: {leg}")
            # --- Extract and Validate Leg Data ---
            tr_type = str(leg.get('tr_type', '')).lower()
            op_type = str(leg.get('op_type', '')).lower() # 'c' or 'p' expected
            strike = _safe_get_float(leg, 'strike')
            premium = _safe_get_float(leg, 'op_pr')
            lots = _safe_get_int(leg, 'lot')
            logger.debug(f"[{func_name}] {leg_desc} Extracted: tr={tr_type}, op={op_type}, K={strike}, prem={premium}, lots={lots}")

            # Determine Lot Size
            leg_lot_size = default_lot_size # Start with default
            raw_ls = leg.get('lot_size')
            if raw_ls is not None:
                temp_ls = _safe_get_int({'ls': raw_ls}, 'ls')
                if temp_ls is not None and temp_ls > 0:
                    leg_lot_size = temp_ls
                    logger.debug(f"[{func_name}] {leg_desc} Using leg-specific lot size: {leg_lot_size}")
            logger.debug(f"[{func_name}] {leg_desc} Final lot size: {leg_lot_size}")

            # Validate parameters
            error_msg = None
            if tr_type not in ['b','s']: error_msg = f"invalid tr_type ({tr_type})"
            elif op_type not in ['c','p']: error_msg = f"invalid op_type ({op_type})"
            elif strike is None or strike <= 0: error_msg = f"invalid strike ({strike})"
            elif premium is None or premium < 0: error_msg = f"invalid premium ({premium})"
            elif lots is None or lots <= 0: error_msg = f"invalid lots ({lots})"
            elif leg_lot_size <= 0: error_msg = f"invalid lot_size ({leg_lot_size})"

            if error_msg:
                 logger.error(f"[{func_name}] Validation failed for {leg_desc}: {error_msg}")
                 raise ValueError(f"Leg {i}: {error_msg}. Data: {leg}")
            logger.debug(f"[{func_name}] {leg_desc} Validation passed.")

            # --- Calculate Charges for the Leg ---
            quantity = lots * leg_lot_size
            turnover = premium * quantity
            logger.debug(f"[{func_name}] {leg_desc} Quantity: {quantity}, Turnover: {turnover:.2f}")
            stt_val = 0.0; stt_note = ""; intrinsic = 0.0; is_itm = False

            if tr_type == 's': # Sell side STT
                stt_val = turnover * STT_SHORT_RATE
                stt_note = f"{STT_SHORT_RATE*100:.4f}% STT (Sell Premium)"
                logger.debug(f"[{func_name}] {leg_desc} Sell STT calculated: {stt_val:.2f}")
            elif tr_type == 'b': # Buy side STT (estimate)
                if op_type == 'c' and spot_price > strike:
                    intrinsic = (spot_price - strike) * quantity; is_itm = True
                elif op_type == 'p' and spot_price < strike:
                    intrinsic = (strike - spot_price) * quantity; is_itm = True
                logger.debug(f"[{func_name}] {leg_desc} Buy side ITM check: {is_itm}, Est. Intrinsic: {intrinsic:.2f}")
                if is_itm and intrinsic > 0:
                    stt_val = intrinsic * STT_EXERCISE_RATE
                    stt_note = f"{STT_EXERCISE_RATE*100:.4f}% STT (Est. Exercise ITM)"
                    logger.debug(f"[{func_name}] {leg_desc} Buy STT calculated: {stt_val:.2f}")
                else:
                    stt_note = "No STT (Est. Buy OTM/ATM)"; logger.debug(f"[{func_name}] {leg_desc} No Buy STT.")

            # Other charges
            stamp_duty_value = turnover * STAMP_DUTY_RATE if tr_type == 'b' else 0.0
            sebi_fee_value = turnover * SEBI_RATE
            txn_charge_value = turnover * NSE_TXN_CHARGE_RATE
            brokerage_value = float(BROKERAGE_FLAT_PER_ORDER)
            gst_on_charges = (brokerage_value + sebi_fee_value + txn_charge_value) * GST_RATE
            logger.debug(f"[{func_name}] {leg_desc} Other Charges: Stamp={stamp_duty_value:.2f}, SEBI={sebi_fee_value:.4f}, Txn={txn_charge_value:.4f}, Broker={brokerage_value:.2f}, GST={gst_on_charges:.2f}")

            # --- Accumulate Totals ---
            totals["stt"] += stt_val
            totals["stamp_duty"] += stamp_duty_value
            totals["sebi_fee"] += sebi_fee_value
            totals["txn_charges"] += txn_charge_value
            totals["brokerage"] += brokerage_value
            totals["gst"] += gst_on_charges
            logger.debug(f"[{func_name}] {leg_desc} Totals Updated: {dict(totals)}")

            # --- Store Leg Breakdown ---
            leg_total_cost = stt_val + stamp_duty_value + sebi_fee_value + txn_charge_value + brokerage_value + gst_on_charges
            leg_breakdown_data = {
                "leg_index": i, "transaction_type": tr_type.upper(), "option_type": op_type.upper(),
                "strike": strike, "lots": lots, "lot_size": leg_lot_size, "quantity": quantity,
                "premium_per_share": round(premium, 2), "turnover": round(turnover, 2),
                "stt": round(stt_val, 2), "stt_note": stt_note,
                "stamp_duty": round(stamp_duty_value, 2), "sebi_fee": round(sebi_fee_value, 4),
                "txn_charge": round(txn_charge_value, 4), "brokerage": round(brokerage_value, 2),
                "gst": round(gst_on_charges, 2), "total_leg_cost": round(leg_total_cost, 2)
            }
            breakdown.append(leg_breakdown_data)
            logger.debug(f"[{func_name}] {leg_desc} Breakdown stored: {leg_breakdown_data}")

        except ValueError as val_err:
            logger.error(f"[{func_name}] Validation Error processing {leg_desc} for {asset}: {val_err}")
            raise ValueError(f"Invalid data in tax leg {i+1}: {val_err}") from val_err
        except Exception as leg_err:
            logger.error(f"[{func_name}] Unexpected Error processing tax {leg_desc} for {asset}: {leg_err}", exc_info=True)
            raise ValueError(f"Unexpected error processing tax leg {i+1}") from leg_err

    # --- Finalize and Return ---
    final_charges_summary = {k: round(v, 4) for k, v in totals.items()} # Use more decimals for summary? Maybe 4?
    final_total_cost = round(sum(totals.values()), 2)

    logger.info(f"[{func_name}] Calculation complete for {asset}. Total estimated charges: {final_total_cost:.2f}")
    logger.debug(f"[{func_name}] Final Charges Summary: {final_charges_summary}")
    logger.debug(f"[{func_name}] Final Breakdown: {breakdown}")

    rate_info = {
        "STT_SHORT_RATE": STT_SHORT_RATE, "STT_EXERCISE_RATE": STT_EXERCISE_RATE,
        "STAMP_DUTY_RATE": STAMP_DUTY_RATE, "SEBI_RATE": SEBI_RATE,
        "NSE_TXN_CHARGE_RATE": NSE_TXN_CHARGE_RATE, "GST_RATE": GST_RATE,
        "BROKERAGE_FLAT_PER_ORDER": BROKERAGE_FLAT_PER_ORDER
    }
    result = {
        "calculation_details": { "asset": asset, "spot_price_used": spot_price, "default_lot_size_used": default_lot_size, "rate_info": rate_info },
        "total_estimated_cost": final_total_cost,
        "charges_summary": final_charges_summary,
        "breakdown_per_leg": breakdown
    }
    logger.debug(f"[{func_name}] Returning result: {result}")
    return result

 

# --- Main Function ---
def generate_payoff_chart_matplotlib(
    strategy_data: List[Dict[str, Any]], # Expects data already prepared (correct types)
    asset: str,
    spot_price: float,
    strategy_metrics: Optional[Dict[str, Any]], # Parameter present but potentially unused here
    payoff_points: int = PAYOFF_POINTS,
    lower_factor: float = PAYOFF_LOWER_FACTOR,
    upper_factor: float = PAYOFF_UPPER_FACTOR
) -> Optional[str]:
    func_name = "generate_payoff_chart_plotly_std_range_v2" # Reverted logic name
    logger.info(f"[{func_name}] Generating Plotly chart JSON for {len(strategy_data)} leg(s), asset: {asset}, Spot: {spot_price}")
    start_time = time.monotonic()

    try:
        # --- Validate Spot Price ---
        if spot_price is None or not isinstance(spot_price, (int, float)) or spot_price <= 0:
             raise ValueError(f"Invalid spot_price ({spot_price})")
        spot_price = float(spot_price)

        # --- Get Default Lot Size (Still needed for fallback in payoff calc) ---
        default_lot_size = get_lot_size(asset)
        if default_lot_size is None or not isinstance(default_lot_size, int) or default_lot_size <= 0:
             # Log error but maybe allow proceeding if prepare_data guarantees leg lot_size?
             # For safety, let's raise an error if default is invalid, assuming prepare_data might rely on it.
             raise ValueError(f"Invalid default lot size ({default_lot_size}) for {asset}")
        logger.debug(f"[{func_name}] Using default lot size: {default_lot_size} (as fallback if needed)")

        # --- Calculate Standard Deviation Move ---
        one_std_dev_move = None
        if strategy_data:
            try:
                # Find first leg with valid IV and DTE (more robust than just taking first leg)
                valid_leg_for_sd = next((leg for leg in strategy_data if _safe_get_float(leg, 'iv') is not None and _safe_get_int(leg, 'days_to_expiry') is not None), None)
                if valid_leg_for_sd:
                    iv_used = _safe_get_float(valid_leg_for_sd, 'iv')
                    dte_used = _safe_get_int(valid_leg_for_sd, 'days_to_expiry')
                    if iv_used is not None and dte_used is not None and iv_used > 0 and spot_price > 0 and dte_used >= 0:
                        calc_dte = max(dte_used, 0.1)
                        time_fraction = calc_dte / 365.0
                        one_std_dev_move = spot_price * iv_used * math.sqrt(time_fraction)
                        logger.debug(f"[{func_name}] Calculated 1 StDev move: {one_std_dev_move:.2f} (using IV={iv_used:.4f}, DTE={dte_used})")
                    else:
                         logger.debug(f"[{func_name}] Could not calculate SD move from leg data (IV={iv_used}, DTE={dte_used})")
                else:
                     logger.debug(f"[{func_name}] No leg found with valid IV/DTE for SD calculation.")
            except Exception as sd_calc_err:
                 logger.error(f"[{func_name}] Error calculating SD move: {sd_calc_err}", exc_info=False)


        # --- Determine Chart X-axis Range ---
        sd_factor = 2.5
        chart_min = 0; chart_max = 0
        if one_std_dev_move and one_std_dev_move > 0:
            chart_min = spot_price - sd_factor * one_std_dev_move
            chart_max = spot_price + sd_factor * one_std_dev_move
            logger.debug(f"[{func_name}] X-axis range based on {sd_factor} StDev.")
        else:
            chart_min = spot_price * lower_factor
            chart_max = spot_price * upper_factor
            logger.debug(f"[{func_name}] X-axis range based on factors {lower_factor}/{upper_factor}.")
        lower_bound = max(chart_min, 0.1)
        upper_bound = max(chart_max, lower_bound + 1.0)
        logger.debug(f"[{func_name}] Final X-axis range: [{lower_bound:.2f}, {upper_bound:.2f}] using {payoff_points} points.")
        price_range = np.linspace(lower_bound, upper_bound, payoff_points)

        # --- Calculate Payoff Data ---
        # Assumes strategy_data contains prepared legs with correct numeric types
        total_payoff = np.zeros_like(price_range)
        processed_legs_count = 0
        for i, leg in enumerate(strategy_data):
            leg_desc = f"Leg {i+1}"
            try:
                tr_type = leg['tr_type']
                op_type = leg['op_type']
                strike = leg['strike']
                premium = leg['op_pr']
                # Use calculated quantity if available from prepare_data, else calculate again
                quantity = leg.get('quantity')
                if quantity is None: # Fallback calculation if 'quantity' missing
                     lots = leg['lot']
                     leg_lot_size = leg['lot_size'] # Should exist after prepare_data
                     quantity = lots * leg_lot_size

                leg_prem_tot = premium * quantity

                if op_type == 'c': intrinsic_value = np.maximum(price_range - strike, 0)
                else: intrinsic_value = np.maximum(strike - price_range, 0)

                if tr_type == 'b': leg_payoff = (intrinsic_value * quantity) - leg_prem_tot
                else: leg_payoff = leg_prem_tot - (intrinsic_value * quantity)

                total_payoff += leg_payoff
                processed_legs_count += 1
            except KeyError as ke:
                 logger.error(f"[{func_name}] Missing expected key '{ke}' in prepared leg data for {leg_desc}. Leg Data: {leg}", exc_info=False)
                 raise ValueError(f"Error processing prepared {leg_desc} due to missing key '{ke}'") from ke
            except Exception as e:
                 logger.error(f"[{func_name}] Error processing prepared {leg_desc} data for chart: {e}. Leg Data: {leg}", exc_info=False)
                 raise ValueError(f"Error processing prepared {leg_desc} for chart: {e}") from e

        if processed_legs_count == 0:
            logger.warning(f"[{func_name}] No valid legs processed for chart generation: {asset}.")
            return None
        logger.debug(f"[{func_name}] Payoff calculation complete for {processed_legs_count} legs.")


        # +++ STANDARD Y-Axis Range Calculation (Auto-Fit Logic) +++
        final_yaxis_range = None
        finite_payoffs = total_payoff[np.isfinite(total_payoff)]

        if len(finite_payoffs) == 0:
            logger.error(f"[{func_name}] No finite payoff values calculated. Cannot determine Y-axis range.")
            return None

        payoff_min = np.min(finite_payoffs)
        payoff_max = np.max(finite_payoffs)
        logger.info(f"[{func_name}] Calculated Actual Finite Payoff Range: [{payoff_min:.2f}, {payoff_max:.2f}]")

        logger.info(f"[{func_name}] Applying standard auto-fit Y-axis padding.")
        payoff_range = payoff_max - payoff_min

        # Calculate base padding
        if payoff_range < 1e-6: # Flat line case
            padding = max(abs(payoff_min) * Y_AXIS_PADDING_FACTOR, MIN_Y_AXIS_ABS_RANGE / 2.0)
            logger.debug(f"[{func_name}] Flat line detected. Using padding: {padding:.2f}")
        else: # Normal range case
            padding = payoff_range * Y_AXIS_PADDING_FACTOR
            logger.debug(f"[{func_name}] Calculated base padding: {padding:.2f}")

        # Calculate initial padded range
        yaxis_min = payoff_min - padding
        yaxis_max = payoff_max + padding

        # Ensure zero line visibility if range doesn't cross zero
        if yaxis_min > 0: # All positive payoffs
            yaxis_min = min(0.0, yaxis_min - padding * 0.5)
            logger.debug(f"[{func_name}] Adjusted yaxis_min for zero visibility (all positive): {yaxis_min:.2f}")
        elif yaxis_max < 0: # All negative payoffs
            yaxis_max = max(0.0, yaxis_max + padding * 0.5)
            logger.debug(f"[{func_name}] Adjusted yaxis_max for zero visibility (all negative): {yaxis_max:.2f}")

        # Enforce Minimum Absolute Range
        current_abs_range = yaxis_max - yaxis_min
        if current_abs_range < MIN_Y_AXIS_ABS_RANGE:
            needed_extra_padding = (MIN_Y_AXIS_ABS_RANGE - current_abs_range) / 2.0
            yaxis_min -= needed_extra_padding
            yaxis_max += needed_extra_padding
            logger.debug(f"[{func_name}] Enforced minimum absolute range. Added padding: {needed_extra_padding:.2f}. New range: [{yaxis_min:.2f}, {yaxis_max:.2f}]")

        # Final Sanity Check
        if yaxis_min >= yaxis_max:
            logger.error(f"[{func_name}] Invalid Standard Y-axis range calculated (min >= max): [{yaxis_min:.2f}, {yaxis_max:.2f}]. Falling back.")
            avg_payoff = np.mean(finite_payoffs)
            yaxis_min = avg_payoff - MIN_Y_AXIS_ABS_RANGE
            yaxis_max = avg_payoff + MIN_Y_AXIS_ABS_RANGE

        final_yaxis_range = [yaxis_min, yaxis_max]
        logger.info(f"[{func_name}] Final Y-Axis range set to: [{final_yaxis_range[0]:.2f}, {final_yaxis_range[1]:.2f}]")
        # +++++++++++++++++++++++++++++++++++++++


        # --- Create Plotly Figure ---
        fig = go.Figure()
        hovertemplate = ("<b>Spot Price:</b> %{x:,.2f}<br>"
                         "<b>P/L:</b> %{y:,.2f}<extra></extra>")

        # --- Add Traces ---
        fig.add_trace(go.Scatter( x=price_range, y=total_payoff, mode='lines', name='Payoff', line=dict(color='mediumblue', width=2.5), hovertemplate=hovertemplate ))
        profit_color = 'rgba(144, 238, 144, 0.4)'; loss_color = 'rgba(255, 153, 153, 0.4)'; x_fill = np.concatenate([price_range, price_range[::-1]]);
        payoff_for_profit_fill = np.maximum(total_payoff, 0); y_profit_fill = np.concatenate([np.zeros_like(price_range), payoff_for_profit_fill[::-1]]); fig.add_trace(go.Scatter(x=x_fill, y=y_profit_fill, fill='toself', mode='none', fillcolor=profit_color, hoverinfo='skip', name='Profit Zone'));
        payoff_for_loss_fill = np.minimum(total_payoff, 0); y_loss_fill = np.concatenate([payoff_for_loss_fill, np.zeros_like(price_range)[::-1]]); fig.add_trace(go.Scatter(x=x_fill, y=y_loss_fill, fill='toself', mode='none', fillcolor=loss_color, hoverinfo='skip', name='Loss Zone'));

        # --- Add Lines and Annotations ---
        fig.add_hline(y=0, line=dict(color='rgba(0, 0, 0, 0.7)', width=1.0, dash='solid'))
        fig.add_vline(x=spot_price, line=dict(color='dimgrey', width=1.5, dash='dash'))
        fig.add_annotation( x=spot_price, y=0.98, yref="paper", text=f"Spot {spot_price:.2f}", showarrow=False, yshift=10, font=dict(color='dimgrey', size=12, family="Arial"), bgcolor="rgba(255,255,255,0.7)" )
        if one_std_dev_move is not None and one_std_dev_move > 0:
            levels = [-2, -1, 1, 2]; sig_color = 'rgba(100, 100, 100, 0.8)';
            for level in levels:
                sd_price = spot_price + level * one_std_dev_move;
                if lower_bound < sd_price < upper_bound:
                     label = f"{level:+}σ"; fig.add_vline(x=sd_price, line=dict(color=sig_color, width=1, dash='dot'));
                     fig.add_annotation( x=sd_price, y=0.98, yref="paper", text=label, showarrow=False, yshift=10, font=dict(color=sig_color, size=11, family="Arial"), bgcolor="rgba(255,255,255,0.7)" )

        # --- Update Layout ---
        fig.update_layout(
            xaxis_title_text="Underlying Spot Price",
            yaxis_title_text="Profit / Loss (₹)",
            xaxis_title_font=dict(size=13),
            yaxis_title_font=dict(size=13),
            hovermode="x unified",
            showlegend=False,
            template='plotly_white',
            xaxis=dict( gridcolor='rgba(220, 220, 220, 0.7)', zeroline=False, tickformat=",.0f" ),
            yaxis=dict( gridcolor='rgba(220, 220, 220, 0.7)', zeroline=False, tickprefix="₹", tickformat=',.0f', range=final_yaxis_range ), # Applies calculated range
            margin=dict(l=60, r=30, t=35, b=50),
            font=dict(family="Arial, sans-serif", size=12)
        )

        # --- Generate JSON Output ---
        try:
            logger.debug(f"[{func_name}] Final Figure Data before JSON conversion: {fig.data}")
            logger.debug(f"[{func_name}] Final Figure Layout before JSON conversion: {fig.layout}")
        except Exception as log_err:
            logger.warning(f"[{func_name}] Could not log final figure details: {log_err}")

        logger.debug(f"[{func_name}] Generating Plotly Figure JSON...")
        figure_json_string = pio.to_json(fig, pretty=False)
        logger.debug(f"[{func_name}] Plotly JSON generation successful.")

        duration = time.monotonic() - start_time
        logger.info(f"[{func_name}] Plotly JSON generation finished in {duration:.3f}s.")
        return figure_json_string

    except ValueError as val_err:
        logger.error(f"[{func_name}] Value Error generating chart for {asset}: {val_err}", exc_info=False)
        return None
    except ImportError as imp_err:
         logger.critical(f"[{func_name}] Import Error (Plotly/Numpy missing?): {imp_err}", exc_info=False)
         return None
    except Exception as e:
        logger.error(f"[{func_name}] Unexpected error generating chart for {asset}: {e}", exc_info=True)
        return None


# ===============================================================
# 3. Calculate Strategy Metrics (Updated with new.py logic)
# ===============================================================
def calculate_strategy_metrics(
    strategy_data: List[Dict[str, Any]], # Expects PREPARED data
    spot_price: float,
    asset: str,
) -> Optional[Dict[str, Any]]:
    """
    Calculates P/L metrics using the simplified logic provided.
    v_integrated_simple: Integrates user-provided logic for BE, Max P/L, R:R.
    """
    func_name = "calculate_strategy_metrics_integrated_simple"
    logger.info(f"[{func_name}] Calculating metrics for {len(strategy_data)} leg(s), Asset: {asset}, Spot: {spot_price}")

    # --- Input Validation ---
    if not strategy_data: logger.error(f"[{func_name}] Received empty strategy_data list."); return None
    if spot_price is None or not isinstance(spot_price, (int, float)) or spot_price <= 0: logger.error(f"[{func_name}] Invalid spot_price ({spot_price})."); return None
    spot_price = float(spot_price)
    default_lot_size = get_lot_size(asset) # Fetch once, assuming it doesn't fail based on earlier checks
    if not default_lot_size: default_lot_size = 1 # Minimal fallback

    # --- 1. Calculate Net Premium (Using simplified loop) ---
    net_premium = 0.0
    cost_breakdown = [] # Still generate this
    legs_for_payoff = [] # Need this for helpers
    processed_legs = 0; skipped_legs = 0
    premium_calc_errors = 0

    print(f"\n>>> DEBUG NET PREMIUM (Integrated): Starting Calculation <<<") # DEBUG
    for i, leg in enumerate(strategy_data):
        leg_desc = f"Leg {i+1}"
        try:
            # Extract and validate core fields for premium and payoff
            tr_type = str(leg['tr_type']).lower()
            op_type = str(leg['op_type']).lower()
            strike = float(leg['strike']) # Expect float
            premium = float(leg['op_pr']) # Expect float
            lots = int(leg['lot']) # Expect int
            leg_lot_size = int(leg['lot_size']) # Expect int
            quantity = int(leg.get('quantity', lots * leg_lot_size)) # Expect int

            if tr_type not in ('b', 's') or premium < 0 or quantity <= 0 or strike <=0 or lots <= 0 or leg_lot_size <= 0:
                raise ValueError("Invalid core leg data for calculation")

            # Calculate net premium contribution
            premium_sign = -1 if tr_type == "b" else 1
            leg_premium_contribution = premium_sign * premium * quantity
            if not np.isfinite(leg_premium_contribution):
                 raise ValueError("Non-finite leg premium calculated")
            net_premium += leg_premium_contribution

            # Prepare for helpers and cost breakdown
            legs_for_payoff.append({
                'tr_type': tr_type, 'op_type': op_type, 'strike': strike,
                'premium': premium, 'quantity': quantity
            })
            action_verb = "Paid" if tr_type == 'b' else "Received"
            cost_bd_leg = { "leg_index": i, "action": action_verb, "type": op_type.upper(), "strike": strike, "premium_per_share": premium, "lots": lots, "lot_size": leg_lot_size, "quantity": quantity, "total_premium": round(-leg_premium_contribution, 2), "effect": action_verb }; cost_breakdown.append(cost_bd_leg)
            processed_legs += 1
            print(f">>> DEBUG NET PREMIUM (Integrated): Leg {i+1}: Type={tr_type.upper()}, Contribution={leg_premium_contribution:.2f}, RunningTotal={net_premium:.2f}") # DEBUG

        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"[{func_name}] Skipping leg {i+1} during processing: {e}. Data: {leg}")
            skipped_legs += 1
            premium_calc_errors += 1
            continue

    if processed_legs == 0: logger.error(f"[{func_name}] No valid legs processed."); return None
    if not np.isfinite(net_premium) or premium_calc_errors > 0:
        logger.warning(f"[{func_name}] Net premium calculation encountered issues. Final value: {net_premium}")
        # Decide if calculation should proceed or return error? Let's proceed but warn.
        calculation_warnings = ["Net premium calculation issues."]
        net_premium = 0.0 # Use fallback if non-finite
    else:
        calculation_warnings = []
    print(f">>> DEBUG NET PREMIUM (Integrated): Final Value = {net_premium:.4f}") # DEBUG
    logger.info(f"[{func_name}] Net Premium calculated: {net_premium:.2f}")


    # --- 2. Find Breakeven Points (Using new helper) ---
    print("\n>>> DEBUG BREAKEVEN (Integrated): Starting Calculation <<<") # DEBUG
    breakeven_points_numeric = find_breakeven_points(legs_for_payoff, net_premium, spot_price)
    # Format for output
    if breakeven_points_numeric:
        breakeven_points_display = [f"{be:.2f}" for be in sorted(breakeven_points_numeric)]
    else:
        breakeven_points_display = "N/A" # Simple N/A if list is empty
    print(f">>> DEBUG BREAKEVEN (Integrated): Final display value: {breakeven_points_display}") # DEBUG
    logger.info(f"[{func_name}] Breakeven points calculated: {breakeven_points_display}")


    # --- 3. Calculate Max Profit / Loss (Using new helper) ---
    print("\n>>> DEBUG MAX P/L (Integrated): Starting Calculation <<<") # DEBUG
    max_profit_numeric, max_loss_numeric = calculate_max_profit_loss(legs_for_payoff, net_premium)
    print(f">>> DEBUG MAX P/L (Integrated): Raw values: MaxP={max_profit_numeric}, MaxL={max_loss_numeric}") # DEBUG

    # Format for output (Handle np.nan which signals error in helper)
    max_profit_str = "N/A"; max_loss_str = "N/A"; # Default
    if max_profit_numeric == np.inf: max_profit_str = "∞"
    elif np.isfinite(max_profit_numeric): max_profit_str = f"{max_profit_numeric:.2f}"
    elif np.isnan(max_profit_numeric): max_profit_str = "N/A (Calc Error)"

    # Note: Corrected max_loss logic expects -np.inf for infinite loss
    if max_loss_numeric == -np.inf: max_loss_str = "-∞"
    elif np.isfinite(max_loss_numeric): max_loss_str = f"{max_loss_numeric:.2f}" # Loss is usually negative
    elif np.isnan(max_loss_numeric): max_loss_str = "N/A (Calc Error)"
    print(f">>> DEBUG MAX P/L (Integrated): Formatted: MaxP='{max_profit_str}', MaxL='{max_loss_str}'") # DEBUG
    logger.info(f"[{func_name}] Max P/L formatted: MaxP='{max_profit_str}', MaxL='{max_loss_str}'")


    # --- 4. Calculate Risk-Reward Ratio (Using new helper) ---
    print("\n>>> DEBUG R:R (Integrated): Starting Calculation <<<") # DEBUG
    rr_result_dict = calculate_risk_reward_ratio(max_profit_numeric, max_loss_numeric) # Pass numeric values
    print(f">>> DEBUG R:R (Integrated): Helper output: {rr_result_dict}") # DEBUG

    # Format for final output (extract desired string representation)
    reward_to_risk_ratio_str = "N/A" # Default
    if rr_result_dict:
        # Prioritize 'normalized' or 'raw' based on preference, fallback to description
        reward_to_risk_ratio_str = rr_result_dict.get("normalized") or rr_result_dict.get("raw") or rr_result_dict.get("description", "N/A")
        # Handle infinite cases based on description if needed for specific string output
        if "Infinite risk or reward" in rr_result_dict.get("description", ""):
            if max_profit_numeric == np.inf and max_loss_numeric == -np.inf: reward_to_risk_ratio_str = "∞ / ∞"
            elif max_profit_numeric == np.inf: reward_to_risk_ratio_str = "∞"
            elif max_loss_numeric == -np.inf: reward_to_risk_ratio_str = "N/A" # Undefined
        elif "Zero risk" in rr_result_dict.get("description", ""):
             if max_profit_numeric > 0: reward_to_risk_ratio_str = "∞"
             else: reward_to_risk_ratio_str = "0 / 0"
        elif "No reward" in rr_result_dict.get("description", ""):
             reward_to_risk_ratio_str = "Loss"
        elif rr_result_dict.get("ratio") is not None and np.isfinite(rr_result_dict.get("ratio")):
             reward_to_risk_ratio_str = f"{rr_result_dict['ratio']:.2f}" # Use formatted ratio if available
        elif "Error" in rr_result_dict.get("description", ""):
             reward_to_risk_ratio_str = "N/A (Calc Error)"


    print(f">>> DEBUG R:R (Integrated): Final display string: '{reward_to_risk_ratio_str}'") # DEBUG
    logger.info(f"[{func_name}] R:R ratio formatted: '{reward_to_risk_ratio_str}'")
    
    # if 'total_net_premium' not in locals() and 'total_net_premium' not in globals():
    #      logger.error(f"[{func_name}] CRITICAL: total_net_premium became undefined before final result. Defaulting to 0.")
    #      total_net_premium = 0.0 # Assign a default value

    # --- Prepare Final Result Dictionary ---
    # Ensure keys match what the frontend expects based on original versions
    result = {
        "calculation_inputs": {
            "asset": asset, "spot_price_used": round(spot_price, 2),
            "default_lot_size": default_lot_size if default_lot_size else "N/A",
            "num_legs_processed": processed_legs, "num_legs_skipped": skipped_legs
        },
        "metrics": {
             "max_profit": max_profit_str,                 # String
             "max_loss": max_loss_str,                   # String
             "breakeven_points": breakeven_points_display, # List[str] or "N/A" string
             "reward_to_risk_ratio": reward_to_risk_ratio_str, # String
             "net_premium": round(net_premium, 2),        # Number
             "warnings": calculation_warnings             # List of strings
        },
        "cost_breakdown_per_leg": cost_breakdown
    }
    logger.debug(f"[{func_name}] Returning final metrics result.")
    return result

# ===============================================================
# 4. Calculate Option Greeks (Updated with new.py logic)
# ===============================================================
def calculate_option_greeks(
    strategy_data: List[Dict[str, Any]], # Expects PREPARED data with numeric types
    asset: str,
    spot_price: Optional[float] = None, # Allow spot to be passed in
    interest_rate_pct: float = DEFAULT_INTEREST_RATE_PCT
) -> List[Dict[str, Any]]:
    """
    Calculates per-share and per-lot option Greeks using Mibian.
    Expects prepared strategy data with correct numeric types.
    Handles potential Mibian errors and invalid inputs gracefully.
    v6: Ensures correct volatility input format for Mibian, uses prepared data keys.
    """
    func_name = "calculate_option_greeks_v6" # Consistent versioning
    logger.info(f"[{func_name}] Calculating Greeks for {len(strategy_data)} legs, asset: {asset}, rate: {interest_rate_pct}%")
    greeks_result_list: List[Dict[str, Any]] = []

    # --- Mibian Availability Check ---
    if not MIBIAN_AVAILABLE or mibian is None:
        logger.error(f"[{func_name}] Mibian library unavailable. Cannot calculate Greeks.")
        # Return empty list, endpoint should handle partial results
        return []

    # --- Spot Price Handling ---
    # Prioritize passed-in spot_price if valid
    if spot_price is not None:
        if not isinstance(spot_price, (int, float)) or spot_price <= 0:
             logger.warning(f"[{func_name}] Invalid spot_price passed ({spot_price}). Attempting to fetch.")
             spot_price = None # Invalidate to trigger fetch
        else:
             spot_price = float(spot_price) # Ensure float
             logger.debug(f"[{func_name}] Using provided spot price: {spot_price}")

    # Fetch if not provided or invalid
    if spot_price is None:
        logger.debug(f"[{func_name}] Fetching spot price for {asset}...")
        try:
            # Replace with your primary method (e.g., DB or live API)
            spot_price_info = get_latest_spot_price_from_db(asset)
            if spot_price_info and 'spot_price' in spot_price_info:
                spot_price = _safe_get_float(spot_price_info, 'spot_price')

            # Fallback to cache if DB failed
            if spot_price is None:
                cached_data = get_cached_option(asset)
                if cached_data and "records" in cached_data:
                     spot_price = _safe_get_float(cached_data["records"], "underlyingValue")
            logger.info(f"[{func_name}] Fetched spot price: {spot_price}")

        except Exception as spot_err:
            logger.error(f"[{func_name}] Critical error fetching spot price for {asset}: {spot_err}", exc_info=True)
            return [] # Cannot proceed without spot price

    # Final validation after fetch attempt
    if spot_price is None or not isinstance(spot_price, (int, float)) or spot_price <= 0:
        logger.error(f"[{func_name}] Unable to obtain valid spot price for {asset} (Value: {spot_price}). Cannot calculate Greeks.")
        return []
    spot_price = float(spot_price) # Ensure float for calculations
    logger.debug(f"[{func_name}] Final spot price for calculations: {spot_price}")


    # --- Process Each Leg ---
    processed_count = 0
    skipped_count = 0
    for i, leg_data in enumerate(strategy_data):
        leg_desc = f"Leg {i+1}"
        error_msg = None # Reset error message per leg
        try:
            # --- Validate Input Data Structure ---
            if not isinstance(leg_data, dict):
                logger.warning(f"[{func_name}] Skipping {leg_desc}: input leg data is not a dictionary.");
                skipped_count += 1; continue

            # --- Extract Data (Expecting prepared numeric types) ---
            # Use .get() for safer access, though keys should exist after prepare_data
            strike_price = leg_data.get('strike')
            days_to_expiry = leg_data.get('days_to_expiry')
            implied_vol_decimal = leg_data.get('iv') # Expecting DECIMAL (e.g., 0.15)
            option_type_flag = str(leg_data.get('op_type', '')).lower() # 'c' or 'p'
            transaction_type = str(leg_data.get('tr_type', '')).lower() # 'b' or 's'
            lots = leg_data.get('lot') # Integer
            lot_size = leg_data.get('lot_size') # Integer

            # --- Rigorous Validation Checks (using expected types) ---
            if not isinstance(strike_price, (int, float)) or strike_price <= 0: error_msg=f"Invalid prepared 'strike' ({strike_price})"
            elif not isinstance(days_to_expiry, int) or days_to_expiry < 0: error_msg=f"Invalid prepared 'days_to_expiry' ({days_to_expiry})"
            elif implied_vol_decimal is None or not isinstance(implied_vol_decimal, (int, float)) or implied_vol_decimal < 0: error_msg=f"Invalid prepared 'iv' ({implied_vol_decimal})"
            elif option_type_flag not in ['c', 'p']: error_msg=f"Invalid prepared 'op_type': {option_type_flag}"
            elif transaction_type not in ['b', 's']: error_msg=f"Invalid prepared 'tr_type': {transaction_type}"
            elif not isinstance(lots, int) or lots <= 0: error_msg = f"Invalid prepared 'lot' value: {lots}"
            elif not isinstance(lot_size, int) or lot_size <=0: error_msg = f"Invalid prepared 'lot_size': {lot_size}"

            if error_msg:
                 logger.warning(f"[{func_name}] Skipping {leg_desc} (Prepared Data Validation): {error_msg}. Data: {leg_data}");
                 skipped_count += 1; continue

            # Skip calculation if IV is effectively zero
            if implied_vol_decimal <= 1e-6:
                 logger.info(f"[{func_name}] Skipping {leg_desc} calculation due to zero/near-zero IV ({implied_vol_decimal}). Returning zero Greeks.");
                 # Provide zero greeks instead of skipping entirely? Helps with totals.
                 zero_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
                 input_data_log = { k: leg_data.get(k) for k in ['strike', 'days_to_expiry', 'iv', 'op_type', 'tr_type', 'lots', 'lot_size'] } # Log key inputs
                 input_data_log.update({'spot_used': spot_price, 'rate_used': interest_rate_pct, 'note': 'Zero IV'})
                 leg_result = { 'leg_index': i, 'input_data': input_data_log, 'calculated_greeks_per_share': zero_greeks, 'calculated_greeks_per_lot': zero_greeks }
                 greeks_result_list.append(leg_result)
                 processed_count += 1 # Count as processed (with zero greeks)
                 continue # Move to next leg

            # --- Mibian Calculation (Per Share) ---
            # Use small positive DTE for calculation stability if input is 0
            mibian_dte = float(days_to_expiry) if days_to_expiry > 0 else 0.0001
            mibian_inputs = [spot_price, strike_price, interest_rate_pct, mibian_dte]

            # Convert the DECIMAL IV (e.g., 0.15) to PERCENTAGE POINTS (e.g., 15.0) for Mibian
            volatility_input_for_mibian = implied_vol_decimal * 100.0

            logger.debug(f"[{func_name}] {leg_desc} Mibian Inputs: {[f'{v:.2f}' for v in mibian_inputs]}, Volatility %: {volatility_input_for_mibian:.4f}")

            delta, gamma, theta, vega, rho = np.nan, np.nan, np.nan, np.nan, np.nan # Initialize
            try:
                bs_model = mibian.BS(mibian_inputs, volatility=volatility_input_for_mibian)

                # Extract Greeks based on option type
                if option_type_flag == 'c':
                    delta, theta, rho = bs_model.callDelta, bs_model.callTheta, bs_model.callRho
                else: # 'p'
                    delta, theta, rho = bs_model.putDelta, bs_model.putTheta, bs_model.putRho
                # Gamma and Vega are the same
                gamma = bs_model.gamma
                vega = bs_model.vega

            except ValueError as mibian_val_err:
                 # Catch potential value errors from mibian (e.g., invalid inputs it detects)
                 logger.warning(f"[{func_name}] Mibian ValueError for {leg_desc}: {mibian_val_err}. Skipping leg. Inputs: {mibian_inputs}, Vol: {volatility_input_for_mibian:.2f}");
                 skipped_count += 1; continue
            except Exception as mibian_err:
                 # Catch other unexpected Mibian errors
                 logger.error(f"[{func_name}] Unexpected Mibian calculation error for {leg_desc}: {mibian_err}. Skipping leg. Inputs: {mibian_inputs}, Vol: {volatility_input_for_mibian:.2f}", exc_info=True);
                 skipped_count += 1; continue

            # Store raw calculated Greeks before sign adjustment
            raw_greeks = {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}
            logger.debug(f"[{func_name}] {leg_desc} Raw Greeks calculated: { {k: f'{v:.4f}' if v is not None and np.isfinite(v) else v for k, v in raw_greeks.items()} }")

            # Check if Mibian returned valid numbers
            if any(v is None or not np.isfinite(v) for v in raw_greeks.values()):
                logger.warning(f"[{func_name}] Skipping {leg_desc} due to non-finite raw Greek value from Mibian.");
                skipped_count += 1; continue

            # --- Adjust Sign based on Transaction Type ---
            sign_multiplier = -1.0 if transaction_type == 's' else 1.0
            adjusted_greeks = {k: v * sign_multiplier for k, v in raw_greeks.items()}
            logger.debug(f"[{func_name}] {leg_desc} Adjusted Greeks (Sign: {sign_multiplier}): { {k: f'{v:.4f}' for k, v in adjusted_greeks.items()} }")

            # --- Calculate Per-Share (Rounded) ---
            # Round after all calculations (sign adjustment)
            greeks_per_share = {k: round(v, 4) for k, v in adjusted_greeks.items()}

            # --- Calculate Per-Lot (Rounded) ---
            greeks_per_lot = {k: round(v * lot_size, 4) for k, v in adjusted_greeks.items()}

            # Final check after rounding (highly unlikely to fail here)
            if any(not np.isfinite(v) for v in greeks_per_share.values()) or \
               any(not np.isfinite(v) for v in greeks_per_lot.values()):
                logger.warning(f"[{func_name}] Skipping {leg_desc} due to non-finite value after rounding.");
                skipped_count += 1; continue

            # --- Append results ---
            input_data_log = { k: leg_data.get(k) for k in ['strike', 'days_to_expiry', 'iv', 'op_type', 'tr_type', 'lot', 'lot_size'] }
            input_data_log.update({'spot_used': spot_price, 'rate_used': interest_rate_pct, 'mibian_dte_used': mibian_dte, 'mibian_vol_pct_used': volatility_input_for_mibian})
            leg_result = {
                'leg_index': i,
                'input_data': input_data_log,
                'calculated_greeks_per_share': greeks_per_share,
                'calculated_greeks_per_lot': greeks_per_lot
            }
            greeks_result_list.append(leg_result)
            processed_count += 1
            logger.debug(f"[{func_name}] Successfully processed {leg_desc}.")

        except Exception as e: # Catch any unexpected loop error
            logger.error(f"[{func_name}] UNEXPECTED error processing {leg_desc}: {e}. Skipping.", exc_info=True)
            skipped_count += 1; continue

    logger.info(f"[{func_name}] Finished Greeks calculation. Processed: {processed_count}, Skipped: {skipped_count}.")
    return greeks_result_list


# ===============================================================
# 5. Fetch Stock Data (Rewritten for RapidAPI)
# ===============================================================
async def fetch_stock_data_async(stock_symbol: str, region: str = "IN") -> Optional[Dict[str, Any]]:
    """
    Fetches stock/index data using RapidAPI (apidojo-yahoo-finance-v1) asynchronously.

    - Gets metadata and historical data from '/stock/v3/get-chart'.
    - Calculates moving averages and average volume from historical data.
    - Fetches analyst ratings from '/stock/get-what-analysts-are-saying'.
    - Handles potential API errors and missing data gracefully.
    - Returns a dictionary containing fetched and calculated data.
    """
    cache_key = f"stock_rapidapi_{stock_symbol}_{region}"
    cached = stock_data_cache.get(cache_key)
    if cached:
        logger.debug(f"Cache hit for RapidAPI stock data: {stock_symbol}")
        return cached

    logger.info(f"[{stock_symbol}] Initiating RapidAPI async fetch (Region: {region}).")

    chart_data = None
    analyst_data = None
    # We need daily data for MAs and avg volume, and preferably recent metadata
    # Fetch ~1 year of daily data for calculations. More data can be requested if needed.
    chart_url = f"{RAPIDAPI_BASE_URL}/stock/v3/get-chart"
    chart_params = {
        "symbol": stock_symbol,
        "region": region,
        "interval": "1d",        # Daily interval for calculations
        "range": "1y",           # 1 year range for 50/200 MA stability
        "includePrePost": "false",
        "useYfid": "true",
        "includeAdjustedClose": "true",
        "events": "div,split" # Include dividends/splits for adjusted close
    }

    analyst_url = f"{RAPIDAPI_BASE_URL}/stock/get-what-analysts-are-saying"
    analyst_params = {
        "symbols": stock_symbol,
        "region": region,
        "lang": "en-US" # Or adapt based on region?
    }

    try:
        timeout = aiohttp.ClientTimeout(total=20.0) # 20 second timeout for API calls
        async with aiohttp.ClientSession(headers=RAPIDAPI_HEADERS, timeout=timeout) as session:

            # --- Fetch Chart Data and Analyst Data Concurrently ---
            async def fetch_api(url, params):
                try:
                    async with session.get(url, params=params) as response:
                        logger.debug(f"[{stock_symbol}] Requesting {response.url}")
                        if response.status == 200:
                            data = await response.json()
                            logger.debug(f"[{stock_symbol}] Received {response.status} from {url}")
                            # Basic validation: Check for expected structure
                            if url == chart_url and not data.get('chart', {}).get('result'):
                                logger.error(f"[{stock_symbol}] Chart API response missing 'chart.result'. Data: {str(data)[:200]}")
                                return None # Indicate fetch failure
                            if url == analyst_url and 'result' not in data:
                                 logger.warning(f"[{stock_symbol}] Analyst API response missing 'result'. Data: {str(data)[:200]}")
                                 # Non-critical, proceed without analyst data
                                 return {} # Return empty dict for analyst data on failure/no data
                            return data
                        elif response.status == 404:
                             logger.error(f"[{stock_symbol}] API returned 404 Not Found for {url}. Symbol likely invalid for region '{region}' or API issue.")
                             return None # Indicate critical failure for chart, non-critical for analyst
                        else:
                            logger.error(f"[{stock_symbol}] API Error {response.status} from {url}: {await response.text()}")
                            return None # Indicate fetch failure
                except asyncio.TimeoutError:
                     logger.error(f"[{stock_symbol}] Timeout fetching data from {url}")
                     return None
                except aiohttp.ClientError as http_err:
                    logger.error(f"[{stock_symbol}] HTTP Client Error fetching data from {url}: {http_err}", exc_info=False)
                    return None
                except json.JSONDecodeError as json_err:
                     logger.error(f"[{stock_symbol}] JSON Decode Error parsing response from {url}: {json_err}", exc_info=False)
                     return None
                except Exception as e:
                    logger.error(f"[{stock_symbol}] Unexpected error fetching data from {url}: {e}", exc_info=True)
                    return None

            # Run fetches concurrently
            api_tasks = [
                fetch_api(chart_url, chart_params),
                fetch_api(analyst_url, analyst_params)
            ]
            results = await asyncio.gather(*api_tasks)
            chart_response, analyst_response = results

            # --- Process Chart Data (CRITICAL) ---
            if not chart_response or not chart_response.get('chart', {}).get('result'):
                logger.error(f"[{stock_symbol}] CRITICAL - Failed to fetch or parse valid chart data from RapidAPI. Returning None.")
                return None # Critical failure if chart data is missing/invalid

            # Extract metadata and indicators safely
            try:
                meta = chart_response['chart']['result'][0].get('meta', {})
                indicators = chart_response['chart']['result'][0].get('indicators', {}).get('quote', [{}])[0]
                timestamps = chart_response['chart']['result'][0].get('timestamp', [])
                closes = indicators.get('close', [])
                volumes = indicators.get('volume', [])
                opens = indicators.get('open', [])
                highs = indicators.get('high', [])
                lows = indicators.get('low', [])
                # Adjusted close might be useful but let's stick to 'close' for MAs for now
                # adj_closes = chart_response['chart']['result'][0].get('indicators', {}).get('adjclose', [{}])[0].get('adjclose', [])

                if not meta or not timestamps or not closes:
                     logger.error(f"[{stock_symbol}] CRITICAL - Essential chart data (meta, timestamps, closes) missing in response. Returning None.")
                     return None

                logger.debug(f"[{stock_symbol}] Chart meta keys: {list(meta.keys())}")
                logger.debug(f"[{stock_symbol}] Fetched {len(timestamps)} daily data points.")

            except (KeyError, IndexError, TypeError) as e:
                 logger.error(f"[{stock_symbol}] CRITICAL - Error accessing expected keys in chart response structure: {e}. Response snippet: {str(chart_response)[:500]}", exc_info=False)
                 return None

            # --- Determine Price (CRITICAL) ---
            cp = meta.get("regularMarketPrice")
            price_source = "meta.regularMarketPrice"
            if cp is None or not isinstance(cp, (int, float)):
                # Fallback to last closing price from history if meta price is bad
                if closes and isinstance(closes[-1], (int, float)):
                    cp = closes[-1]
                    price_source = "history.last_close"
                    logger.warning(f"[{stock_symbol}] Used last close price from history ({cp}) as regularMarketPrice was invalid/missing.")
                else:
                    logger.error(f"[{stock_symbol}] CRITICAL - Could not determine valid finite price from meta or history. Last close: {closes[-1] if closes else 'N/A'}. Returning None.")
                    return None # CRITICAL FAILURE

            logger.info(f"[{stock_symbol}] Determined price: {cp} (Source: {price_source})")


            # --- Prepare DataFrame for Calculations ---
            # Ensure all lists have the same length corresponding to timestamps
            min_len = len(timestamps)
            if len(closes) != min_len or len(volumes) != min_len or len(opens) != min_len or len(highs) != min_len or len(lows) != min_len:
                logger.warning(f"[{stock_symbol}] Mismatch in lengths of historical data arrays. Truncating to shortest length ({min_len}).")
                closes = closes[:min_len]
                volumes = volumes[:min_len]
                opens = opens[:min_len]
                highs = highs[:min_len]
                lows = lows[:min_len]

            # Filter out potential nulls/None before creating DataFrame
            # Find indices where close is valid (not None and finite)
            valid_indices = [i for i, price in enumerate(closes) if price is not None and np.isfinite(price)]

            if not valid_indices:
                 logger.error(f"[{stock_symbol}] CRITICAL - No valid closing prices found in historical data. Cannot proceed.")
                 return None

            # Create DataFrame using only valid data points, ensuring alignment
            try:
                 df = pd.DataFrame({
                     'Timestamp': pd.to_datetime([timestamps[i] for i in valid_indices], unit='s'),
                     'Open': [opens[i] for i in valid_indices],
                     'High': [highs[i] for i in valid_indices],
                     'Low': [lows[i] for i in valid_indices],
                     'Close': [closes[i] for i in valid_indices],
                     # Ensure volume is integer, handle None -> 0
                     'Volume': [int(volumes[i]) if volumes[i] is not None and np.isfinite(volumes[i]) else 0 for i in valid_indices]
                 }).set_index('Timestamp')
                 logger.debug(f"[{stock_symbol}] Created DataFrame with {len(df)} rows for calculations.")
            except Exception as df_err:
                logger.error(f"[{stock_symbol}] CRITICAL - Failed to create DataFrame from historical data: {df_err}", exc_info=True)
                return None


            # --- Calculate Moving Averages & Average Volume ---
            ma50 = None
            ma200 = None
            avg_vol = None

            if len(df) >= 50:
                try:
                    ma50 = df["Close"].rolling(window=50).mean().iloc[-1]
                    # Calculate 50-day average volume as well
                    avg_vol = df["Volume"].rolling(window=50).mean().iloc[-1]
                    if not np.isfinite(ma50): ma50 = None
                    if not np.isfinite(avg_vol): avg_vol = None
                except Exception as ma_ex: logger.warning(f"[{stock_symbol}] Error calculating MA50/AvgVol50: {ma_ex}")
            else: logger.warning(f"[{stock_symbol}] Not enough data for MA50/AvgVol50 (need 50, got {len(df)}).")

            if len(df) >= 200:
                try:
                    ma200 = df["Close"].rolling(window=200).mean().iloc[-1]
                    # Optionally update avg_vol to 200 day average if available
                    # avg_vol = df["Volume"].rolling(window=200).mean().iloc[-1] # Uncomment if 200d avg vol is preferred
                    if not np.isfinite(ma200): ma200 = None
                except Exception as ma_ex: logger.warning(f"[{stock_symbol}] Error calculating MA200: {ma_ex}")
            else: logger.warning(f"[{stock_symbol}] Not enough data for MA200 (need 200, got {len(df)}).")

            logger.debug(f"[{stock_symbol}] MA50 (calc): {ma50}, MA200 (calc): {ma200}, AvgVol50 (calc): {avg_vol}")

            # --- Get Other Fields from Meta or History ---
            name = meta.get("shortName", meta.get("longName", stock_symbol))
            qtype = meta.get("instrumentType") # e.g., "EQUITY", "INDEX"
            exch = meta.get("exchangeName") # e.g., "NSI" for NSE
            # Use meta volume if available and seems valid, otherwise use last history volume
            current_vol = meta.get("regularMarketVolume")
            vol_source = "meta.regularMarketVolume"
            if current_vol is None or not isinstance(current_vol, (int, float)) or current_vol < 0 :
                 if not df.empty and 'Volume' in df.columns:
                    current_vol = df['Volume'].iloc[-1]
                    vol_source = "history.last_volume"
                    logger.debug(f"[{stock_symbol}] Used last volume from history ({current_vol}) as meta volume was invalid/missing.")
                 else:
                    current_vol = None
                    vol_source = "None"
                    logger.warning(f"[{stock_symbol}] Could not determine current volume from meta or history.")

            # Fundamental data is generally NOT in the chart endpoint
            mc = meta.get("marketCap") # Check if it exists
            pe = None # meta.get("trailingPE") # Typically not here
            eps = None # meta.get("trailingEps") # Typically not here
            sec = None # meta.get("sector") # Typically not here
            ind = None # meta.get("industry") # Typically not here

            # Previous Close: Use 2nd last close from daily history
            prev_close = df["Close"].iloc[-2] if len(df) >= 2 else None

            # Today's Open: Use meta if available, otherwise last open from history
            day_open = meta.get("regularMarketOpen")
            if day_open is None and not df.empty:
                day_open = df['Open'].iloc[-1]

            day_high = meta.get("regularMarketDayHigh")
            day_low = meta.get("regularMarketDayLow")
            week_52_high = meta.get("fiftyTwoWeekHigh")
            week_52_low = meta.get("fiftyTwoWeekLow")

            logger.debug(f"[{stock_symbol}] Meta Fields - Name: {name}, MC: {mc}, Type: {qtype}, Exch: {exch}")
            logger.debug(f"[{stock_symbol}] Volume: {current_vol} (Source: {vol_source}), AvgVol: {avg_vol}")
            logger.debug(f"[{stock_symbol}] Day: O:{day_open} H:{day_high} L:{day_low} PrevC:{prev_close}")
            logger.debug(f"[{stock_symbol}] 52Wk: H:{week_52_high} L:{week_52_low}")

            # --- Process Analyst Data (Non-critical) ---
            analyst_summary = []
            if analyst_response and analyst_response.get('result'):
                try:
                    hits = analyst_response['result'][0].get('hits', [])
                    logger.debug(f"[{stock_symbol}] Found {len(hits)} analyst reports.")
                    for report in hits[:2]: # Get latest 2 reports
                         analyst_summary.append({
                             "provider": report.get("provider"),
                             "rating": report.get("investment_rating"),
                             "target_price": report.get("target_price"),
                             "action": report.get("investment_rating_status"), # e.g., Maintained, Increased
                             "date": pd.to_datetime(report.get("report_date"), unit='ms', errors='coerce') if report.get("report_date") else None,
                             "abstract": report.get("abstract", "")[:150] + "..." # Short abstract
                         })
                except Exception as an_err:
                    logger.warning(f"[{stock_symbol}] Error processing analyst data: {an_err}. Data: {str(analyst_response)[:200]}", exc_info=False)
                    analyst_summary = [] # Clear summary on error
            else:
                 logger.debug(f"[{stock_symbol}] No analyst data found or failed to fetch.")


            # --- Construct Final Data Dictionary ---
            data = {
                "symbol": stock_symbol,
                "name": name,
                "quote_type": qtype,
                "exchange": exch,
                "current_price": float(cp) if cp is not None and np.isfinite(cp) else None,
                "volume": int(current_vol) if current_vol is not None and np.isfinite(current_vol) else None,
                "average_volume": int(avg_vol) if avg_vol is not None and np.isfinite(avg_vol) else None, # Using 50d avg vol
                "moving_avg_50": float(ma50) if ma50 is not None and np.isfinite(ma50) else None,
                "moving_avg_200": float(ma200) if ma200 is not None and np.isfinite(ma200) else None,
                "market_cap": int(mc) if mc is not None and np.isfinite(mc) else None, # Will likely be None often
                "pe_ratio": float(pe) if pe is not None and np.isfinite(pe) else None, # None
                "eps": float(eps) if eps is not None and np.isfinite(eps) else None,   # None
                "sector": sec, # None
                "industry": ind, # None
                "previous_close": float(prev_close) if prev_close is not None and np.isfinite(prev_close) else None,
                "open": float(day_open) if day_open is not None and np.isfinite(day_open) else None,
                "day_high": float(day_high) if day_high is not None and np.isfinite(day_high) else None,
                "day_low": float(day_low) if day_low is not None and np.isfinite(day_low) else None,
                "fifty_two_week_high": float(week_52_high) if week_52_high is not None and np.isfinite(week_52_high) else None,
                "fifty_two_week_low": float(week_52_low) if week_52_low is not None and np.isfinite(week_52_low) else None,
                "analyst_ratings": analyst_summary # Add the analyst summary list
            }

            stock_data_cache[cache_key] = data
            logger.info(f"[{stock_symbol}] RapidAPI fetch successful. Caching and returning data.")
            return data

    except aiohttp.ClientError as e:
        logger.error(f"[{stock_symbol}] Network or Client Error during RapidAPI fetch setup: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"[{stock_symbol}] UNEXPECTED error during overall RapidAPI process: {e}", exc_info=True)
        return None

# ===============================================================
# 6. Fetch News (Robust Scraping - Keep existing)
# ===============================================================
# (Keep the fetch_latest_news_async function exactly as it was in the initial prompt)
async def fetch_latest_news_async(asset: str) -> List[Dict[str, str]]:
    """Fetches latest news using feedparser from Google News RSS."""
    cache_key = f"news_{asset.upper()}"
    cached = news_cache.get(cache_key)
    if cached:
        logger.debug(f"Cache hit for news: {asset}")
        return cached

    logger.info(f"Fetching news for: {asset} using feedparser/Google News")
    news_list = []
    max_news = 5 # Fetch a few headlines

    try:
        # Construct Google News RSS URL (India, English)
        search_term = f"{asset} stock market India" # Be more specific
        encoded_search_term = quote_plus(search_term)
        gnews_url = f"https://news.google.com/rss/search?q={encoded_search_term}&hl=en-IN&gl=IN&ceid=IN:en"

        loop = asyncio.get_running_loop()
        # feedparser is synchronous, run in executor
        logger.debug(f"Parsing Google News RSS feed: {gnews_url}")
        try:
            feed_data = await asyncio.wait_for(
                loop.run_in_executor(None, feedparser.parse, gnews_url),
                timeout=15.0 # Example: 15 second timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching/parsing news feed for {asset} from {gnews_url}")
            return [{"headline": f"Timeout fetching news for {asset}.", "summary": "Feedparser took too long.", "link": "#"}]
        except Exception as parse_exec_err:
             logger.error(f"Error running feedparser in executor for {asset}: {parse_exec_err}", exc_info=True)
             return [{"headline": f"Error processing news feed task for {asset}.", "summary": str(parse_exec_err), "link": "#"}]

        if feed_data.bozo:
             bozo_exception = feed_data.get('bozo_exception', 'Unknown parsing error')
             logger.warning(f"Feedparser marked feed as bozo for {asset}. URL: {gnews_url}. Exception: {bozo_exception}")
             if isinstance(bozo_exception, (feedparser.exceptions.HTTPError, ConnectionRefusedError)):
                 return [{"headline": f"Error fetching news feed for {asset}.", "summary": f"HTTP/Connection Error: {bozo_exception}", "link": "#"}]
             else:
                 return [{"headline": f"Error parsing news feed for {asset}.", "summary": f"Parsing issue: {bozo_exception}", "link": "#"}]

        if not feed_data.entries:
            logger.warning(f"No news entries found in Google News RSS feed for query: {search_term}")
            return [{"headline": f"No recent news found for {asset}.", "summary": f"Query: '{search_term}'", "link": "#"}]

        for entry in feed_data.entries[:max_news]:
            headline = entry.get('title', 'No Title')
            link = entry.get('link', '#')
            summary = entry.get('summary', entry.get('description', 'No summary available.'))
            try:
                summary_soup = BeautifulSoup(summary, "html.parser")
                cleaned_summary = summary_soup.get_text(strip=True)
            except Exception as bs_err:
                 logger.warning(f"Could not clean summary HTML for '{headline[:50]}...': {bs_err}")
                 cleaned_summary = summary

            news_list.append({
                "headline": headline,
                "summary": cleaned_summary,
                "link": link
            })

        if not news_list:
             logger.warning(f"Filtered out all news entries for {asset}")
             return [{"headline": f"No relevant news found for {asset} after filtering.", "summary": "", "link": "#"}]

        news_cache[cache_key] = news_list
        logger.info(f"Successfully parsed {len(news_list)} news items for {asset} via feedparser.")
        return news_list

    except Exception as e:
        logger.error(f"Unexpected error in fetch_latest_news_async for {asset}: {e}", exc_info=True)
        return [{"headline": f"Unexpected error fetching news for {asset}.", "summary": str(e), "link": "#"}]


# ===============================================================
# 7. Build Analysis Prompt (May need slight adjustments)
# ===============================================================
# (Keep the fmt function as it was)
def fmt(v, p="₹", s="", pr=2, na="N/A"):
    """Formats numbers nicely, handling None, non-finite, and Indian Crores/Lakhs."""
    if v is None or v == 'N/A' or (isinstance(v, (float, np.number)) and not np.isfinite(v)): return na
    if isinstance(v,(int,float, np.number)):
        v_float = float(v) # Convert numpy types first
        try:
            if abs(v_float) >= 1e7: return f"{p}{v_float/1e7:.{pr}f} Cr{s}"
            if abs(v_float) >= 1e5: return f"{p}{v_float/1e5:.{pr}f} L{s}"
            return f"{p}{v_float:,.{pr}f}{s}"
        except Exception: return str(v) # Fallback
    return str(v)

# --- Updated Prompt Building Function (to potentially include analyst ratings) ---
def build_stock_analysis_prompt_for_options(
    stock_symbol_display: str,
    stock_data: Optional[Dict[str, Any]],
    latest_news: Optional[List[Dict[str, str]]]
) -> str:
    """
    Generates a structured prompt for LLM analysis focused on Option Trader insights.
    Requires the LLM to start with a specific bias heading (Bullish/Bearish/Neutral Outlook)
    and state the net bias explicitly. Generalized for stocks and indices.
    """
    func_name = "build_stock_analysis_prompt_for_options_v2" # Version tracking

    # --- Handle potentially missing input data (Keep as is) ---
    if not stock_data:
        logger.error(f"[{func_name}] Critical error: stock_data is None for {stock_symbol_display}. Cannot build prompt.")
        return f"Analysis for {stock_symbol_display} failed: Essential stock data is missing."
    if latest_news is None:
         logger.warning(f"[{func_name}] Warning: latest_news is None for {stock_symbol_display}. Proceeding without news.")
         latest_news = [{"headline": "News data not available.", "summary": "", "link": "#"}]

    logger.debug(f"[{func_name}] Building structured options-focused prompt for {stock_symbol_display}")

    # --- Extract Data Safely (Keep as is) ---
    name = stock_data.get('name', stock_symbol_display)
    price = stock_data.get('current_price')
    prev_close = stock_data.get('previous_close')
    day_open = stock_data.get('open')
    day_high = stock_data.get('day_high')
    day_low = stock_data.get('day_low')
    ma50 = stock_data.get('moving_avg_50')
    ma200 = stock_data.get('moving_avg_200')
    volume = stock_data.get('volume')
    avg_volume = stock_data.get('average_volume')
    week_52_high = stock_data.get('fifty_two_week_high')
    week_52_low = stock_data.get('fifty_two_week_low')
    market_cap = stock_data.get('market_cap')
    pe_ratio = stock_data.get('pe_ratio')
    eps = stock_data.get('eps')
    sector = stock_data.get('sector')
    industry = stock_data.get('industry')
    qtype = stock_data.get('quote_type', 'Unknown')
    analyst_ratings = stock_data.get('analyst_ratings', [])

    display_title = f"{stock_symbol_display}" + (f" ({name})" if name and name != stock_symbol_display else "")
    is_index = qtype == 'INDEX'

    # --- Prepare Context Strings (Keep as is) ---
    # Technical Context Calculation (Trend, Support/Resistance)
    trend = "N/A"; support_str = "N/A"; resistance_str = "N/A"
    ma_available = ma50 is not None and ma200 is not None and price is not None
    if ma_available:
        support_levels = sorted([lvl for lvl in [ma50, ma200] if lvl is not None and lvl < price], reverse=True)
        resistance_levels = sorted([lvl for lvl in [ma50, ma200] if lvl is not None and lvl >= price])
        support_str = " / ".join([fmt(lvl) for lvl in support_levels]) if support_levels else "Below Key MAs"
        resistance_str = " / ".join([fmt(lvl) for lvl in resistance_levels]) if resistance_levels else "Above Key MAs"
        # Trend description
        if price > ma50 > ma200: trend = "Strong Uptrend (Price > 50MA > 200MA)"
        elif price < ma50 < ma200: trend = "Strong Downtrend (Price < 50MA < 200MA)"
        elif ma50 > price > ma200 : trend = "Sideways/Testing Support (Below 50MA, Above 200MA)"
        elif ma200 > price > ma50 : trend = "Sideways/Testing Resistance (Below 200MA, Above 50MA) - Caution"
        elif price > ma50 and price > ma200: trend = "Uptrend (Price > Both MAs, 50MA potentially below 200MA)"
        elif price < ma50 and price < ma200: trend = "Downtrend (Price < Both MAs, 50MA potentially above 200MA)"
        else: trend = "Indeterminate (MAs crossing or very close)"
    elif price and ma50:
        support_str = fmt(ma50) if price > ma50 else "N/A (Below 50MA)"
        resistance_str = fmt(ma50) if price <= ma50 else "N/A (Above 50MA)"
        trend = "Above 50MA" if price > ma50 else "Below 50MA"
        resistance_str += " (200MA N/A)"; support_str += " (200MA N/A)"
    else: trend = "Trend Unknown (MA data insufficient)"
    # Volume context
    vol_str = fmt(volume, p='', pr=0, na='N/A'); avg_vol_str = fmt(avg_volume, p='', pr=0, na='N/A')
    vol_comment = f"Volume: {vol_str}."
    if avg_volume is not None and volume is not None and avg_volume > 0:
        vol_ratio = volume / avg_volume
        if vol_ratio > 1.5: vol_comment += f" Significantly ABOVE {fmt(avg_volume, p='', pr=0, s=' (Avg)') }."
        elif vol_ratio < 0.7: vol_comment += f" Significantly BELOW {fmt(avg_volume, p='', pr=0, s=' (Avg)') }."
        else: vol_comment += f" Near {fmt(avg_volume, p='', pr=0, s=' (Avg)') }."
    elif avg_volume is not None: vol_comment += f" Average Volume: {avg_vol_str}."
    else: vol_comment += f" Average Volume N/A."
    # Assemble Technical Context
    tech_context = ( f"Price: {fmt(price)} (Open: {fmt(day_open)}, Day Range: {fmt(day_low)} - {fmt(day_high)}, Prev Close: {fmt(prev_close)}). " f"52wk Range: {fmt(week_52_low)} - {fmt(week_52_high)}. " f"MAs: 50D={fmt(ma50)}, 200D={fmt(ma200)}. " f"{vol_comment} " f"Trend Context: {trend}. " f"Key Levels (from MAs): Support near {support_str}, Resistance near {resistance_str}." )

    # Fundamental Context String
    fund_context = ""; pe_comparison_note = "Note: P/E data likely unavailable."
    if is_index: fund_context = "N/A (Index)"
    else: fund_context = ( f"Market Cap: {fmt(market_cap, p='', na='N/A')}, " f"P/E Ratio: {fmt(pe_ratio, p='', s='x', na='N/A')}, " f"EPS: {fmt(eps, p='', na='N/A')}, " f"Sector: {fmt(sector, p='', na='N/A')}, " f"Industry: {fmt(industry, p='', na='N/A')}" )
    if pe_ratio is not None: pe_comparison_note = f"Note: P/E ({fmt(pe_ratio, p='', s='x')}) requires peer/historical comparison."

    # News Context String
    news_formatted = []; is_news_error_or_empty = not latest_news or any(err in latest_news[0].get("headline","").lower() for err in ["error", "no recent", "no relevant", "timeout", "not available"])
    if not is_news_error_or_empty: news_context = "\n".join([f'- "{n.get("headline","N/A")}" ({n.get("link","#")}): {n.get("summary","N/A").replace("`","").replace("{","(").replace("}",")")}' for n in latest_news[:3]])
    else: news_context = f"- {latest_news[0].get('headline', 'No recent news summaries available.')}" if latest_news else "- No recent news summaries available."

    # Analyst Ratings Context
    analyst_context = "No recent analyst ratings found or fetch failed."
    if analyst_ratings:
         analyst_formatted = [f"- {r.get('provider','N/A')}: {r.get('rating','N/A')}" + (f" (Target: {fmt(r.get('target_price'))})" if r.get('target_price') is not None else "") + (f" [{r.get('action')}]" if r.get('action') else "") + (f" ({r['date'].strftime('%Y-%m-%d')})" if r.get('date') else "") for r in analyst_ratings]
         analyst_context = "\n".join(analyst_formatted)

# ===============================================================
# 7. Build Analysis Prompt (Updated w/ OptionsPlaybook Link)
# ===============================================================
    # --- Construct the Structured Prompt (with new instructions) ---
def build_stock_analysis_prompt_for_options( # <<< FUNCTION NAME RETAINED
    stock_symbol_display: str,
    stock_data: Optional[Dict[str, Any]],
    latest_news: Optional[List[Dict[str, str]]]
) -> str:
    """
    Generates a structured prompt for LLM analysis focused on Option Trader insights.
    Requires the LLM to start with a specific bias heading, state the net bias,
    suggest relevant strategy types, and link to OptionsPlaybook.com categories.
    """
    func_name = "build_stock_analysis_prompt_for_options" # Using original name for logging

    # --- Handle potentially missing input data (Keep as is) ---
    if not stock_data:
        logger.error(f"[{func_name}] Critical error: stock_data is None for {stock_symbol_display}. Cannot build prompt.")
        return f"Analysis for {stock_symbol_display} failed: Essential stock data is missing."
    if latest_news is None:
         logger.warning(f"[{func_name}] Warning: latest_news is None for {stock_symbol_display}. Proceeding without news.")
         latest_news = [{"headline": "News data not available.", "summary": "", "link": "#"}]

    logger.debug(f"[{func_name}] Building structured options-focused prompt for {stock_symbol_display}")

    # --- Extract Data Safely (Keep as is) ---
    # ... (keep all data extraction logic: name, price, ma50, etc.) ...
    name = stock_data.get('name', stock_symbol_display)
    price = stock_data.get('current_price')
    prev_close = stock_data.get('previous_close')
    day_open = stock_data.get('open')
    day_high = stock_data.get('day_high')
    day_low = stock_data.get('day_low')
    ma50 = stock_data.get('moving_avg_50')
    ma200 = stock_data.get('moving_avg_200')
    volume = stock_data.get('volume')
    avg_volume = stock_data.get('average_volume')
    week_52_high = stock_data.get('fifty_two_week_high')
    week_52_low = stock_data.get('fifty_two_week_low')
    market_cap = stock_data.get('market_cap')
    pe_ratio = stock_data.get('pe_ratio')
    eps = stock_data.get('eps')
    sector = stock_data.get('sector')
    industry = stock_data.get('industry')
    qtype = stock_data.get('quote_type', 'Unknown')
    analyst_ratings = stock_data.get('analyst_ratings', [])

    display_title = f"{stock_symbol_display}" + (f" ({name})" if name and name != stock_symbol_display else "")
    is_index = qtype == 'INDEX'


    # --- Prepare Context Strings (Keep as is) ---
    # ... (keep tech_context, fund_context, news_context, analyst_context calculations) ...
    # Technical Context Calculation
    trend = "N/A"; support_str = "N/A"; resistance_str = "N/A"
    ma_available = ma50 is not None and ma200 is not None and price is not None
    if ma_available: support_levels = sorted([lvl for lvl in [ma50, ma200] if lvl is not None and lvl < price], reverse=True); resistance_levels = sorted([lvl for lvl in [ma50, ma200] if lvl is not None and lvl >= price]); support_str = " / ".join([fmt(lvl) for lvl in support_levels]) if support_levels else "Below Key MAs"; resistance_str = " / ".join([fmt(lvl) for lvl in resistance_levels]) if resistance_levels else "Above Key MAs"; # Trend description
    if ma_available:
        if price > ma50 > ma200: trend = "Strong Uptrend (Price > 50MA > 200MA)"
        elif price < ma50 < ma200: trend = "Strong Downtrend (Price < 50MA < 200MA)"
        elif ma50 > price > ma200 : trend = "Sideways/Testing Support (Below 50MA, Above 200MA)"
        elif ma200 > price > ma50 : trend = "Sideways/Testing Resistance (Below 200MA, Above 50MA) - Caution"
        elif price > ma50 and price > ma200: trend = "Uptrend (Price > Both MAs)"
        elif price < ma50 and price < ma200: trend = "Downtrend (Price < Both MAs)"
        else: trend = "Indeterminate (MAs crossing)"
    elif price and ma50: trend = "Above 50MA" if price > ma50 else "Below 50MA"; resistance_str = fmt(ma50) if price <= ma50 else "N/A (Above 50MA)"; support_str = fmt(ma50) if price > ma50 else "N/A (Below 50MA)"; resistance_str += " (200MA N/A)"; support_str += " (200MA N/A)"
    else: trend = "Trend Unknown (MA data insufficient)"
    # Volume context
    vol_str = fmt(volume, p='', pr=0, na='N/A'); avg_vol_str = fmt(avg_volume, p='', pr=0, na='N/A'); vol_comment = f"Volume: {vol_str}."
    if avg_volume is not None and volume is not None and avg_volume > 0: vol_ratio = volume / avg_volume; vol_comment += f" vs Avg: {avg_vol_str} ({'Above' if vol_ratio > 1.2 else 'Below' if vol_ratio < 0.8 else 'Near'})";
    elif avg_volume is not None: vol_comment += f" Average Volume: {avg_vol_str}."
    else: vol_comment += f" Average Volume N/A."
    # Assemble Technical Context
    tech_context = ( f"Price: {fmt(price)} (Open: {fmt(day_open)}, Day Range: {fmt(day_low)} - {fmt(day_high)}, Prev Close: {fmt(prev_close)}). " f"52wk Range: {fmt(week_52_low)} - {fmt(week_52_high)}. " f"MAs: 50D={fmt(ma50)}, 200D={fmt(ma200)}. " f"{vol_comment} " f"Trend Context: {trend}. " f"Key Levels (from MAs): Support near {support_str}, Resistance near {resistance_str}." )
    # Fundamental Context String
    fund_context = ""; pe_comparison_note = "Note: P/E data likely unavailable."
    if is_index: fund_context = "N/A (Index)"
    else: fund_context = ( f"Market Cap: {fmt(market_cap, p='', na='N/A')}, P/E: {fmt(pe_ratio, p='', s='x', na='N/A')}, EPS: {fmt(eps, p='', na='N/A')}, Sector: {fmt(sector, p='', na='N/A')}, Industry: {fmt(industry, p='', na='N/A')}" )
    if pe_ratio is not None: pe_comparison_note = f"Note: P/E ({fmt(pe_ratio, p='', s='x')}) requires context."
    # News Context String
    news_formatted = []; is_news_error_or_empty = not latest_news or any(err in latest_news[0].get("headline","").lower() for err in ["error", "no recent", "no relevant", "timeout", "not available"])
    if not is_news_error_or_empty: news_context = "\n".join([f'- "{n.get("headline","N/A")}" ({n.get("link","#")}): {n.get("summary","N/A").replace("`","").replace("{","(").replace("}",")")}' for n in latest_news[:3]])
    else: news_context = f"- {latest_news[0].get('headline', 'No recent news summaries available.')}" if latest_news else "- No recent news summaries available."
    # Analyst Ratings Context
    analyst_context = "No recent analyst ratings found or fetch failed."
    if analyst_ratings: analyst_formatted = [f"- {r.get('provider','N/A')}: {r.get('rating','N/A')}" + (f" (Target: {fmt(r.get('target_price'))})" if r.get('target_price') is not None else "") + (f" [{r.get('action')}]" if r.get('action') else "") + (f" ({r['date'].strftime('%Y-%m-%d')})" if r.get('date') else "") for r in analyst_ratings]; analyst_context = "\n".join(analyst_formatted)

    # --- Construct the Structured Prompt (with OptionsPlaybook link instruction) ---
    prompt = f"""
**IMPORTANT:** Start your response with ONLY ONE of the following Level 2 Markdown headings, reflecting your overall assessment based on the provided data: `## Bullish Outlook`, `## Bearish Outlook`, or `## Neutral Outlook`. Note, you are option trader and want the following varible in easy languange and comprehensive. Provide output in structured manner, that is easy to read.
Immediately after the heading, state the **Net Bias** on a new line, like this: `Net Bias: [Your Bias (e.g., Bullish, Moderately Bearish, Neutral, Range-bound)]`.

Then, provide the detailed analysis for **{display_title}** from the perspective of an **Options Trader**. Use the subsequent headings and bullet points as outlined below. Focus on potential direction, volatility hints, and key levels. Explicitly acknowledge missing options-specific data (like Implied Volatility) and potential gaps in fundamental data.

**Crucially, in section 5 ("Options Strategy Angle"), suggest 1-2 general strategy types fitting your derived Net Bias and include a relevant general link to OptionsPlaybook.com.**

**Provided Data Snapshots (Source: RapidAPI / Google News):**

*   **Technical Context:** {tech_context}
*   **Fundamental Context:** {fund_context} (Note: May have data gaps)
*   **Analyst Ratings Summary (Max 2 Recent):**
{analyst_context}
*   **Recent News Snippets (Max 3):**
{news_context}

---
**Detailed Analysis Request (Options Trader Focus):**

**(Remember to place the `## [Bullish/Bearish/Neutral] Outlook` heading and `Net Bias:` statement BEFORE this section)**

1.  **Overall Market Posture & Bias Rationale:**
    *   *Explain* the reasoning behind the **Net Bias** you stated above, synthesizing technicals (trend, MAs, price action), volume, news, and analyst ratings.
    *   Comment on potential **Volatility Hints** (News? Extremes? Volume?). Acknowledge IV data is missing.

2.  **Key Technical Levels & Observations:**
    *   **Support & Resistance:** Reiterate key levels based *only* on provided MAs, 52-week range, and recent day's range.
    *   **Trend Strength & Volume:** Comment on trend conviction. Is volume confirming or diverging? Is price extended or consolidating?
    *   **Significant Price Zones:** Note if near 52-week high/low relevant for potential breakouts/downs.

3.  **News & Analyst Sentiment Impact:**
    *   **Sentiment:** Elaborate on combined sentiment (Positive, Negative, Neutral, Mixed, N/A).
    *   **Volatility/Catalyst Potential:** How might news/analyst action influence volatility or act as catalysts?

4.  **Options Strategy Angle & Considerations (Based on Net Bias):**
    *   **CRITICAL CAVEAT:** State clearly: "**Implied Volatility (IV) data is MISSING.** IV is crucial for option pricing and strategy selection. The following are *general conceptual ideas only* based on the derived directional bias and ignore the current IV environment."
    *   **General Strategy Types & Link:**
        *   *Based on your derived Net Bias*, suggest **one or two appropriate conceptual strategy types** (e.g., "Long Call / Bull Call Spread", "Iron Condor", "Long Put / Bear Put Spread" and more).
        *   **Then, provide ONE relevant general link from the list below corresponding to your Net Bias:**
            *   If Bullish: Learn more about bullish strategies: https://www.optionsplaybook.com/option-strategies/#bullish-strategies
            *   If Bearish: Learn more about bearish strategies: https://www.optionsplaybook.com/option-strategies/#bearish-strategies
            *   If Neutral/Range-Bound: Learn more about neutral strategies: https://www.optionsplaybook.com/option-strategies/#neutral-strategies
            *   (Do not link if bias is very uncertain).
    *   **Key Factors Ignored:** Remind the user that this analysis **ignores IV, risk tolerance, specific expiry/strike selection, option liquidity, and theta implications.**

5.  **Identified Risks & Data Limitations:**
    *   Summarize risks suggested *only* by the provided data (e.g., Price vs MAs, negative news/ratings, divergence, 52wk proximity).
    *   Explicitly list major **Data Limitations**: **No Implied Volatility (IV), No Historical Volatility (HV), No Option Chain Data (Greeks, OI, Volume), No Event Calendar, Potential gaps in Fundamental data.**

---
**Disclaimer:** This analysis is auto-generated based on limited data. It is NOT financial advice. Data accuracy depends on sources. Crucial options data (IV) and potentially key fundamental data are missing. Always conduct independent research, understand risks, consider your risk tolerance, and consult a qualified financial advisor before trading. Market conditions are dynamic.
"""
    logger.debug(f"[{func_name}] Generated options-focused prompt for {stock_symbol_display} (v3 - OptionsPlaybook link)")
    return prompt

        

# ===============================================================
#  8. Greeks analysis by LLM
# ===============================================================
async def fetch_greeks_analysis(
    asset_symbol: str,
    portfolio_greeks: Dict[str, Optional[float]] # Expects TOTALS: delta, gamma, theta, vega, rho
) -> Optional[str]:
    """
    Generates an LLM analysis focused *solely* on the provided TOTAL portfolio Greeks.

    Args:
        asset_symbol: The underlying asset symbol (for context).
        portfolio_greeks: A dictionary containing the TOTAL portfolio Greeks,
                          e.g., {"delta": 10.5, "gamma": 2.1, "theta": -5.0, "vega": 50.2, "rho": 1.5}.
                          These values should represent the SUM for the entire portfolio
                          (e.g., sum of (per-share greek * quantity) for all legs).

    Returns:
        A string containing the LLM-generated analysis, or an error message string
        starting with '*' if analysis fails.
    """
    func_name = "fetch_greeks_analysis"
    logger.info(f"[{func_name}][{asset_symbol}] Request received for Greeks analysis.")

    if not portfolio_greeks or not isinstance(portfolio_greeks, dict):
        logger.warning(f"[{func_name}][{asset_symbol}] Invalid or empty portfolio_greeks dictionary provided.")
        return "*Cannot analyze Greeks: Input data is missing or invalid.*"

    # --- Check Gemini API Key ---
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        logger.error(f"[{func_name}][{asset_symbol}] Cannot generate Greeks analysis: Gemini API Key not configured.")
        return "*Greeks analysis feature not configured (API Key missing).*"

    # --- Format Greeks for the Prompt ---
    # Use the format_greek helper for consistency, NaN/None handling, and +/- sign
    delta_str = format_greek(portfolio_greeks.get('delta'))
    gamma_str = format_greek(portfolio_greeks.get('gamma'))
    theta_str = format_greek(portfolio_greeks.get('theta'))
    vega_str = format_greek(portfolio_greeks.get('vega'))
    rho_str = format_greek(portfolio_greeks.get('rho'))

    # --- Build the Prompt ---
    # Focus explicitly on interpreting the provided portfolio total numbers
    prompt = f"""You are an expert-level Options Trader and Risk Analyst. Analyze the **Total Portfolio Option Greeks** for a multi-leg options strategy on the underlying **{asset_symbol}**.

Your goal is to interpret what each Greek reveals about the **current directional exposure, risk profile, and sensitivity to market factors** – from a trader's perspective.

Assume these values represent the **net Greek exposure across the full position**.

---

🔢 **Portfolio Greeks Provided:**

- **Delta (Δ): {delta_str}**
- **Gamma (Γ): {gamma_str}**
- **Theta (Θ): {theta_str}** *(per day)*
- **Vega: {vega_str}** *(per 1% change in implied volatility)*
- **Rho (Ρ): {rho_str}** *(per 1% change in interest rate)*

---

### 🔍 **Analysis Instructions**:

#### 1️⃣ **Directional Bias and Sensitivities**  
- **Delta Insight:**  
  - What directional bias does the portfolio reflect? (bullish, bearish, or neutral)  
  - How much P/L movement can be expected for a $1 move in the underlying?
  - Is the Delta large enough to suggest a directional conviction or minor tilt?

- **Vega Insight:**  
  - Is the portfolio positioned to benefit from a rise or fall in implied volatility?  
  - Is Vega exposure large enough to be a primary risk/edge?  
  - Comment on whether the trader is effectively "long or short volatility".

- **Theta Insight:**  
  - Does the portfolio gain or lose value daily from time decay?
  - How impactful is this decay in dollar terms relative to the size of the position?

---

#### 2️⃣ **Convexity and Delta Stability (Gamma Analysis)**  
- **Gamma Insight:**  
  - Does Delta accelerate in the same direction as price movement (positive Gamma), or the opposite (negative Gamma)?  
  - What does Gamma imply about **how often** the position will need re-hedging or adjustment?
  - Is Gamma exposure consistent with an income-generating strategy or a directional play?

- **Delta-Gamma Interaction:**  
  - How does Gamma modify Delta during price moves?  
  - Example: Positive Gamma + Positive Delta = gains accelerate on up-moves but slow losses on down-moves.

---

#### 3️⃣ **Interest Rate Sensitivity (Rho)**  
- **Rho Insight:**  
  - Does the portfolio benefit from rising or falling interest rates?  
  - Is Rho exposure significant enough to impact strategy (usually minimal unless in longer-dated options)?

---

#### 4️⃣ **Synthesized Portfolio Risk Profile**  
- **Summarize:**  
  - What is the overall **strategic positioning** based on these Greeks (e.g., bullish with negative carry, Vega dependent, Gamma-scalping strategy)?
  - Which Greeks are **dominant risk factors**, and which are secondary?
  
- **Trader Takeaways:**  
  - What are the **key vulnerabilities** (e.g., large Theta decay, unstable Delta due to high Gamma, risk of Vega crush)?
  - Mention **scenarios where this position thrives vs where it suffers**.

---

### ⚠️ **Assumptions & Limitations**  
- This analysis is based *only* on the provided total Greeks.  
- No assumptions are made about strike prices, expirations, implied volatility surfaces, current underlying price, or market context.  
- This output is **not a recommendation**, but an **interpretation of risk and sensitivity metrics**.

---

### 📄 **Output Format Guidelines:**  
Use **clear section headers** and **bullet points**. Avoid generic summaries – focus on **specific, actionable insights** based solely on the Greek values."""

    logger.debug(f"[{func_name}][{asset_symbol}] Generating Greeks analysis prompt...")

    # --- Call LLM ---
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        logger.info(f"[{func_name}][{asset_symbol}] Calling Gemini API for Greeks analysis...")
        start_llm = asyncio.get_event_loop().time()

        # Add safety settings if desired
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     # ... other categories
        # ]
        # response = await model.generate_content_async(prompt, safety_settings=safety_settings)
        response = await model.generate_content_async(prompt)

        llm_duration = asyncio.get_event_loop().time() - start_llm
        logger.info(f"[{func_name}][{asset_symbol}] Gemini response for Greeks analysis received in {llm_duration:.3f}s.")

        # --- Robust Response Handling ---
        analysis_text = None
        try:
            # Check for blocks first
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
                logger.error(f"[{func_name}][{asset_symbol}] Gemini Greeks analysis blocked. Reason: {block_reason}")
                raise ValueError(f"Greeks analysis blocked by content filter ({block_reason}).")

            # Check for parts *after* checking for blocks
            if not response.parts:
                # If blocked, this might also be true, but the block reason is more specific
                feedback_info = f"Feedback: {response.prompt_feedback}" if response.prompt_feedback else "No parts and no feedback."
                logger.error(f"[{func_name}][{asset_symbol}] Gemini Greeks analysis response missing parts. {feedback_info}")
                # Avoid showing generic "missing parts" if it was actually blocked.
                if not (response.prompt_feedback and response.prompt_feedback.block_reason):
                     raise ValueError("Greeks analysis failed (No response parts received).")
                else:
                    # If it was blocked, the ValueError from the block check already covers it.
                    # This part is just for logging if parts are missing *without* a block.
                     pass # Error already raised by block check

            # If parts exist and not blocked, try getting text
            analysis_text = response.text

        except ValueError as e: # Catch errors from checks above
            logger.error(f"[{func_name}][{asset_symbol}] Error processing Gemini Greeks analysis response: {e}")
            raise ValueError(f"{e}") # Re-raise the user-friendly message
        except Exception as e: # Catch other unexpected errors
            logger.error(f"[{func_name}][{asset_symbol}] Unexpected error processing Gemini Greeks analysis response: {e}", exc_info=True)
            raise ValueError(f"Greeks analysis failed (Unexpected response processing error: {type(e).__name__}).")

        # Final check for empty text
        if not analysis_text or not analysis_text.strip():
            # This case might occur if the model responds but with empty content,
            # or if response.text fails for some reason after passing the .parts check.
            feedback_info = f"Feedback: {response.prompt_feedback}" if response.prompt_feedback else "No feedback available."
            logger.error(f"[{func_name}][{asset_symbol}] Gemini Greeks analysis response text is empty. {feedback_info}")
            raise ValueError("Greeks analysis failed (Empty response text received).")

        logger.info(f"[{func_name}][{asset_symbol}] Successfully generated Greeks analysis.")
        return analysis_text # Return the generated text

    except ValueError as gen_err: # Catch specific errors raised during response processing
        logger.error(f"[{func_name}][{asset_symbol}] Greeks analysis generation processing error: {gen_err}")
        return f"*Greeks Analysis Generation Error: {gen_err}*" # Return error message string
    except Exception as e: # Catch broader API errors (network, config, etc.)
        logger.error(f"[{func_name}][{asset_symbol}] Gemini API call or other unexpected error for Greeks analysis: {e}", exc_info=True)
        # Check for specific Google API errors if the library provides them, otherwise use generic
        # Example: if isinstance(e, google.api_core.exceptions.PermissionDenied): ...
        return f"*Greeks analysis generation failed (API/Processing Error: {type(e).__name__})*"



# ===============================================================
# Helper to get latest spot price from DB (used by multiple functions)
# ===============================================================
def get_latest_spot_price_from_db(asset: str, max_age_minutes: int = 5) -> Optional[Dict[str, Any]]:
    """
    Retrieves the latest spot price for an asset, prioritizing cache data.
    This function *now* fetches from the option chain cache instead of the DB
    to avoid DB schema issues, while maintaining the original function signature.
    The 'max_age_minutes' argument is effectively ignored in this cache-based implementation.

    Args:
        asset: The underlying asset symbol.
        max_age_minutes: This argument is ignored in this cache-based version.

    Returns:
        A dictionary {"spot_price": price, "timestamp": ts_str} if a valid
        spot price is found in the cache, otherwise None.
    """
    func_name = "get_latest_spot_price_from_db_USING_CACHE" # Rename for clarity in logs
    logger.debug(f"[{func_name}] Attempting to get spot price for '{asset}' from cache (max_age_minutes ignored).")

    try:
        # Call the function that handles cache logic and potential live fetch
        # This function should return the full option chain dictionary or None
        cached_data = get_cached_option(asset)

        if cached_data and isinstance(cached_data, dict):
            # Attempt to extract spot price from the nested structure
            records = cached_data.get("records", {})
            if not isinstance(records, dict):
                 logger.warning(f"[{func_name}] Cached data for {asset} has invalid 'records' structure.")
                 return None

            # Use safe float conversion for the spot price
            # Key is often 'underlyingValue' in NSE option chain data
            spot_price = _safe_get_float(records, "underlyingValue")

            if spot_price is not None and spot_price > 0:
                # Attempt to extract a timestamp associated with the cached data
                ts_str = records.get("timestamp") # Check common key name
                if not ts_str:
                    ts_str = cached_data.get("timestamp") # Check top level
                if not ts_str:
                     # If no timestamp, maybe use current time? Or just None? Let's use None.
                     ts_str = None
                     logger.debug(f"[{func_name}] Timestamp not found in cached data for {asset}.")
                else:
                     # Ensure ts_str is a string if found
                     ts_str = str(ts_str)
                     logger.debug(f"[{func_name}] Using timestamp from cached data: {ts_str}")


                logger.debug(f"[{func_name}] Found valid spot price in cache for {asset}: {spot_price}")
                # Return dictionary with standard keys expected by calculation functions
                return {"spot_price": spot_price, "timestamp": ts_str}
            else:
                logger.warning(f"[{func_name}] Invalid spot price ({spot_price}) found within cached data for {asset}.")
                return None
        else:
            # Log if cache miss or invalid data type returned by get_cached_option
            logger.debug(f"[{func_name}] No valid cached data found via get_cached_option for {asset}.")
            return None

    except Exception as e:
        # Log any unexpected errors during the cache access or processing
        logger.error(f"[{func_name}] Unexpected error retrieving/processing spot price from cache for {asset}: {e}", exc_info=True)
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

    
@app.get("/get_asset_details")
async def get_asset_details_endpoint(asset: str):
    """
    Fetches details like lot size for a given asset.
    """
    if not asset:
        raise HTTPException(status_code=400, detail="Asset parameter is required.")

    logger.info(f"API: Fetching details for asset: {asset}")
    try:
        # Call your existing database function
        lot_size = get_lot_size(asset) # Use the function you provided

        # You could fetch other details here too if needed (e.g., spot price)

        if lot_size is None:
            # Logged within get_lot_size, but raise specific HTTP error
             logger.warning(f"API: Lot size not found for asset '{asset}'.")
             # Return success but with null lot size, let frontend handle it
             # Or raise HTTPException(status_code=404, detail=f"Lot size not found for asset: {asset}")
             return {"asset": asset, "lot_size": None}

        logger.info(f"API: Returning details for {asset}: lot_size={lot_size}")
        return {"asset": asset, "lot_size": lot_size}

    except Exception as e:
        logger.error(f"API: Unexpected error fetching details for {asset}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while fetching asset details.")

# Position Management Endpoints
#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------
@app.post("/api/positions/add")
async def add_position(position_data: PositionInput):
    """
    Adds a new position based on LTP selection or manual entry.
    """
    global strategy_positions
    
    try:
        logger.info(f"Adding position: {position_data}")
        
        # Create position dictionary with standardized format
        new_position = {
            "symbol": position_data.symbol,
            "strike": position_data.strike,
            "type": position_data.type,  # CE or PE
            "op_type": "c" if position_data.type == "CE" else "p",  # Convert to 'c' or 'p' for internal use
            "tr_type": "b",  # Default to buy
            "op_pr": position_data.price,
            "quantity": position_data.quantity,
            "lot": 1,  # Default values, adjust as needed
            "lot_size": get_lot_size(position_data.symbol) or 50  # Fallback to 50 if lot size fetch fails
        }
        
        # Add to strategy positions
        strategy_positions.append(new_position)
        
        logger.info(f"Added new position: {new_position}")
        
        return {"status": "success", "message": "Position added", "position": new_position}
    
    except Exception as e:
        logger.error(f"Error adding position: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add position: {str(e)}")

@app.delete("/api/positions/{position_index}")
async def delete_position(position_index: int):
    """
    Deletes a position by index.
    """
    global strategy_positions
    
    try:
        logger.info(f"Deleting position at index: {position_index}")
        
        if position_index < 0 or position_index >= len(strategy_positions):
            raise HTTPException(status_code=404, detail=f"Position index {position_index} out of range")
        
        deleted_position = strategy_positions.pop(position_index)
        
        logger.info(f"Deleted position: {deleted_position}")
        
        return {"status": "success", "message": "Position deleted", "position": deleted_position}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting position: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete position: {str(e)}")

@app.delete("/api/positions")
async def clear_positions():
    """
    Clears all positions.
    """
    global strategy_positions
    
    try:
        logger.info("Clearing all positions")
        
        positions_count = len(strategy_positions)
        strategy_positions.clear()
        
        logger.info(f"Cleared {positions_count} positions")
        
        return {"status": "success", "message": f"Cleared {positions_count} positions"}
    
    except Exception as e:
        logger.error(f"Error clearing positions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear positions: {str(e)}")

@app.get("/api/positions")
async def get_positions():
    """
    Returns all current positions.
    """
    global strategy_positions
    
    try:
        logger.info(f"Returning {len(strategy_positions)} positions")
        return {"positions": strategy_positions}
    
    except Exception as e:
        logger.error(f"Error retrieving positions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve positions: {str(e)}")

#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------



@app.post("/update_selected_asset/")
async def update_selected_asset(request: AssetUpdateRequest):
    """
    Updates the global selected_asset variable based on frontend selection.
    """
    global selected_asset
    asset_name = request.asset.strip().upper() if isinstance(request.asset, str) else None

    if not asset_name:
        logger.warning(f"[update_selected_asset] Invalid asset name received: {request.asset}")
        return JSONResponse(status_code=400, content={"success": False, "message": "Invalid asset name provided"})

    old_asset = selected_asset
    selected_asset = asset_name
    logger.info(f"[update_selected_asset] Selected asset updated from '{old_asset}' to '{selected_asset}'")
    return {"success": True, "message": f"Selected asset updated to: {selected_asset}"}




@app.get("/get_news", tags=["Data"])
async def get_news_endpoint(asset: str = Query(...)):
    """
    Fetches the latest news headlines and summaries for a given asset using feedparser.
    """
    endpoint_name = "get_news_endpoint"
    logger.info(f"[{endpoint_name}] Request received for news: Asset={asset}")
    asset_upper = asset.strip().upper()
    if not asset_upper:
        raise HTTPException(status_code=400, detail="Asset name required.")

    try:
        # Use the existing robust async news fetching function (which uses feedparser)
        news_items = await fetch_latest_news_async(asset_upper)

        # Check if the fetch itself indicated an error internally
        # Use more robust check for error messages or empty states
        is_error_or_empty = not news_items or any(
            err_indicator in news_items[0].get("headline", "").lower()
            for err_indicator in ["error", "no recent news", "no relevant news", "timeout", "not available"]
        )

        if is_error_or_empty:
            if not news_items:
                 logger.warning(f"[{endpoint_name}] News fetch for {asset} returned empty list.")
                 # Standardize response for no news found
                 news_items = [{"headline": f"No recent news found for {asset}.", "summary": "", "link": "#"}]
            else:
                # Log the specific error/state returned by the fetch function
                logger.warning(f"[{endpoint_name}] News fetch for {asset} returned an error/empty state: {news_items[0].get('headline', 'N/A')}")
            # Return the error/empty message list with a 200 OK status
            return {"news": news_items}
        else:
            logger.info(f"[{endpoint_name}] Successfully fetched {len(news_items)} news items for {asset}")
            return {"news": news_items}

    except Exception as e:
        logger.error(f"[{endpoint_name}] Unexpected error in news endpoint for {asset}: {e}", exc_info=True)
        # Use 503 Service Unavailable for internal server errors during fetch
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
    Uses updated calculation functions with improved internal logic.
    Handles data preparation and concurrent execution of calculations.
    Returns chart HTML content along with other results.
    """
    asset = request.asset
    strategy_input = request.strategy
    func_name = "get_payoff_chart_endpoint" # For logging context
    logger.info(f"[{func_name}] Endpoint Request received: Asset={asset}, Legs={len(strategy_input)}")

    if not asset or not strategy_input:
        logger.warning(f"[{func_name}] Bad Request: Missing asset or strategy legs.")
        raise HTTPException(status_code=400, detail="Missing asset or strategy legs")

    # --- Step 1: Fetch Prerequisite Spot Price (Async) ---
    logger.info(f"[{func_name}] Starting Step 1: Fetch Spot Price...")
    spot_price = None
    try:
        spot_response = await get_spot_price(asset)
        spot_price = spot_response.get('spot_price')
        if spot_price is None:
            raise ValueError("Spot price key not found in response")
        spot_price = float(spot_price) # Validate and convert
        if spot_price <= 0:
            raise ValueError(f"Invalid spot price value received: {spot_price}")
        logger.info(f"[{func_name}] Spot price fetched successfully: {spot_price}")
    except HTTPException as http_err:
        logger.error(f"[{func_name}] HTTP error during spot price fetch: {http_err.detail}")
        raise http_err
    except (ValueError, TypeError) as val_err:
        logger.error(f"[{func_name}] Prerequisite spot price validation failed: {val_err}")
        raise HTTPException(status_code=404, detail=f"Invalid or unavailable spot price data for {asset}: {val_err}")
    except Exception as e:
        logger.error(f"[{func_name}] Unexpected error fetching spot price: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Server error fetching initial market data for {asset}")

    # --- Step 2: Prepare Strategy Data (Sync) ---
    logger.info(f"[{func_name}] Starting Step 2: Prepare Strategy Data...")
    prepared_strategy_data = None
    try:
        strategy_input_dicts = [leg.dict() for leg in strategy_input]
        prepared_strategy_data = prepare_strategy_data(strategy_input_dicts, asset, spot_price)
        if not prepared_strategy_data:
            logger.error(f"[{func_name}] No valid strategy legs remaining after preparation for asset {asset}.")
            raise HTTPException(status_code=400, detail="Invalid, incomplete, or inconsistent strategy leg data provided. Check input and market data availability.")
        logger.info(f"[{func_name}] Strategy data prepared successfully for {len(prepared_strategy_data)} legs.")
    except HTTPException as http_err:
        raise http_err
    except Exception as prep_err:
         logger.error(f"[{func_name}] Error during strategy data preparation: {prep_err}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Server error preparing strategy data: {prep_err}")

    # --- Step 3: Perform Calculations Concurrently ---
    logger.info(f"[{func_name}] Starting Step 3: Concurrent Calculations...")
    start_calc_time = time.monotonic()
    results = {}
    metrics_result = None # Initialize

    try:
        # --- Run Metrics Calculation First (Fundamental) ---
        logger.debug(f"[{func_name}] Calculating strategy metrics...")
        metrics_result = await asyncio.to_thread(
            calculate_strategy_metrics,
            prepared_strategy_data,
            spot_price,  # <-- CORRECTED ARGUMENT ORDER
            asset
        )
        if metrics_result is None:
             logger.error(f"[{func_name}] Core strategy metrics calculation failed (returned None). Unable to proceed.")
             raise HTTPException(status_code=500, detail="Core strategy metrics calculation failed. Check leg data validity and server logs.")
        results["metrics"] = metrics_result
        logger.debug(f"[{func_name}] Metrics calculation successful.")

        # --- Run Chart, Tax, Greeks Concurrently ---
        logger.debug(f"[{func_name}] Calculating Chart, Tax, and Greeks concurrently...")
        tasks = {
            "chart": asyncio.to_thread( # Correct
                generate_payoff_chart_matplotlib,
                prepared_strategy_data,
                asset,
                spot_price,
                metrics_result
            ),
            "tax": asyncio.to_thread( # Correct
                calculate_option_taxes,
                prepared_strategy_data,
                spot_price,
                asset
            ),
            "greeks": asyncio.to_thread( # CORRECTED ARGUMENT ORDER
                calculate_option_greeks,
                prepared_strategy_data,
                asset,
                spot_price
            )
        }
        # Gather results
        task_values = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # --- Process results ---
        results_map = list(tasks.keys()) # ["chart", "tax", "greeks"]
        results["chart_html"] = None
        results["tax"] = None
        results["greeks"] = []

        failed_tasks = []
        for i, task_name in enumerate(results_map):
            task_result = task_values[i]
            if isinstance(task_result, Exception):
                failed_tasks.append(task_name)
                logger.error(f"[{func_name}] Exception during concurrent calculation of '{task_name}': {type(task_result).__name__}: {task_result}", exc_info=False)
            else:
                if task_name == "chart": results["chart_html"] = task_result
                elif task_name == "tax": results["tax"] = task_result
                elif task_name == "greeks": results["greeks"] = task_result if isinstance(task_result, list) else []

        if failed_tasks:
             logger.warning(f"[{func_name}] One or more non-critical calculations failed: {', '.join(failed_tasks)}. Results may be partial.")

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"[{func_name}] Unexpected error during calculation phase: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected Server Error during strategy analysis.")

    calc_duration = time.monotonic() - start_calc_time
    logger.info(f"[{func_name}] Calculations finished in {calc_duration:.3f}s")

    # --- Step 4: Return Results ---
    success_status = results["metrics"] is not None
    chart_generated = results.get("chart_html") is not None
    tax_calculated = results.get("tax") is not None
    greeks_calculated = len(results.get("greeks", [])) > 0

    message = "Analysis complete."
    if not success_status: message = "Core analysis failed."
    elif failed_tasks: message = f"Analysis complete, but failed to calculate: {', '.join(failed_tasks)}."
    else:
        missing_parts = []
        if not chart_generated: missing_parts.append("chart")
        if not tax_calculated: missing_parts.append("taxes")
        if not greeks_calculated: missing_parts.append("greeks")
        if missing_parts: message = f"Analysis complete, but some parts are unavailable ({', '.join(missing_parts)})."

    final_response = {
        "success": success_status,
        "message": message,
        "metrics": results.get("metrics"),
        "charges": results.get("tax"),
        "greeks": results.get("greeks", []),
        # Ensure this key matches the frontend check
        "chart_figure_json": results.get("chart_html") # Assuming 'chart_html' now holds the JSON string
        # Or rename the key in the 'tasks' / 'results' dict earlier if preferred
        # "chart_figure_json": results.get("chart")
    }
    logger.info(f"[{func_name}] Returning response. Success: {success_status}, Chart: {chart_generated}, Tax: {tax_calculated}, Greeks: {greeks_calculated}")
    return final_response




# --- LLM Stock Analysis Endpoint (FIXED 404/Data Handling) ---
@app.post("/get_stock_analysis", tags=["Analysis & Payoff"])
async def get_stock_analysis_endpoint(request: StockRequest):
    """
    Fetches stock data (via RapidAPI) and news (via feedparser), then generates
    an LLM analysis focused on insights for Options Traders.
    Handles missing data gracefully. Returns 404 only if essential stock data
    (e.g., price from RapidAPI) is unavailable.
    """
    asset = request.asset.strip().upper()
    endpoint_name = "get_stock_analysis_endpoint"
    if not asset: raise HTTPException(status_code=400, detail="Asset name required.")

    analysis_cache_key = f"analysis_rapidapi_{asset}" # Include source in key
    cached_analysis = analysis_cache.get(analysis_cache_key)
    if cached_analysis:
        logger.info(f"[{endpoint_name}][{asset}] Cache hit analysis.")
        return {"analysis": cached_analysis}

    # --- Symbol Mapping & Region Definition ---
    # Keep the existing mapping logic, define the region (assuming IN for now)
    region_code = "IN" # Explicitly define region for RapidAPI call
    symbol_map = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS", # Verify this symbol if needed
        "NIFTY 50": "^NSEI",
        "NIFTY BANK": "^NSEBANK"
        # Add more mappings for BSE or other exchanges if needed
    }
    # Determine symbol: Use map, default to .NS if not index, else use asset as is
    stock_symbol = symbol_map.get(asset, f"{asset}.NS" if '^' not in asset and '.' not in asset else asset)

    logger.info(f"[{endpoint_name}][{asset}] Analysis request. Using Symbol='{stock_symbol}', Region='{region_code}'")

    stock_data: Optional[Dict[str, Any]] = None
    latest_news: Optional[List[Dict[str, str]]] = []

    try:
        # --- Fetch Data Concurrently ---
        logger.debug(f"[{endpoint_name}][{asset}] Starting concurrent fetch (RapidAPI Stock + Feedparser News)...")
        start_fetch = time.monotonic()

        # === CORE CHANGE: Call new fetch_stock_data_async with symbol AND region ===
        stock_task = asyncio.create_task(fetch_stock_data_async(stock_symbol, region=region_code))
        # ==========================================================================
        news_task = asyncio.create_task(fetch_latest_news_async(asset)) # Use original asset name for news

        results = await asyncio.gather(stock_task, news_task, return_exceptions=True)
        fetch_duration = time.monotonic() - start_fetch
        logger.debug(f"[{endpoint_name}][{asset}] Concurrent fetch completed in {fetch_duration:.3f}s")

        # --- Process Stock Data Result (from RapidAPI) ---
        if isinstance(results[0], Exception):
            logger.error(f"[{endpoint_name}][{asset}({stock_symbol})] Unexpected error during RapidAPI stock data fetch task: {results[0]}", exc_info=results[0])
            raise HTTPException(status_code=503, detail=f"Server error fetching stock data for {stock_symbol} via external API.")
        elif results[0] is None:
            # fetch_stock_data_async (RapidAPI version) returns None if essential data (chart/price) is missing
            logger.error(f"[{endpoint_name}][{asset}({stock_symbol})] CRITICAL - Essential stock data (e.g., price/chart) could not be determined from RapidAPI. Symbol might be invalid for region '{region_code}', delisted, or API issue.")
            # 404 Not Found is appropriate if the core data for the requested asset isn't available
            raise HTTPException(status_code=404, detail=f"Essential stock data not found for: {stock_symbol} (Region: {region_code}). Verify symbol or try later.")
        else:
            stock_data = results[0]
            logger.info(f"[{endpoint_name}][{asset}({stock_symbol})] Successfully fetched RapidAPI stock data.")

        # --- Process News Data Result (from feedparser) ---
        if isinstance(results[1], Exception):
            logger.error(f"[{endpoint_name}][{asset}] Unexpected error during news fetch task: {results[1]}", exc_info=results[1])
            # Provide placeholder indicating news fetch failure, but don't fail the whole request
            latest_news = [{"headline": f"Server error fetching news for {asset}.", "summary": "Task failed.", "link": "#"}]
        elif not results[1]:
             logger.warning(f"[{endpoint_name}][{asset}] News fetch returned an empty list unexpectedly.")
             latest_news = [{"headline": f"No news data available for {asset}.", "summary": "", "link": "#"}]
        else:
            latest_news = results[1]
            # Log if the news fetch itself reported an error/empty state
            is_news_error_or_empty = any(
                err_indicator in latest_news[0].get("headline", "").lower()
                for err_indicator in ["error", "no recent news", "no relevant news", "timeout", "not available"]
            )
            if is_news_error_or_empty:
                logger.warning(f"[{endpoint_name}][{asset}] News fetch resulted in an error/empty state: {latest_news[0]['headline']}")
            else:
                logger.info(f"[{endpoint_name}][{asset}] Successfully processed news fetch result ({len(latest_news)} items).")

    except HTTPException as http_err:
         # Log and re-raise specific 4xx/5xx errors from fetch processing
         logger.error(f"[{endpoint_name}][{asset}({stock_symbol})] Raising HTTPException status {http_err.status_code}: {http_err.detail}")
         raise http_err
    except Exception as e:
        # Catch-all for unexpected errors during the fetch orchestration phase
        logger.error(f"[{endpoint_name}][{asset}] Unexpected error during data fetch orchestration: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Internal server error during data fetching.")

    # --- Generate Analysis using LLM ---
    # stock_data is guaranteed to be a populated Dict here if no exception was raised above

    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        logger.error(f"[{endpoint_name}][{asset}] Cannot generate analysis: Gemini API Key not configured.")
        raise HTTPException(status_code=501, detail="Analysis feature not configured (API Key missing).")

    logger.debug(f"[{endpoint_name}][{asset}] Building options-focused analysis prompt using RapidAPI data...")

    # === CORE CHANGE: Use the updated prompt builder function ===
    # Pass the original 'asset' name for display, but use fetched 'stock_data' and 'latest_news'
    prompt = build_stock_analysis_prompt_for_options(asset, stock_data, latest_news)
    # ============================================================

    try:
        # Use the same Gemini model and generation logic
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        logger.info(f"[{endpoint_name}][{asset}] Generating Gemini analysis (options focus)...")
        start_llm = time.monotonic()

        response = await model.generate_content_async(prompt) # Async call

        llm_duration = time.monotonic() - start_llm
        logger.info(f"[{endpoint_name}][{asset}] Gemini response received in {llm_duration:.3f}s.")

        # Use the same robust Gemini response handling logic as before
        analysis_text = None
        try:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
                logger.error(f"[{endpoint_name}][{asset}] Gemini response blocked. Reason: {block_reason}")
                # Raise a specific error that can be caught below
                raise ValueError(f"Analysis blocked by content filter ({block_reason}).")

            if not response.parts:
                feedback_info = f"Feedback: {response.prompt_feedback}" if response.prompt_feedback else "No feedback."
                logger.error(f"[{endpoint_name}][{asset}] Gemini response missing parts. {feedback_info}")
                raise ValueError(f"Analysis generation failed (No response parts received).")

            analysis_text = response.text # Extract text

        except ValueError as e:
            # Catch errors from response processing (blocked, missing parts)
            logger.error(f"[{endpoint_name}][{asset}] Error processing Gemini response object: {e}")
            # Raise the same error message to be caught by the outer try-except
            raise ValueError(f"{e}")
        except Exception as e:
            # Catch unexpected errors during response processing
            logger.error(f"[{endpoint_name}][{asset}] Unexpected error processing Gemini response: {e}", exc_info=True)
            raise ValueError(f"Analysis generation failed (Unexpected response processing error: {type(e).__name__}).")

        # Check for empty text after successful processing
        if not analysis_text or not analysis_text.strip():
            feedback_info = f"Feedback: {response.prompt_feedback}" if response.prompt_feedback else "No feedback."
            logger.error(f"[{endpoint_name}][{asset}] Gemini response text is empty. {feedback_info}")
            raise ValueError(f"Analysis generation failed (Empty response text received).")

        # Cache the valid analysis text
        analysis_cache[analysis_cache_key] = analysis_text
        logger.info(f"[{endpoint_name}][{asset}] Successfully generated and cached options-focused analysis (RapidAPI source).")
        return {"analysis": analysis_text}

    except ValueError as gen_err:
        # Catch specific errors raised during response processing (blocked, missing parts, empty text)
        logger.error(f"[{endpoint_name}][{asset}] Analysis generation processing error: {gen_err}")
        # Return a 503 with the user-friendly error message from the ValueError
        raise HTTPException(status_code=503, detail=f"Analysis Generation Error: {gen_err}")
    except Exception as e:
        # Catch broader errors from the genai API call itself or other unexpected issues
        logger.error(f"[{endpoint_name}][{asset}] Gemini API call or other unexpected error during generation: {e}", exc_info=True)
        # Provide a generic error for unexpected API issues
        raise HTTPException(status_code=503, detail=f"Analysis generation failed (External API/Processing Error: {type(e).__name__})")
        # ===============================================================


@app.post("/get_greeks_analysis", tags=["Analysis & Payoff"], summary="Get LLM analysis based on portfolio Greeks")
async def get_greeks_analysis_endpoint(request: GreeksAnalysisRequest):
    """
    Generates an LLM analysis based *only* on the provided total portfolio Greeks.
    The Greeks should be calculated client-side and represent the entire portfolio.
    """
    endpoint_name = "get_greeks_analysis_endpoint"
    logger.info(f"[{endpoint_name}] Request received for asset: {request.asset_symbol}")
    logger.debug(f"[{endpoint_name}] Received Greeks data: {request.portfolio_greeks.dict()}") # Log received data

    # Convert Pydantic model back to dict for the function
    greeks_dict = request.portfolio_greeks.dict()

    analysis = await fetch_greeks_analysis(request.asset_symbol, greeks_dict)

    # Check if the function returned an error message string
    if isinstance(analysis, str) and analysis.startswith("*"):
        error_message = analysis.strip('*')
        logger.warning(f"[{endpoint_name}][{request.asset_symbol}] Greeks analysis generation failed: {error_message}")
        # Map specific errors to appropriate HTTP status codes
        if "API Key missing" in error_message:
             raise HTTPException(status_code=501, detail=error_message) # 501 Not Implemented
        elif "blocked by content filter" in error_message:
             raise HTTPException(status_code=400, detail=error_message) # 400 Bad Request (content issue)
        else:
             # Other generation errors typically indicate server-side issues
             raise HTTPException(status_code=503, detail=error_message) # 503 Service Unavailable
    elif not analysis: # Should not happen if errors return strings, but as a fallback
         logger.error(f"[{endpoint_name}][{request.asset_symbol}] Greeks analysis returned None unexpectedly.")
         raise HTTPException(status_code=503, detail="Unknown error generating Greeks analysis.")
    else:
         # Success
         logger.info(f"[{endpoint_name}][{request.asset_symbol}] Successfully generated Greeks analysis.")
         return {"greeks_analysis": analysis} # Return the valid analysis


# ===============================================================
# Main Execution Block 
# ===============================================================
# # For live hosting
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
#===============================================================


# Local hosting
# if __name__ == "__main__":
#     host = os.getenv("HOST", "127.0.0.1") 
#     # ----------------------------
#     port = int(os.getenv("PORT", 8000)) 
#     reload = os.getenv("RELOAD", "true").lower() == "true" 
#     # ---------------------------------------------

#     log_level = "debug" if reload else "info" # Keep this logic

#     logger.info(f"Starting Uvicorn server on http://{host}:{port} (Reload: {reload}, LogLevel: {log_level})")
#     uvicorn.run(
#         "backend.app:app", # Point to the FastAPI app instance
#         host=host,
#         port=port,
#         reload=reload, # Enable auto-reload for local development if needed
#         log_level=log_level
#         # Consider adding reload_dirs=["."] if reload=True and you have other modules
#     )

