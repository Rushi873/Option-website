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
import feedparser

# --- Calculation & Plotting ---
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import math
import mibian


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
PAYOFF_POINTS = 200 # Increased points for smoother curve
BREAKEVEN_CLUSTER_GAP_PCT = 0.01

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

def cluster_points(points: List[float], tolerance_ratio: float, reference_value: float) -> List[float]:
    """Clusters nearby breakeven points."""
    if not points:
        return []
    points.sort()
    clustered = []
    current_cluster = [points[0]]
    tolerance_value = tolerance_ratio * reference_value

    for i in range(1, len(points)):
        if abs(points[i] - current_cluster[-1]) <= tolerance_value:
            current_cluster.append(points[i])
        else:
            clustered.append(np.mean(current_cluster)) # Use mean of the cluster
            current_cluster = [points[i]]

    clustered.append(np.mean(current_cluster)) # Add the last cluster
    return [round(p, 2) for p in clustered] # Return rounded means

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
    raw_strategy: List[Dict[str, Any]],
    asset: str,
    spot_price: float # Pass spot price for potential use in IV extraction
) -> List[Dict[str, Any]]:
    """
    Validates raw strategy leg data, calculates Days to Expiry (DTE),
    extracts Implied Volatility (IV) using a helper function, determines
    lot size, and formats the data for downstream calculation functions.

    Args:
        raw_strategy: List of dictionaries, likely from API request.
                      Expected keys per dict: 'expiry_date', 'strike_price',
                      'option_type', 'tr_type', 'option_price', 'lots'.
                      Optional: 'lot_size'.
        asset: The underlying asset symbol (e.g., "NIFTY").
        spot_price: The current spot price of the underlying asset.

    Returns:
        A list of dictionaries, where each dictionary represents a valid,
        prepared strategy leg with keys required by calculation functions
        (op_type, strike, tr_type, op_pr, lot, lot_size, iv, days_to_expiry).
        Invalid legs from the input are skipped.
    """
    func_name = "prepare_strategy_data"
    logger.info(f"[{func_name}] Preparing data for {len(raw_strategy)} raw legs for asset {asset} (Spot: {spot_price}).")
    prepared_data: List[Dict[str, Any]] = []
    today = date.today()

    # --- Get Default Lot Size for Fallback ---
    default_lot_size = None
    try:
        default_lot_size = get_lot_size(asset)
        if default_lot_size is None or not isinstance(default_lot_size, int) or default_lot_size <= 0:
             raise ValueError(f"Invalid default lot size ({default_lot_size}) retrieved for asset {asset}")
        logger.debug(f"[{func_name}] Using default lot size: {default_lot_size} for {asset}")
    except Exception as lot_err:
         logger.error(f"[{func_name}] Failed to get valid default lot size for {asset}: {lot_err}. Cannot prepare legs without a default.", exc_info=True)
         return [] # Cannot proceed without a valid default

    # --- Process Each Raw Leg ---
    for i, leg_input in enumerate(raw_strategy):
        leg_desc = f"Leg {i+1}"
        try:
            logger.debug(f"[{func_name}] Processing raw {leg_desc}: {leg_input}")

            # --- Extract and Validate Core Inputs ---
            expiry_str = leg_input.get('expiry_date') # Expects "YYYY-MM-DD"
            # Use safe float conversion for strike
            strike_raw = leg_input.get('strike_price')
            strike_price = _safe_get_float({'sp': strike_raw}, 'sp')

            # Standardize option type (CE/PE -> c/p)
            opt_type_req = str(leg_input.get('option_type', '')).upper()
            op_type = None
            if opt_type_req == 'CE': op_type = 'c'
            elif opt_type_req == 'PE': op_type = 'p'

            # Standardize transaction type (b/s)
            tr_type = str(leg_input.get('tr_type', '')).lower()

            # Basic validation for essential fields
            error_msg = None
            if not expiry_str: error_msg = "Missing 'expiry_date'"
            elif strike_price is None or strike_price <= 0: error_msg = f"Invalid 'strike_price' ({strike_raw})"
            elif op_type is None: error_msg = f"Invalid 'option_type' ({leg_input.get('option_type')})"
            elif tr_type not in ('b', 's'): error_msg = f"Invalid 'tr_type' ({leg_input.get('tr_type')})"

            if error_msg:
                logger.warning(f"[{func_name}] Skipping {leg_desc} due to basic info error: {error_msg}. Input: {leg_input}")
                continue

            # --- Calculate Days to Expiry (DTE) ---
            expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            days_to_expiry = (expiry_dt - today).days
            # Allow DTE=0 (expiry today), but reject past expiry
            if days_to_expiry < 0:
                 logger.warning(f"[{func_name}] Skipping {leg_desc} with expiry {expiry_str} in the past (DTE={days_to_expiry}).")
                 continue
            logger.debug(f"[{func_name}] {leg_desc} DTE calculated: {days_to_expiry}")

            # --- Extract Implied Volatility (IV) ---
            # Assumes extract_iv handles API calls/DB lookups/mocks and returns float or None
            iv_extracted = extract_iv(asset, strike_price, expiry_str, opt_type_req)
            iv_float = 0.0 # Default placeholder

            if iv_extracted is None:
                logger.warning(f"[prepare_strategy_data] IV extraction failed for {leg_desc} ({opt_type_req} {strike_price} {expiry_str}). Using 0.0 placeholder. Greeks may be skipped or inaccurate.")
            elif not isinstance(iv_extracted, (int, float)) or iv_extracted < 0:
                 logger.warning(f"[prepare_strategy_data] IV extraction for {leg_desc} returned invalid value {iv_extracted}. Using 0.0 placeholder. Greeks may be skipped or inaccurate.")
            else:
                 iv_float = float(iv_extracted)
                 # Warn if IV is effectively zero, might cause issues in calculations
                 if iv_float <= 1e-6:
                     logger.warning(f"[prepare_strategy_data] IV for {leg_desc} is near zero ({iv_float}). Using 0.0 placeholder. Greeks may be skipped.")
                     iv_float = 0.0 # Force to exactly 0.0 for consistency downstream

            logger.debug(f"[{func_name}] {leg_desc} IV assigned: {iv_float}")

            # --- Extract and Validate Other Numeric Inputs ---
            # Use safe float conversion, default to 0.0 if missing/invalid
            option_price_raw = leg_input.get('option_price')
            option_price = _safe_get_float({'op': option_price_raw}, 'op', default=0.0)
            if option_price < 0:
                 logger.warning(f"[{func_name}] Invalid negative option_price ({option_price_raw}) for {leg_desc}. Using 0.0.")
                 option_price = 0.0

            # Use safe int conversion, default to 1 if missing/invalid? Or fail? Let's fail if invalid.
            lots_raw = leg_input.get('lots')
            lots = _safe_get_int({'l': lots_raw}, 'l')
            if lots is None or lots <= 0:
                 logger.warning(f"[{func_name}] Skipping {leg_desc} due to invalid 'lots' ({lots_raw}).")
                 continue

            # --- Determine Final Lot Size ---
            lot_size_raw = leg_input.get('lot_size')
            leg_specific_lot_size = _safe_get_int({'ls': lot_size_raw}, 'ls')
            final_lot_size = default_lot_size # Start with default

            if leg_specific_lot_size is not None and leg_specific_lot_size > 0:
                final_lot_size = leg_specific_lot_size # Use leg-specific if valid
                logger.debug(f"[{func_name}] {leg_desc} Using leg-specific lot size: {final_lot_size}")
            else:
                 # Log if leg-specific was provided but invalid
                 if lot_size_raw is not None:
                      logger.warning(f"[{func_name}] Invalid leg-specific lot size '{lot_size_raw}' for {leg_desc}. Using default: {default_lot_size}")
                 else:
                      logger.debug(f"[{func_name}] {leg_desc} No valid leg-specific lot size. Using default: {default_lot_size}")
            # Final check on the chosen lot size (should be redundant if default is valid)
            if final_lot_size <= 0:
                 logger.error(f"[{func_name}] Skipping {leg_desc} due to final invalid lot size ({final_lot_size}). Check default lot size for {asset}.")
                 continue

            # --- Assemble Prepared Leg Data ---
            # Use keys expected by downstream calculation functions
            prepared_leg = {
                "op_type": op_type,           # 'c' or 'p'
                "strike": strike_price,       # float
                "tr_type": tr_type,           # 'b' or 's'
                "op_pr": option_price,        # float (premium per share)
                "lot": lots,                  # int
                "lot_size": final_lot_size,   # int (validated)
                "iv": iv_float,               # float (validated IV, can be 0.0)
                "days_to_expiry": days_to_expiry, # int (can be 0)
                # Optional: Keep original info for deeper debugging if needed
                # "original_input": leg_input
                "expiry_date_str": expiry_str, # Keep for reference
            }
            prepared_data.append(prepared_leg)
            logger.debug(f"[{func_name}] Successfully prepared {leg_desc}: {prepared_leg}")

        except (ValueError, TypeError, KeyError) as prep_err:
            # Catch errors during conversion (strptime, float, int) or unexpected key issues
            logger.error(f"[{func_name}] Error preparing {leg_desc}: {prep_err}. Skipping leg. Input: {leg_input}", exc_info=False)
            # Skip this leg and continue with the next
            continue
        except Exception as unexpected_err:
             # Catch any other unexpected error during leg processing
             logger.error(f"[{func_name}] UNEXPECTED Error preparing {leg_desc}: {unexpected_err}. Skipping leg. Input: {leg_input}", exc_info=True)
             continue

    # --- Final Log ---
    logger.info(f"[{func_name}] Finished preparation. Prepared {len(prepared_data)} valid legs out of {len(raw_strategy)} raw legs for asset {asset}.")
    return prepared_data


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

def generate_payoff_chart_matplotlib( # <<< Function name retained
    strategy_data: List[Dict[str, Any]],
    asset: str,
    spot_price: float,
    strategy_metrics: Optional[Dict[str, Any]] # Pass calculated metrics (used for potential future enhancements, not directly in this plot logic)
) -> Optional[str]: # <<< Return type is now string (HTML)
    """
    Generates an interactive Plotly payoff chart resembling the target style.
    Calculates and displays standard deviation lines if possible.
    Returns the chart HTML content as a string, or None on failure.
    Optionally saves a standalone HTML file for local viewing.

    NOTE: Function name is 'generate_payoff_chart_matplotlib' for compatibility,
          but the implementation uses Plotly.
    """
    # <<< Use the *correct* function name for logging consistency >>>
    func_name = "generate_payoff_chart_matplotlib"
    logger.info(f"[{func_name}] Generating Plotly chart (via function '{func_name}') for {len(strategy_data)} leg(s), asset: {asset}")
    logger.debug(f"[{func_name}] Input strategy_data: {strategy_data}")
    logger.debug(f"[{func_name}] Input strategy_metrics: {strategy_metrics}") # Logged but not used in current plot logic
    start_time = time.monotonic()
    # Define filename for the optional standalone file
    # Consider adding a timestamp or unique ID if this runs concurrently
    output_filename_standalone = f"{asset}_payoff_chart_plotly_standalone_{func_name}.html" # Added func_name to filename for clarity

    try:
        # 1. Fetch Prerequisites (Spot, Default Lot Size)
        logger.debug(f"[{func_name}] Fetching prerequisites...")
        spot_price_info = get_latest_spot_price_from_db(asset)
        spot_price = None
        if spot_price_info and 'spot_price' in spot_price_info:
             spot_price = _safe_get_float(spot_price_info, 'spot_price')
             logger.debug(f"[{func_name}] Spot price from DB: {spot_price}")
        if spot_price is None:
            logger.debug(f"[{func_name}] Spot price from DB failed or missing key, trying cache...")
            cached_data = get_cached_option(asset)
            spot_price = _safe_get_float(cached_data.get("records", {}), "underlyingValue") if cached_data else None
            logger.debug(f"[{func_name}] Spot price from Cache/Live: {spot_price}")


        default_lot_size = get_lot_size(asset)
        logger.debug(f"[{func_name}] Default lot size from config/DB: {default_lot_size}")

        if spot_price is None or not isinstance(spot_price, (int, float)) or spot_price <= 0:
            logger.error(f"[{func_name}] Prerequisite validation failed! Invalid or missing spot price ({spot_price}) for chart generation")
            raise ValueError(f"Invalid or missing spot price ({spot_price}) for chart generation")
        if default_lot_size is None or not isinstance(default_lot_size, int) or default_lot_size <= 0:
            logger.error(f"[{func_name}] Prerequisite validation failed! Invalid or missing default lot size ({default_lot_size}) for chart generation")
            raise ValueError(f"Invalid or missing default lot size ({default_lot_size}) for chart generation")

        spot_price = float(spot_price) # Ensure it's a float for calculations
        logger.debug(f"[{func_name}] Prerequisites fetched: Spot={spot_price}, DefaultLotSize={default_lot_size}")

        # --- Calculate Standard Deviation Move (Needs IV/DTE) ---
        one_std_dev_move = None
        iv_used = None
        dte_used = None
        if strategy_data: # Check if list is not empty
            try:
                # Use IV/DTE from the first leg as an approximation
                # More sophisticated: Average IV/DTE or use ATM leg if available
                first_leg = strategy_data[0]
                iv_raw = first_leg.get('iv') # Could be None or string
                dte_raw = first_leg.get('days_to_expiry') # Could be None or string

                # Safely convert IV and DTE
                iv_used = _safe_get_float({'iv': iv_raw}, 'iv')
                dte_used = _safe_get_int({'dte': dte_raw}, 'dte') # DTE should likely be int

                if iv_used is not None and dte_used is not None and iv_used > 0 and spot_price > 0:
                    # DTE can be 0, use 1 day minimum for calculation stability
                    calc_dte = max(dte_used, 1) # Use integer DTE
                    iv_decimal = iv_used / 100.0
                    # Consider business days (e.g., 252) vs calendar days (365)
                    time_fraction = calc_dte / 365.0
                    one_std_dev_move = spot_price * iv_decimal * math.sqrt(time_fraction)
                    logger.debug(f"[{func_name}] Calculated 1 StDev move: {one_std_dev_move:.2f} (using IV={iv_used}%, DTE={calc_dte})")
                else:
                     logger.warning(f"[{func_name}] Cannot calculate StDev move due to missing/invalid IV ({iv_used}) or DTE ({dte_used}) or Spot ({spot_price}) in first leg.")
            except Exception as sd_calc_err:
                 logger.error(f"[{func_name}] Error calculating standard deviation move: {sd_calc_err}", exc_info=True)
        else:
             logger.warning(f"[{func_name}] Cannot calculate StDev move as strategy_data is empty.")
        # ----------------------------------------------------------


        # 2. Calculate Payoff Data
        logger.debug(f"[{func_name}] Calculating payoff data...")
        # Adjust bounds slightly if StDev is known, otherwise use defaults
        sd_factor = 2.5 # How many SDs to extend the chart range
        if one_std_dev_move and one_std_dev_move > 0: # Ensure StDev is positive
             chart_min = spot_price - sd_factor * one_std_dev_move
             chart_max = spot_price + sd_factor * one_std_dev_move
        else:
             chart_min = spot_price * PAYOFF_LOWER_BOUND_FACTOR
             chart_max = spot_price * PAYOFF_UPPER_BOUND_FACTOR
        # Ensure bounds are sensible and don't cross
        lower_bound = max(chart_min, 0.1) # Ensure > 0
        upper_bound = max(chart_max, lower_bound + 1) # Ensure upper > lower

        price_range = np.linspace(lower_bound, upper_bound, PAYOFF_POINTS)
        total_payoff = np.zeros_like(price_range)
        processed_legs_count = 0
        logger.debug(f"[{func_name}] Price Range: {lower_bound:.2f} to {upper_bound:.2f} ({PAYOFF_POINTS} points)")


        for i, leg in enumerate(strategy_data):
            leg_desc = f"Leg {i+1}"
            try:
                logger.debug(f"[{func_name}] Processing {leg_desc}: {leg}")
                # Use safe extraction for all leg parameters
                tr_type = str(leg.get('tr_type', '')).lower()
                op_type = str(leg.get('op_type', '')).lower()
                strike = _safe_get_float(leg, 'strike')
                premium = _safe_get_float(leg, 'op_pr')
                lots = _safe_get_int(leg, 'lot')

                # Determine Lot Size safely (using default as fallback)
                raw_ls = leg.get('lot_size') # Get raw value first
                temp_ls = _safe_get_int({'ls': raw_ls}, 'ls') # Try safe conversion
                leg_lot_size = temp_ls if temp_ls is not None and temp_ls > 0 else default_lot_size
                logger.debug(f"[{func_name}] {leg_desc} Final lot size used: {leg_lot_size}")

                # Validate parameters after safe extraction
                error_msg = None
                if tr_type not in ('b','s'): error_msg = f"invalid tr_type ('{tr_type}')"
                elif op_type not in ('c','p'): error_msg = f"invalid op_type ('{op_type}')"
                elif strike is None or strike <= 0: error_msg = f"invalid strike ({strike})"
                elif premium is None or premium < 0: error_msg = f"invalid premium ({premium})"
                elif lots is None or lots <= 0: error_msg = f"invalid lots ({lots})"
                elif not isinstance(leg_lot_size, int) or leg_lot_size <= 0: error_msg = f"invalid final lot_size ({leg_lot_size})"

                if error_msg:
                     logger.error(f"[{func_name}] Validation failed for {leg_desc}: {error_msg}. Leg Data: {leg}")
                     raise ValueError(f"Error in {leg_desc} for chart: {error_msg}")

                # If validation passes, proceed with calculation
                quantity = lots * leg_lot_size
                leg_prem_tot = premium * quantity
                intrinsic_value = np.maximum(price_range - strike, 0) if op_type == 'c' else np.maximum(strike - price_range, 0)
                leg_payoff = (intrinsic_value * quantity - leg_prem_tot) if tr_type == 'b' else (leg_prem_tot - intrinsic_value * quantity)
                total_payoff += leg_payoff
                processed_legs_count += 1
                logger.debug(f"[{func_name}] {leg_desc} payoff calculated and added.")

            except (KeyError, ValueError, TypeError) as e:
                 logger.error(f"[{func_name}] Error processing {leg_desc} data for chart: {e}. Leg Data: {leg}", exc_info=False)
                 # Re-raise to stop chart generation if any leg is invalid
                 raise ValueError(f"Error in {leg_desc} for chart: {e}") from e


        if processed_legs_count == 0:
            logger.warning(f"[{func_name}] No valid legs processed for chart: {asset}.")
            # If no legs processed, it's likely due to errors raised above,
            # but add a check here just in case.
            return None
        logger.debug(f"[{func_name}] Payoff calculation complete for {processed_legs_count} legs.")
        # --- End Payoff Calculation ---

        # 3. Create Plotly Figure
        logger.debug(f"[{func_name}] Creating Plotly figure...")
        fig = go.Figure()

        # 4. Add Payoff Trace (Solid Blue Line)
        logger.debug(f"[{func_name}] Adding main payoff trace...")
        fig.add_trace(go.Scatter(
            x=price_range,
            y=total_payoff,
            mode='lines',
            name='Payoff', # Simple name, legend will be hidden
            line=dict(color='mediumblue', width=2.5), # Match previous style
            hoverinfo='x+y' # Show x (price) and y (P/L) on hover
        ))

        # 5. Add Shading for Profit/Loss Zones
        logger.debug(f"[{func_name}] Adding profit/loss zone shading...")
        # Use slightly different colors matching the target image better
        profit_color = 'rgba(144, 238, 144, 0.4)' # Light Green (semi-transparent)
        loss_color = 'rgba(255, 153, 153, 0.4)'   # Lighter Red/Coral (semi-transparent)

        # Create arrays filled up/down to the payoff line itself for better visual
        # Need the price_range array duplicated for the boundary shape
        x_fill = np.concatenate([price_range, price_range[::-1]]) # x -> x_reversed

        # Profit fill: From 0 up to positive payoff
        payoff_for_profit_fill = np.maximum(total_payoff, 0) # P/L line clipped at 0
        # Shape: (x_min -> x_max -> x_max -> x_min), (0 -> 0 -> y_max(reversed) -> y_min(reversed))
        y_profit_fill = np.concatenate([np.zeros_like(price_range), payoff_for_profit_fill[::-1]])
        fig.add_trace(go.Scatter(
            x=x_fill, y=y_profit_fill,
            fill='toself', # Fill the shape defined by the points
            mode='none', # No lines or markers for the fill trace itself
            fillcolor=profit_color,
            hoverinfo='skip' # Don't show hover info for the fill areas
        ))

        # Loss fill: From negative payoff up to 0
        payoff_for_loss_fill = np.minimum(total_payoff, 0) # P/L line clipped at 0
        # Shape: (x_min -> x_max -> x_max -> x_min), (y_min -> y_max -> 0 -> 0)
        y_loss_fill = np.concatenate([payoff_for_loss_fill, np.zeros_like(price_range)[::-1]]) # Need reversed zero array
        fig.add_trace(go.Scatter(
            x=x_fill, y=y_loss_fill,
            fill='toself', # Fill the shape defined by the points
            mode='none',
            fillcolor=loss_color,
            hoverinfo='skip'
        ))


        # 6. Add Reference Lines and Annotations
        logger.debug(f"[{func_name}] Adding reference lines & annotations...")
        # Zero P/L line (Subtle but clear)
        fig.add_hline(y=0, line=dict(color='rgba(0, 0, 0, 0.7)', width=1.0, dash='solid')) # Slightly transparent black solid line

        # Current Spot Price line & Annotation
        fig.add_vline(x=spot_price, line=dict(color='dimgrey', width=1.5, dash='dash'))
        fig.add_annotation(
            x=spot_price, y=1, yref="paper", # Position annotation near the top edge
            text=f"Spot {spot_price:.2f}", showarrow=False,
            yshift=10, # Push slightly above the top edge
            font=dict(color='dimgrey', size=11, family="Arial"),
            bgcolor="rgba(255,255,255,0.6)" # Optional: Semi-transparent background
        )

        # Standard Deviation Lines & Annotations (if calculated)
        if one_std_dev_move is not None and one_std_dev_move > 0:
            levels = [-2, -1, 1, 2]
            sig_color = 'rgba(100, 100, 100, 0.8)' # Darker grey for SD lines/text
            for level in levels:
                sd_price = spot_price + level * one_std_dev_move
                # Only plot if within the chart's x-axis range for tidiness
                if lower_bound < sd_price < upper_bound:
                    label = f"{level:+}σ" # Format as +1σ, -1σ, etc.
                    # Add the line
                    fig.add_vline(x=sd_price, line=dict(color=sig_color, width=1, dash='dot'))
                    # Add the annotation near the top
                    fig.add_annotation(
                        x=sd_price, y=1, yref="paper",
                        text=label, showarrow=False,
                        yshift=10, font=dict(color=sig_color, size=10, family="Arial"),
                        bgcolor="rgba(255,255,255,0.6)"
                    )
            logger.debug(f"[{func_name}] Added standard deviation lines.")
        else:
            logger.debug(f"[{func_name}] Standard deviation lines skipped (calculation failed or result was zero/negative).")


        # REMOVED: Strike Price lines plotting loop (Plotly doesn't need manual text placement like Matplotlib)
        # Could add vertical lines for strikes if desired, similar to Spot/SD lines

        # 7. Update Layout (Clean look, no legend)
        logger.debug(f"[{func_name}] Updating layout...")
        fig.update_layout(
            title=dict(text=f"{asset} Strategy Payoff", font=dict(size=16, family="Arial, sans-serif"), x=0.5), # Centered title
            xaxis_title="Underlying Price at Expiry",
            yaxis_title="Profit / Loss (₹)",
            hovermode="x unified", # Show hover data for all traces at a given x
            showlegend=False, # <-- HIDE LEGEND as requested
            template='plotly_white', # Clean background, enhances grid visibility
            xaxis=dict(
                gridcolor='rgba(220, 220, 220, 0.7)', # Light grey grid lines
                zeroline=False # Turn off the explicit x=0 line
            ),
            yaxis=dict(
                gridcolor='rgba(220, 220, 220, 0.7)',
                zeroline=False, # Turn off the explicit y=0 line (we added our own hline)
                tickprefix="₹" # Add currency symbol to y-axis ticks
            ),
            margin=dict(l=60, r=40, t=80, b=60), # Adjust margins for title/labels
            font=dict(family="Arial, sans-serif", size=11) # Default font
        )

        # 8. Generate HTML Content and Optionally Save File
        logger.debug(f"[{func_name}] Generating Plotly HTML content...")
        # Get HTML content as a string (suitable for embedding in web pages)
        # full_html=False gives just the <div>, include_plotlyjs='cdn' keeps it small by linking to CDN
        html_content = fig.to_html(include_plotlyjs='cdn', full_html=False, default_height='550px', default_width='100%') # Make responsive

        # Optionally save a *standalone* full HTML file for easy local viewing/debugging
        # This part is optional for the primary function of returning HTML content
        try:
             # Ensure directory exists if saving to a specific path
             # os.makedirs(os.path.dirname(output_filename_standalone), exist_ok=True) # Uncomment if saving outside current dir

             # Remove existing file first if necessary (be cautious with this in production)
             if os.path.exists(output_filename_standalone):
                  try:
                      os.remove(output_filename_standalone)
                  except OSError as rm_err:
                       logger.warning(f"[{func_name}] Could not remove existing standalone file '{output_filename_standalone}': {rm_err}")

             # Need full_html=True for the standalone file to be self-contained
             fig.write_html(output_filename_standalone, include_plotlyjs='cdn', full_html=True)
             logger.info(f"[{func_name}] Also saved standalone Plotly chart to {output_filename_standalone}")
        except Exception as write_err:
             # Log error but don't prevent returning the main HTML content if generated
             logger.error(f"[{func_name}] Error saving standalone HTML file '{output_filename_standalone}': {write_err}", exc_info=True)


        duration = time.monotonic() - start_time
        logger.info(f"[{func_name}] Plotly HTML generation finished in {duration:.3f}s.")
        return html_content # Return the HTML string for embedding

    except ValueError as val_err:
        # Log specific value errors (e.g., bad inputs, prerequisite failures) without full stack trace usually
        logger.error(f"[{func_name}] Value Error during Plotly chart generation for {asset}: {val_err}", exc_info=False)
        return None # Signal failure
    except ImportError as imp_err:
         # Catch if Plotly or Numpy are missing
         logger.critical(f"[{func_name}] Failed to import Plotly or Numpy. Ensure they are installed ('pip install plotly numpy'). Error: {imp_err}", exc_info=False)
         return None
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"[{func_name}] Unexpected error generating Plotly chart for {asset}: {e}", exc_info=True)
        return None # Signal failure


# ===============================================================
# 3. Calculate Strategy Metrics (Updated with new.py logic)
# ===============================================================
def calculate_strategy_metrics(strategy_data: List[Dict[str, Any]], spot_price: float, asset: str,) -> Optional[Dict[str, Any]]:
    """
    Calculates Profit & Loss metrics for a multi-leg options strategy,
    using theoretical P/L limits and refined breakeven calculation.
    Incorporates improved validation, Max P/L logic, and BE interval generation.
    """
    func_name = "calculate_strategy_metrics"
    logger.info(f"[{func_name}] Calculating metrics for {len(strategy_data)} leg(s), asset: {asset}")
    logger.debug(f"[{func_name}] Input strategy_data: {strategy_data}")

    # --- 1. Fetch Essential Data (Using new.py's error handling) ---
    try:
        logger.debug(f"[{func_name}] Fetching prerequisites...")
        spot_price_info = get_latest_spot_price_from_db(asset)
        spot_price = None
        if spot_price_info and 'spot_price' in spot_price_info:
            spot_price = _safe_get_float(spot_price_info, 'spot_price')
            logger.debug(f"[{func_name}] Using Spot Price from DB: {spot_price}")
        if spot_price is None: # Try cache if DB failed
            logger.debug(f"[{func_name}] Spot price from DB failed or missing key, trying cache...")
            cached_data = get_cached_option(asset)
            spot_price = _safe_get_float(cached_data.get("records", {}), "underlyingValue") if cached_data else None
            logger.debug(f"[{func_name}] Using Spot Price from Cache/Live: {spot_price}")

        if spot_price is None or not isinstance(spot_price, (int, float)) or spot_price <= 0:
             # More specific error
             raise ValueError(f"Spot price missing/invalid ({spot_price}) after DB/Cache check")
        spot_price = float(spot_price) # Ensure float

        default_lot_size = get_lot_size(asset)
        if default_lot_size is None or not isinstance(default_lot_size, int) or default_lot_size <= 0:
             # More specific error
            raise ValueError(f"Default lot size missing or invalid ({default_lot_size})")
        logger.debug(f"[{func_name}] Prerequisites OK - Spot: {spot_price}, Default Lot Size: {default_lot_size}")
    except ValueError as prereq_err: # Specific handling for ValueErrors
        logger.error(f"[{func_name}] Failed prerequisite data fetch for {asset}: {prereq_err}")
        return None
    except Exception as e: # Catch any other unexpected error during prereq fetch
        logger.error(f"[{func_name}] Unexpected error fetching prerequisites for {asset}: {e}", exc_info=True)
        return None

    # --- 2. Process Strategy Legs & Store Details ---
    logger.debug(f"[{func_name}] Processing strategy legs...")
    total_net_premium = 0.0; cost_breakdown = []; processed_legs = 0
    net_long_call_qty = 0; net_short_call_qty = 0; net_long_put_qty = 0; net_short_put_qty = 0
    payoff_at_S_equals_0 = 0.0; legs_for_payoff_calc = []
    all_strikes_list = []

    for i, leg in enumerate(strategy_data):
        leg_desc = f"Leg {i+1}"
        try:
            logger.debug(f"[{func_name}] Processing {leg_desc}: {leg}")
            tr_type = str(leg.get('tr_type', '')).lower()
            option_type = str(leg.get('op_type', '')).lower() # 'c' or 'p'
            strike = _safe_get_float(leg, 'strike'); premium = _safe_get_float(leg, 'op_pr')
            lots = _safe_get_int(leg, 'lot')

            # Determine and validate leg_lot_size (using new.py's approach)
            raw_ls = leg.get('lot_size')
            temp_ls = _safe_get_int({'ls': raw_ls}, 'ls') # Try safe conversion
            leg_lot_size = temp_ls if temp_ls is not None and temp_ls > 0 else default_lot_size
            # No need for extra check here if default_lot_size is guaranteed valid by prereq check

            logger.debug(f"[{func_name}] {leg_desc} Extracted: tr={tr_type}, op={option_type}, K={strike}, prem={premium}, lots={lots}, ls={leg_lot_size}")

            # Validate parameters robustly (using new.py's specific checks)
            error_msg = None
            if tr_type not in ('b','s'): error_msg = f"Invalid tr_type: '{tr_type}'"
            elif option_type not in ('c','p'): error_msg = f"Invalid op_type: '{option_type}'"
            elif strike is None or strike <= 0: error_msg = f"Invalid strike: {strike}"
            elif premium is None or premium < 0: error_msg = f"Invalid premium: {premium}"
            elif lots is None or lots <= 0: error_msg = f"Invalid lots: {lots}"
            elif not isinstance(leg_lot_size, int) or leg_lot_size <= 0: error_msg = f"Invalid final lot_size: {leg_lot_size}"

            if error_msg:
                # Log specific error and fail fast
                logger.error(f"[{func_name}] Validation failed for {leg_desc}: {error_msg}. Leg Data: {leg}")
                raise ValueError(f"{leg_desc} error: {error_msg}")

            logger.debug(f"[{func_name}] {leg_desc} Validation Passed.")

            quantity = lots * leg_lot_size; leg_premium_total = premium * quantity; action_verb = ""
            all_strikes_list.append(strike)
            logger.debug(f"[{func_name}] {leg_desc} Quantity: {quantity}, Leg Total Premium: {leg_premium_total:.2f}")

            # Accumulate net premium and quantities
            if tr_type == 'b': # Buy
                total_net_premium -= leg_premium_total; action_verb = "Paid"
                if option_type == 'c': net_long_call_qty += quantity
                else: net_long_put_qty += quantity
            else: # Sell
                total_net_premium += leg_premium_total; action_verb = "Received"
                if option_type == 'c': net_short_call_qty += quantity
                else: net_short_put_qty += quantity
            logger.debug(f"[{func_name}] {leg_desc} {action_verb}: NetPrem={total_net_premium:.2f}, NetLongCall={net_long_call_qty}, NetShortCall={net_short_call_qty}, NetLongPut={net_long_put_qty}, NetShortPut={net_short_put_qty}")

            # Calculate payoff component if underlying price goes to 0 (using new.py's logic)
            intrinsic_at_zero = 0.0
            if option_type == 'p': intrinsic_at_zero = strike # Max(strike - 0, 0)
            # Payoff at S=0: Buy = Intrinsic - Premium, Sell = Premium - Intrinsic
            leg_payoff_at_zero = (intrinsic_at_zero * quantity - leg_premium_total) if tr_type == 'b' else (leg_premium_total - intrinsic_at_zero * quantity)
            payoff_at_S_equals_0 += leg_payoff_at_zero
            logger.debug(f"[{func_name}] {leg_desc} Intrinsic@0={intrinsic_at_zero:.2f}, PayoffAtZero Component: {leg_payoff_at_zero:.2f}, Cumulative Payoff@0: {payoff_at_S_equals_0:.2f}")

            # Store details for cost breakdown and payoff function
            cost_bd_leg = { "leg_index": i, "action": tr_type.upper(), "type": option_type.upper(), "strike": strike, "premium_per_share": premium, "lots": lots, "lot_size": leg_lot_size, "quantity": quantity, "total_premium": round(leg_premium_total if tr_type=='s' else -leg_premium_total, 2), "effect": action_verb }
            cost_breakdown.append(cost_bd_leg)

            payoff_calc_leg = {'tr_type': tr_type, 'op_type': option_type, 'strike': strike, 'premium': premium, 'quantity': quantity}
            legs_for_payoff_calc.append(payoff_calc_leg)
            logger.debug(f"[{func_name}] {leg_desc} Data for Payoff Calc Added: {payoff_calc_leg}")

            processed_legs += 1
        except (ValueError, KeyError, TypeError) as leg_err:
            # Fail fast if any leg has invalid data for metrics calculation
            # Error logged within the validation block above if ValueError
            if not isinstance(leg_err, ValueError): # Log other types here
                 logger.error(f"[{func_name}] Type/Key Error processing metrics {leg_desc}: {leg_err}. Data: {leg}")
            return None
        except Exception as leg_exp_err: # Catch any other unexpected error during leg processing
             logger.error(f"[{func_name}] UNEXPECTED Error processing metrics {leg_desc}: {leg_exp_err}. Data: {leg}", exc_info=True)
             return None # Fail fast

    if processed_legs == 0:
        logger.error(f"[{func_name}] No valid legs processed for metrics calculation for {asset}.")
        return None
    logger.debug(f"[{func_name}] Finished processing {processed_legs} legs.")
    logger.debug(f"[{func_name}] Net Calls Qty: {net_long_call_qty-net_short_call_qty}, Net Puts Qty: {net_long_put_qty-net_short_put_qty}, Total Net Prem: {total_net_premium:.2f}, Payoff@S=0: {payoff_at_S_equals_0:.2f}")

    # --- 3. Define Payoff Function (Helper) ---
    # This function calculates the total payoff for the strategy at a given underlying price S
    def _calculate_payoff_at_price(S: float, legs: List[Dict]) -> float:
        total_pnl = 0.0
        if S < 0: S = 0 # Ensure price doesn't go below zero for calculation
        for leg_payoff in legs: # Use unique variable name
            intrinsic = 0.0; premium = leg_payoff['premium']; strike = leg_payoff['strike']
            quantity = leg_payoff['quantity']; op_type = leg_payoff['op_type']; tr_type = leg_payoff['tr_type']
            leg_prem_tot = premium * quantity

            if op_type == 'c': intrinsic = max(S - strike, 0)
            else: intrinsic = max(strike - S, 0)
            # PnL = (Expiration Value - Cost) for long, (Income - Expiration Value) for short
            pnl = (intrinsic * quantity - leg_prem_tot) if tr_type == 'b' else (leg_prem_tot - intrinsic * quantity)
            total_pnl += pnl
        return total_pnl

    # --- 4. Determine Theoretical Max Profit / Loss (Using new.py's improved logic) ---
    logger.debug(f"[{func_name}] Determining theoretical Max P/L...")
    net_calls_qty = net_long_call_qty - net_short_call_qty
    net_puts_qty = net_long_put_qty - net_short_put_qty
    all_strikes_unique_sorted = sorted(list(set(all_strikes_list))) if all_strikes_list else []
    logger.debug(f"[{func_name}] Net Call Qty: {net_calls_qty}, Net Put Qty: {net_puts_qty}")
    logger.debug(f"[{func_name}] Unique Strikes Sorted: {all_strikes_unique_sorted}")

    # Payoff function bound for root finding and checks
    payoff_func = lambda s: _calculate_payoff_at_price(s, legs_for_payoff_calc)

    # Determine Max Profit
    max_profit_val = -np.inf # Initialize for finding maximum
    # Profit is unlimited if net long calls OR net short puts (or both)
    if net_calls_qty > 0 or net_puts_qty < 0:
        max_profit_val = np.inf
        logger.debug(f"[{func_name}] Max Profit: Potentially Infinite (NetCalls={net_calls_qty}, NetPuts={net_puts_qty})")
    else:
        # Max profit is bounded, check payoff at S=0 and all strikes
        logger.debug(f"[{func_name}] Max Profit: Bounded. Checking payoff at S=0 and strikes...")
        # Start with payoff at S=0
        max_profit_val = payoff_at_S_equals_0
        logger.debug(f"[{func_name}]   Initial MaxP candidate (Payoff@0): {max_profit_val:.2f}")
        # Check payoff at each unique strike price
        for k in all_strikes_unique_sorted:
            payoff_at_k = payoff_func(k)
            logger.debug(f"[{func_name}]   Payoff@{k:.2f}: {payoff_at_k:.2f}")
            max_profit_val = max(max_profit_val, payoff_at_k)
        # !! Crucial addition from new.py !!
        # Also consider the initial net premium for strategies like short straddles/strangles
        # where max profit might be the premium itself if options expire worthless
        max_profit_val = max(max_profit_val, total_net_premium if total_net_premium > 0 else -np.inf)
        logger.debug(f"[{func_name}]   Final Bounded Max Profit: {max_profit_val:.2f}")

    # Determine Max Loss
    max_loss_val = np.inf # Initialize for finding minimum
    # Loss is unlimited if net short calls OR net long puts (or both)
    if net_calls_qty < 0 or net_puts_qty > 0:
        max_loss_val = -np.inf
        logger.debug(f"[{func_name}] Max Loss: Potentially Infinite (NetCalls={net_calls_qty}, NetPuts={net_puts_qty})")
    else:
        # Max loss is bounded, check payoff at S=0 and all strikes
        logger.debug(f"[{func_name}] Max Loss: Bounded. Checking payoff at S=0 and strikes...")
        # Start with payoff at S=0
        max_loss_val = payoff_at_S_equals_0
        logger.debug(f"[{func_name}]   Initial MaxL candidate (Payoff@0): {max_loss_val:.2f}")
        # Check payoff at each unique strike price
        for k in all_strikes_unique_sorted:
             payoff_at_k = payoff_func(k)
             logger.debug(f"[{func_name}]   Payoff@{k:.2f}: {payoff_at_k:.2f}")
             max_loss_val = min(max_loss_val, payoff_at_k)
        # !! Crucial addition from new.py !!
        # Also consider the initial net premium for strategies like long straddles/strangles
        # where max loss might be the premium paid if options expire worthless
        max_loss_val = min(max_loss_val, total_net_premium if total_net_premium < 0 else np.inf)
        logger.debug(f"[{func_name}]   Final Bounded Max Loss: {max_loss_val:.2f}")

    # --- 5. Breakeven Points (Using new.py's improved interval logic) ---
    logger.debug(f"[{func_name}] Starting breakeven search...")
    breakeven_points_found = []
    # Define points around which to search for roots: 0 and all unique strikes
    search_points = sorted(list(set([0.0] + all_strikes_list)))
    logger.debug(f"[{func_name}] Points for defining search intervals: {search_points}")

    # Define search intervals using new.py's approach
    search_intervals = []
    if not search_points: # Should not happen if legs exist, but handle defensively
         logger.warning(f"[{func_name}] No strikes found, cannot define search intervals for BE.")
    else:
        # Interval before the first strike (or starting near 0 if first strike is 0)
        first_point = search_points[0]
        if first_point > 1e-6: # Search below first strike if it's not zero
             # Start search slightly above 0, extend slightly beyond first strike
             search_intervals.append((max(0.0, first_point * 0.5), first_point * 1.05))
        elif len(search_points) > 1: # First point is 0, search between 0 and next strike
             second_point = search_points[1]
             search_intervals.append((max(0.0, second_point*0.1), second_point * 1.05))

        # Intervals between strikes (with slight overlap)
        for i in range(len(search_points) - 1):
            k1 = search_points[i]
            k2 = search_points[i+1]
            if k2 > k1 + 1e-6: # Ensure strikes are distinct
                 search_intervals.append((k1 * 0.99, k2 * 1.01)) # Overlap intervals slightly

        # Interval above the last strike
        last_strike = search_points[-1]
        # Define a reasonable upper bound for search based on spot/strikes/payoff factor
        upper_search_limit = max(last_strike * 1.5, spot_price * (PAYOFF_UPPER_BOUND_FACTOR + 0.5)) # Use constant
        search_intervals.append((last_strike * 0.99, upper_search_limit))

    logger.debug(f"[{func_name}] Potential Search Intervals: {search_intervals}")

    processed_intervals = set(); root_finder_used = "None"
    for p1_raw, p2_raw in search_intervals:
        # Ensure valid interval where p1 < p2 and p1 >= 0
        p1 = max(0.0, p1_raw); p2 = max(p1 + 1e-6, p2_raw) # Ensure p2 > p1
        if p1 >= p2: continue # Skip invalid intervals

        interval_key = (round(p1, 4), round(p2, 4))
        if interval_key in processed_intervals: continue # Avoid redundant checks
        processed_intervals.add(interval_key)
        logger.debug(f"[{func_name}] Searching BE in interval [{p1:.2f}, {p2:.2f}]")

        try:
            y1 = payoff_func(p1); y2 = payoff_func(p2)
            logger.debug(f"[{func_name}]   Payoff({p1:.2f}) = {y1:.2f}, Payoff({p2:.2f}) = {y2:.2f}")

            # Check if payoff crosses zero within the interval (sign change)
            # Use np.sign correctly: 0 is different from positive/negative
            sign1 = np.sign(y1); sign2 = np.sign(y2)
            if sign1 != sign2 and sign1 != 0 and sign2 != 0: # Must cross zero, not just touch it
                logger.debug(f"[{func_name}]   Sign change detected.")
                found_be = None
                root_finder_used = "None"
                # Try precise root finding with brentq if available
                if SCIPY_AVAILABLE and brentq is not None:
                    try:
                        # Use a small tolerance for finding root
                        be = brentq(payoff_func, p1, p2, xtol=1e-6, rtol=1e-6, maxiter=100)
                        # Check if root is reasonably positive (avoiding near-zero artifacts)
                        if be > 1e-6:
                            found_be = be; root_finder_used = "brentq"
                            logger.debug(f"[{func_name}]   Brentq result: {be:.4f}")
                        else: logger.debug(f"[{func_name}]   Brentq result near zero ({be:.4f}), ignoring.")
                    except ValueError as brentq_val_err: # Catches cases where sign isn't different at ends for brentq
                         logger.debug(f"[{func_name}]   Brentq ValueError (likely no strict sign change for solver): {brentq_val_err}")
                    except Exception as brentq_err: # Catch other convergence errors
                         logger.warning(f"[{func_name}]   Brentq failed in [{p1:.2f}, {p2:.2f}]: {brentq_err}. Will try interpolation.")

                # Fallback: Linear Interpolation if brentq failed or not available
                # Check if y values are different enough to avoid division by zero
                if found_be is None and abs(y2 - y1) > 1e-9:
                    # Formula for x-intercept of line between (p1, y1) and (p2, y2)
                    be = p1 - y1 * (p2 - p1) / (y2 - y1)
                    logger.debug(f"[{func_name}]   Interpolation attempt result: {be:.4f}")
                    # Check if interpolated point is within the interval bounds and positive
                    # Note: strict inequality for interval check might be better here
                    if ((p1 < be < p2) or (p2 < be < p1)) and be > 1e-6:
                        found_be = be; root_finder_used = "interpolation"
                        logger.debug(f"[{func_name}]   Using Interpolation result: {be:.4f}")
                    else: logger.debug(f"[{func_name}]   Interpolation result outside interval or near zero.")

                # If a valid breakeven point was found by either method
                if found_be is not None:
                     logger.debug(f"[{func_name}]   Found BE Point: {found_be:.4f} (Method: {root_finder_used})")
                     # Avoid adding duplicates very close to each other
                     is_close_to_existing = any(abs(found_be - existing_be) < 0.01 for existing_be in breakeven_points_found)
                     if not is_close_to_existing:
                          breakeven_points_found.append(found_be)
                          logger.debug(f"[{func_name}]   Added BE {found_be:.4f} to list.")
                     else:
                          logger.debug(f"[{func_name}]   BE {found_be:.4f} is too close to an existing found BE, skipping.")
            else:
                 logger.debug(f"[{func_name}]   No strict sign change detected in interval [{p1:.2f}, {p2:.2f}].")

        except Exception as search_err:
             # Catch errors during payoff calculation or checks within the loop
             logger.error(f"[{func_name}] Error during BE search interval [{p1:.2f}, {p2:.2f}]: {search_err}", exc_info=True)


    # Check strikes themselves for exact zero touch (within tolerance)
    # This catches cases where a strike is exactly a breakeven point
    zero_tolerance = 1e-4 # How close to zero counts as zero
    logger.debug(f"[{func_name}] Checking strikes for exact zero touch (tolerance={zero_tolerance})...")
    for k in all_strikes_unique_sorted:
        try:
            payoff_at_k = payoff_func(k)
            logger.debug(f"[{func_name}]   Payoff@{k:.2f}: {payoff_at_k:.4f}")
            if abs(payoff_at_k) < zero_tolerance:
                 # Avoid adding if it's very close to an already found root
                 is_close_to_existing = any(abs(k - be) < 0.01 for be in breakeven_points_found)
                 if not is_close_to_existing:
                      breakeven_points_found.append(k)
                      logger.debug(f"[{func_name}]   Found BE (strike touch): {k:.4f}, added to list.")
                 else:
                      logger.debug(f"[{func_name}]   Strike touch {k:.4f} is too close to an existing found BE, skipping.")
        except Exception as payoff_err:
             logger.error(f"[{func_name}] Error calculating payoff at strike {k} for BE check: {payoff_err}")


    # Filter, sort unique positive BE points (using new.py's strictly positive filter)
    unique_be_points_raw = sorted([p for p in list(set(breakeven_points_found)) if p > 1e-6])
    logger.debug(f"[{func_name}] Raw Unique Positive BE Points Found: {unique_be_points_raw}")

    # --- Cluster Breakeven Points ---
    logger.debug(f"[{func_name}] Clustering BE points with tolerance ratio {BREAKEVEN_CLUSTER_GAP_PCT}...")
    # Assume cluster_points helper function exists and is imported/defined elsewhere
    breakeven_points_clustered = cluster_points(unique_be_points_raw, BREAKEVEN_CLUSTER_GAP_PCT, spot_price)
    logger.debug(f"[{func_name}] Clustered BE Points: {breakeven_points_clustered}")

    # --- 6. Reward to Risk Ratio (Using new.py's string formatting logic) ---
    logger.debug(f"[{func_name}] Calculating Reward:Risk Ratio...")
    reward_to_risk_ratio_str = "N/A"; zero_threshold = 1e-9
    max_p_num = max_profit_val # Use theoretical calculated above
    max_l_num = max_loss_val   # Use theoretical calculated above
    logger.debug(f"[{func_name}] R:R Inputs - MaxP={max_p_num}, MaxL={max_l_num}")

    if np.isinf(max_l_num): # Infinite Loss
        reward_to_risk_ratio_str = "0.00" # Or "Poor" / "Negligible"
    elif np.isinf(max_p_num): # Infinite Profit
        # Check if loss is finite and non-zero
        if np.isfinite(max_l_num) and abs(max_l_num) >= zero_threshold:
             reward_to_risk_ratio_str = "∞" # Infinite reward, finite risk
        else: # Infinite profit, near-zero or zero loss
             reward_to_risk_ratio_str = "∞" # Effectively infinite R:R
    else: # Both Profit and Loss are Bounded
        max_l_num_abs = abs(max_l_num)
        if max_l_num_abs < zero_threshold: # Max loss is effectively zero
             if max_p_num > zero_threshold:
                 reward_to_risk_ratio_str = "∞" # Positive profit, zero risk
             else:
                 reward_to_risk_ratio_str = "0.00" # Zero/Neg profit, zero risk
        elif max_p_num <= zero_threshold: # Max profit is negative or zero (and loss is non-zero)
            reward_to_risk_ratio_str = "Loss" # No potential reward relative to risk
        else: # Both positive profit and positive loss (absolute)
             try:
                 ratio = max_p_num / max_l_num_abs
                 # Format to 2 decimal places
                 reward_to_risk_ratio_str = f"{ratio:.2f}"
             except ZeroDivisionError: # Should be caught by max_l_num_abs check
                 reward_to_risk_ratio_str = "∞" # Should not happen if checks above are correct

    logger.debug(f"[{func_name}] Calculated R:R String = {reward_to_risk_ratio_str}")

    # --- 7. Format Final Output ---
    # Format infinities and numbers consistently
    max_profit_str = "∞" if max_profit_val == np.inf else f"{max_profit_val:.2f}"
    max_loss_str = "-∞" if max_loss_val == -np.inf else f"{max_loss_val:.2f}"
    # reward_to_risk_ratio_str is already formatted correctly

    logger.info(f"[{func_name}] Metrics calculated. MaxP: {max_profit_str}, MaxL: {max_loss_str}, BE(Clustered): {breakeven_points_clustered}, R:R: {reward_to_risk_ratio_str}, NetPrem: {total_net_premium:.2f}")

    # Maintain the consistent return structure expected by the endpoint
    result = {
        "calculation_inputs": {
             "asset": asset,
             "spot_price_used": round(spot_price, 2),
             "default_lot_size": default_lot_size,
             "num_legs_processed": processed_legs
        },
        "metrics": {
             "max_profit": max_profit_str,
             "max_loss": max_loss_str,
             "breakeven_points": breakeven_points_clustered, # Use clustered list
             "reward_to_risk_ratio": reward_to_risk_ratio_str, # Use formatted string
             "net_premium": round(total_net_premium, 2)
        },
        "cost_breakdown_per_leg": cost_breakdown
    }
    logger.debug(f"[{func_name}] Returning result: {result}")
    return result

# ===============================================================
# 4. Calculate Option Greeks (Updated with new.py logic)
# ===============================================================
def calculate_option_greeks(
    strategy_data: List[Dict[str, Any]],
    asset: str,
    spot_price: float,
    interest_rate_pct: float = DEFAULT_INTEREST_RATE_PCT # Use global default
) -> List[Dict[str, Any]]:
    """
    Calculates per-share option Greeks for each leg using the mibian
    Black-Scholes model. Requires 'iv' and 'days_to_expiry' in strategy_data
    for each leg. Skips legs where calculation fails.
    Incorporates improved validation and logging from new.py version.
    """
    func_name = "calculate_option_greeks"
    logger.info(f"[{func_name}] Calculating PER-SHARE Greeks for {len(strategy_data)} legs, asset: {asset}, rate: {interest_rate_pct}%")
    logger.debug(f"[{func_name}] Input strategy_data: {strategy_data}")
    greeks_result_list: List[Dict[str, Any]] = []

    # --- Check if mibian is available (using new.py's check) ---
    if not MIBIAN_AVAILABLE or mibian is None:
        logger.error(f"[{func_name}] Mibian library not available. Cannot calculate Greeks.")
        return [] # Return empty list if library missing

    # --- 1. Fetch Spot Price ---
    spot_price: Optional[float] = None
    try:
        logger.debug(f"[{func_name}] Fetching spot price...")
        spot_price_info = get_latest_spot_price_from_db(asset)
        if spot_price_info and 'spot_price' in spot_price_info:
            spot_price = _safe_get_float(spot_price_info, 'spot_price')
            logger.debug(f"[{func_name}] Using spot price from DB {spot_price}")
        if spot_price is None: # Try cache if DB failed
            logger.debug(f"[{func_name}] Spot price from DB failed or missing key, trying cache...")
            cached_data = get_cached_option(asset)
            spot_price = _safe_get_float(cached_data.get("records", {}), "underlyingValue") if cached_data else None
            logger.debug(f"[{func_name}] Using spot price from Cache/Live {spot_price}")

        if spot_price is None or not isinstance(spot_price, (int, float)) or spot_price <= 0:
             # Use new.py's more specific error log, keep return [] behaviour
            logger.error(f"[{func_name}] Spot price missing/invalid ({spot_price}) for greeks calculation after DB/Cache check. Cannot proceed.")
            return []
        spot_price = float(spot_price) # Ensure float
        logger.debug(f"[{func_name}] Using spot price {spot_price} for asset {asset}")
    except Exception as spot_err:
        logger.error(f"[{func_name}] Unexpected error fetching spot price for {asset}: {spot_err}", exc_info=True)
        return [] # Return empty list if spot price fails catastrophically

    # --- 2. Process Each Leg ---
    logger.debug(f"[{func_name}] Processing legs for Greeks...")
    processed_count = 0
    skipped_count = 0
    for i, leg_data in enumerate(strategy_data):
        leg_desc = f"Leg {i+1}"
        try:
            logger.debug(f"[{func_name}] Processing {leg_desc}: {leg_data}")
            # --- Validate and Extract Leg Data Safely ---
            if not isinstance(leg_data, dict):
                logger.warning(f"[{func_name}] Skipping {leg_desc}: leg data is not a dictionary.")
                skipped_count += 1
                continue

            strike_price = _safe_get_float(leg_data, 'strike')
            # DTE must be present from preparation step
            days_to_expiry = _safe_get_int(leg_data, 'days_to_expiry')
            # IV must be present from preparation step
            implied_vol_pct = _safe_get_float(leg_data, 'iv')
            option_type_flag = str(leg_data.get('op_type', '')).lower()
            transaction_type = str(leg_data.get('tr_type', '')).lower()
            lots = _safe_get_int(leg_data, 'lot') # Keep for context log
            lot_size = _safe_get_int(leg_data, 'lot_size') # Keep for context log

            logger.debug(f"[{func_name}] {leg_desc} Extracted: K={strike_price}, DTE={days_to_expiry}, IV={implied_vol_pct}%, Type={option_type_flag}, Tr={transaction_type}, Lots={lots}, Size={lot_size}")

            # --- Input Validation (using new.py's more detailed block) ---
            error_msg = None
            if strike_price is None or strike_price <= 0: error_msg=f"Missing/invalid 'strike' ({strike_price})"
            # DTE can be 0, but not negative or missing
            elif days_to_expiry is None or days_to_expiry < 0: error_msg=f"Missing/invalid 'days_to_expiry' ({days_to_expiry})"
            # IV can be 0 (placeholder), but not negative or missing
            elif implied_vol_pct is None or implied_vol_pct < 0: error_msg=f"Missing/invalid 'iv' ({implied_vol_pct})"
            elif option_type_flag not in ['c', 'p']: error_msg=f"Invalid 'op_type': {option_type_flag}"
            elif transaction_type not in ['b', 's']: error_msg=f"Invalid 'tr_type': {transaction_type}"
            # Basic check for context fields (if deemed necessary here)
            elif lots is None or lot_size is None: error_msg="Missing 'lot' or 'lot_size' in input data"

            if error_msg:
                 logger.warning(f"[{func_name}] Skipping {leg_desc} due to invalid input: {error_msg}. Data: {leg_data}")
                 skipped_count += 1
                 continue # Skip this leg

            # Handle non-positive IV specifically - calculation invalid (from new.py)
            if implied_vol_pct <= 1e-6: # Use tolerance for near-zero
                 logger.warning(f"[{func_name}] Skipping {leg_desc} due to zero or near-zero IV ({implied_vol_pct}). Cannot calculate Greeks. Data: {leg_data}")
                 skipped_count += 1
                 continue # Skip leg if IV is effectively zero or negative

            # --- Mibian Calculation ---
            mibian_dte = 0.0 # Initialize
            # Handle DTE=0 using floor (explicit check from new.py for clarity)
            if days_to_expiry == 0:
                logger.warning(f"[{func_name}] {leg_desc} DTE is 0. Using small floor (0.0001) for Mibian. Results may be unreliable or calculation might fail.")
                mibian_dte = 0.0001 # Use small positive floor for DTE
            else:
                mibian_dte = float(days_to_expiry) # Ensure float

            # Mibian expects interest rate as percentage, volatility as percentage
            mibian_inputs = [spot_price, strike_price, interest_rate_pct, mibian_dte]
            volatility_input = implied_vol_pct # Already checked > 0
            logger.debug(f"[{func_name}] {leg_desc} Mibian Inputs: S={spot_price}, K={strike_price}, R={interest_rate_pct}, T={mibian_dte}, IV={volatility_input}")

            # Initialize greeks to NaN to indicate failure if calculation fails
            delta, gamma, theta, vega, rho = np.nan, np.nan, np.nan, np.nan, np.nan

            try:
                # Calculate BS model once
                bs_model = mibian.BS(mibian_inputs, volatility=volatility_input)
                logger.debug(f"[{func_name}] {leg_desc} Mibian model calculated.")

                # Extract Greeks based on type
                if option_type_flag == 'c':
                    delta = bs_model.callDelta
                    theta = bs_model.callTheta
                    rho = bs_model.callRho
                else: # Put
                    delta = bs_model.putDelta
                    theta = bs_model.putTheta
                    rho = bs_model.putRho

                # Gamma and Vega are the same for calls and puts
                gamma = bs_model.gamma
                vega = bs_model.vega
                logger.debug(f"[{func_name}] {leg_desc} Raw Greeks: Delta={delta:.4f}, Gamma={gamma:.4f}, Theta={theta:.4f}, Vega={vega:.4f}, Rho={rho:.4f}")

            except OverflowError as math_err:
                 # Log warning and continue (skip leg's greeks)
                 logger.warning(f"[{func_name}] Mibian math error (Overflow) for {leg_desc} (asset {asset}, K={strike_price}, DTE={mibian_dte}, IV={volatility_input}): {math_err}. Skipping greeks for this leg.")
                 skipped_count += 1
                 continue # Skip this leg
            except Exception as mibian_err:
                 # Log general Mibian error and continue (skip leg's greeks)
                 logger.error(f"[{func_name}] Mibian calculation error for {leg_desc} (asset {asset}, K={strike_price}, DTE={mibian_dte}, IV={volatility_input}): {mibian_err}. Skipping greeks for this leg.", exc_info=True)
                 skipped_count += 1
                 continue # Skip this leg

            # --- Check for non-finite results immediately after calculation ---
            raw_greeks_tuple = (delta, gamma, theta, vega, rho)
            if any(not np.isfinite(v) for v in raw_greeks_tuple):
                 logger.warning(f"[{func_name}] Skipping {leg_desc} for {asset} due to non-finite raw Greek result. Raw Greeks: {raw_greeks_tuple}")
                 skipped_count += 1
                 continue # Skip this leg

            # --- Adjust Sign for Short Positions ---
            sign_multiplier = -1.0 if transaction_type == 's' else 1.0
            logger.debug(f"[{func_name}] {leg_desc} Sign Multiplier (based on tr_type='{transaction_type}'): {sign_multiplier}")

            # Apply sign multiplier
            delta *= sign_multiplier
            gamma *= sign_multiplier
            theta *= sign_multiplier
            vega *= sign_multiplier
            rho *= sign_multiplier
            logger.debug(f"[{func_name}] {leg_desc} Greeks (After Sign Adjustment): Delta={delta:.4f}, Gamma={gamma:.4f}, Theta={theta:.4f}, Vega={vega:.4f}, Rho={rho:.4f}")

            # --- Store PER-SHARE Greeks (Rounded) ---
            calculated_greeks = {
                'delta': round(delta, 4),
                'gamma': round(gamma, 4),
                'theta': round(theta, 4), # Per day
                'vega': round(vega, 4),   # Per 1% IV change
                'rho': round(rho, 4)      # Per 1% rate change
            }

            # Final check for non-finite values after rounding/sign adjustment (from new.py)
            if any(not np.isfinite(v) for v in calculated_greeks.values()):
                logger.warning(f"[{func_name}] Skipping {leg_desc} for {asset} due to non-finite Greek result *after* sign adjustment/rounding. Greeks: {calculated_greeks}")
                skipped_count += 1
                continue # Skip this leg

            # Append results for this leg
            input_data_log = { # Log key inputs for debugging
                 'strike': strike_price,
                 'dte_input': days_to_expiry, # Original DTE from input
                 'iv_input': implied_vol_pct, # Original IV from input
                 'op_type': option_type_flag,
                 'tr_type': transaction_type,
                 'spot_used': spot_price,
                 'rate_used': interest_rate_pct,
                 'mibian_dte_used': mibian_dte # Include the DTE used by Mibian (from new.py)
            }
            leg_result = {
                'leg_index': i,
                'input_data': input_data_log,
                'calculated_greeks_per_share': calculated_greeks
            }
            greeks_result_list.append(leg_result)
            processed_count += 1
            logger.debug(f"[{func_name}] {leg_desc} Greek result appended: {leg_result}")

        except (ValueError, KeyError, TypeError) as validation_err:
            # Log expected validation errors as warnings and continue
            logger.warning(f"[{func_name}] Skipping Greek calculation for {leg_desc} due to validation/type error: {validation_err}. Leg data snapshot: strike={leg_data.get('strike')}, dte={leg_data.get('days_to_expiry')}, iv={leg_data.get('iv')}")
            skipped_count += 1
            continue # Skip to the next leg
        except Exception as e:
            # Log unexpected errors more severely and continue
            logger.error(f"[{func_name}] Unexpected error calculating Greeks for {leg_desc}: {e}. Leg data: {leg_data}", exc_info=True)
            skipped_count += 1
            continue # Skip to the next leg

    # Use new.py's more informative final log message
    logger.info(f"[{func_name}] Finished calculating PER-SHARE Greeks. Processed: {processed_count}, Skipped: {skipped_count} (out of {len(strategy_data)} legs initially).")
    logger.debug(f"[{func_name}] Returning greeks list: {greeks_result_list}")
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
    # ----- DEBUG LOG -----
    logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Starting fetch.")

    info = None
    h1d, h5d, h50d, h200d = None, None, None, None
    ticker_obj = None # Variable for ticker

    try:
        loop = asyncio.get_running_loop()
        # --- Step 1: Get Ticker object ---
        try:
             # Run synchronous yf.Ticker in executor
             ticker_obj = await loop.run_in_executor(None, yf.Ticker, stock_symbol)
             if not ticker_obj:
                 logger.error(f"[{stock_symbol}] fetch_stock_data_async: yf.Ticker returned None or invalid object.")
                 raise ValueError("yf.Ticker returned None or invalid object")
             logger.debug(f"[{stock_symbol}] fetch_stock_data_async: yf.Ticker object CREATED successfully.")
        except Exception as ticker_err:
             logger.error(f"[{stock_symbol}] fetch_stock_data_async: Failed to create yf.Ticker object: {ticker_err}", exc_info=False)
             return None

        # --- Step 2: Fetch info (best effort) ---
        try:
            logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Attempting to fetch ticker.info...")
            info = await loop.run_in_executor(None, getattr, ticker_obj, 'info')

            if not info:
                 logger.debug(f"[{stock_symbol}] fetch_stock_data_async: ticker.info dictionary is EMPTY.")
            else:
                 logger.debug(f"[{stock_symbol}] fetch_stock_data_async: ticker.info fetched. Type: {type(info)}. Keys: {list(info.keys()) if isinstance(info, dict) else 'Not a dict'}")
                 if isinstance(info, dict):
                     price_fields_in_info = {k: info.get(k) for k in ["currentPrice", "regularMarketPrice", "bid", "ask", "previousClose", "open"]} # Added 'open' just in case
                     logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Price-related fields in ticker.info: {price_fields_in_info}")
        except json.JSONDecodeError as json_err:
             logger.error(f"[{stock_symbol}] fetch_stock_data_async: JSONDecodeError fetching info: {json_err}. Symbol likely invalid or delisted.", exc_info=False)
             return None
        except Exception as info_err:
             logger.warning(f"[{stock_symbol}] fetch_stock_data_async: Non-critical error fetching 'info': {info_err}", exc_info=False)
             info = {} # Ensure info is an empty dict if fetch fails

        # --- Step 3: Fetch history concurrently (best effort) ---
        try:
            logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Attempting history fetches...")
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

            logger.debug(f"[{stock_symbol}] fetch_stock_data_async: History fetch results - 1d: {'OK' if h1d is not None else 'Fail/Empty'}, 5d: {'OK' if h5d is not None else 'Fail/Empty'}, 50d: {'OK' if h50d is not None else 'Fail/Empty'}, 200d: {'OK' if h200d is not None else 'Fail/Empty'}")
            if h1d is not None and 'Close' in h1d.columns:
                logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Latest Close from 1d history: {h1d['Close'].iloc[-1]}")
        except Exception as hist_err:
             logger.warning(f"[{stock_symbol}] fetch_stock_data_async: Unexpected error during history fetch: {hist_err}", exc_info=False)

        # --- Step 4: Determine Price (CRITICAL) ---
        # Refined logic based on check_yf.py output for ^NSEI
        cp = None
        price_source = "None"

        if isinstance(info, dict) and info: # Check if info is a non-empty dictionary
            logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Attempting price extraction from info dict...")
            # Check fields in preferred order
            if info.get("regularMarketPrice") is not None: # Check this BEFORE currentPrice for indices
                cp = info.get("regularMarketPrice")
                price_source = "info.regularMarketPrice"
            elif info.get("currentPrice") is not None:
                cp = info.get("currentPrice")
                price_source = "info.currentPrice"
            elif info.get("bid") is not None and info.get("bid") > 0: # Ignore bid/ask if 0
                 cp = info.get("bid")
                 price_source = "info.bid"
            elif info.get("ask") is not None and info.get("ask") > 0:
                 cp = info.get("ask")
                 price_source = "info.ask"
            elif info.get("open") is not None: # Try 'open' as another fallback
                 cp = info.get("open")
                 price_source = "info.open"
            elif info.get("previousClose") is not None:
                cp = info.get("previousClose")
                price_source = "info.previousClose"

            logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Price from info: {cp} (Source: {price_source})")
        else:
             logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Skipping price extraction from info (info is not a dict or is empty).")

        # Fallback to history if price not found in info
        if cp is None:
             logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Price not found in info, attempting fallback to history...")
             if h1d is not None and 'Close' in h1d.columns and not h1d.empty:
                 try:
                     cp = h1d["Close"].iloc[-1]
                     # Basic validation: ensure it's likely a number
                     if isinstance(cp, (int, float, np.number)) and np.isfinite(cp):
                        price_source = "history.1d.Close"
                        logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Using 1d history close price: {cp}")
                     else:
                         logger.warning(f"[{stock_symbol}] fetch_stock_data_async: 1d history Close value is invalid: {cp}. Resetting cp.")
                         cp = None # Reset cp if value is not valid
                 except Exception as hist_ex:
                     logger.warning(f"[{stock_symbol}] fetch_stock_data_async: Error accessing 1d history Close: {hist_ex}")
                     cp = None # Error accessing data

             elif h5d is not None and 'Close' in h5d.columns and not h5d.empty: # Further fallback to 5d
                  try:
                     cp = h5d["Close"].iloc[-1]
                     if isinstance(cp, (int, float, np.number)) and np.isfinite(cp):
                        price_source = "history.5d.Close"
                        logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Using 5d history close price: {cp}")
                     else:
                         logger.warning(f"[{stock_symbol}] fetch_stock_data_async: 5d history Close value is invalid: {cp}. Resetting cp.")
                         cp = None
                  except Exception as hist_ex:
                     logger.warning(f"[{stock_symbol}] fetch_stock_data_async: Error accessing 5d history Close: {hist_ex}")
                     cp = None

             if cp is None: # Check if fallback failed
                 logger.debug(f"[{stock_symbol}] fetch_stock_data_async: History fallback also failed to find a valid price.")

        # Final check: If NO price found after all attempts
        if cp is None:
             logger.error(f"[{stock_symbol}] fetch_stock_data_async: CRITICAL - Could not determine valid price from info OR history. Returning None.")
             return None # CRITICAL FAILURE: Can't proceed without price
        # Validate the type of the found price before proceeding
        elif not isinstance(cp, (int, float, np.number)) or not np.isfinite(cp):
            logger.error(f"[{stock_symbol}] fetch_stock_data_async: CRITICAL - Determined price is not a valid finite number: {cp} (Type: {type(cp)}). Returning None.")
            return None

        # ----- INFO LOG -----
        logger.info(f"[{stock_symbol}] fetch_stock_data_async: Successfully determined price: {cp} (Source: {price_source})")

        # --- Step 5: Get other fields (handle missing gracefully) ---
        # (Keep Step 5 and 6 as they were in the previous debug version)
        logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Extracting other fields...")
        safe_info = info if isinstance(info, dict) else {}
        vol = safe_info.get("volume")
        if vol is None and h1d is not None and 'Volume' in h1d.columns and not h1d.empty:
             try: # Add try-except for safety
                 vol = h1d["Volume"].iloc[-1]
                 logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Used 1d history volume: {vol}")
             except Exception as vol_ex:
                 logger.warning(f"[{stock_symbol}] fetch_stock_data_async: Error accessing 1d history Volume: {vol_ex}")
                 vol = None

        ma50 = None
        if h50d is not None and 'Close' in h50d.columns:
             try: ma50 = h50d["Close"].mean()
             except Exception as ma_ex: logger.warning(f"[{stock_symbol}] fetch_stock_data_async: Error calculating MA50: {ma_ex}")
        ma200 = None
        if h200d is not None and 'Close' in h200d.columns:
             try: ma200 = h200d["Close"].mean()
             except Exception as ma_ex: logger.warning(f"[{stock_symbol}] fetch_stock_data_async: Error calculating MA200: {ma_ex}")
        logger.debug(f"[{stock_symbol}] fetch_stock_data_async: MA50: {ma50}, MA200: {ma200}")

        mc = safe_info.get("marketCap")
        pe = safe_info.get("trailingPE")
        eps = safe_info.get("trailingEps")
        sec = safe_info.get("sector")
        ind = safe_info.get("industry")
        logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Fundamentals - MC: {mc}, P/E: {pe}, EPS: {eps}, Sector: {sec}, Industry: {ind}")

        # --- Step 6: Construct final data dictionary ---
        data = {
            "current_price": float(cp), # Convert numpy types to float for JSON compatibility
            "volume": int(vol) if vol is not None and np.isfinite(vol) else None, # Convert to int if possible
            "moving_avg_50": float(ma50) if ma50 is not None and np.isfinite(ma50) else None,
            "moving_avg_200": float(ma200) if ma200 is not None and np.isfinite(ma200) else None,
            "market_cap": mc,
            "pe_ratio": pe,
            "eps": eps,
            "sector": sec,
            "industry": ind
        }

        stock_data_cache[cache_key] = data
        logger.debug(f"[{stock_symbol}] fetch_stock_data_async: Successfully processed data. Returning dict: {data}")
        return data

    except Exception as e:
        logger.error(f"[{stock_symbol}] fetch_stock_data_async: UNEXPECTED error during overall process: {e}", exc_info=True)
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
    """
    Queries the DB for the latest spot price updated within max_age_minutes.
    **NOTE:** Uses assumed column names based on previous errors.
            Verify 'spot_price_column', 'timestamp_column', 'asset_column',
            and 'table_name' against your actual database schema.
    """
    func_name = "get_latest_spot_price_from_db"

    # --- !! VERIFY THESE NAMES AGAINST YOUR DATABASE !! ---
    spot_price_column = "spot_price"       # ASSUMED actual column name for the spot price
    timestamp_column = "updated_at"        # ASSUMED actual column name for the timestamp
    asset_column = "asset_name"            # Keep 'asset_name' as used before, verify if needed
    table_name = "option_data.assets"      # Keep table name as used before, verify if needed
    # --- End Verification Section ---

    # Construct SQL query using the (verified) column names
    sql = f"""
        SELECT {spot_price_column}, {timestamp_column}
        FROM {table_name}
        WHERE {asset_column} = %s
          AND {timestamp_column} >= NOW() - INTERVAL %s MINUTE
        ORDER BY {timestamp_column} DESC
        LIMIT 1
    """
    logger.debug(f"[{func_name}] Executing SQL: {sql} with params: ({asset}, {max_age_minutes})")

    try:
        # Assuming get_db_connection is a context manager providing connection and cleanup
        with get_db_connection() as conn:
            # Use dictionary=True to access results by column name
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(sql, (asset, max_age_minutes))
                result = cursor.fetchone() # Fetch the latest row

                # Check if a result was found and the assumed spot price column exists and is not None
                if result and result.get(spot_price_column) is not None:
                    # Use safe conversion for the spot price using the correct column name
                    price = _safe_get_float(result, spot_price_column)

                    # Get the timestamp using the correct column name
                    ts = result.get(timestamp_column)

                    # Safely format the timestamp
                    ts_str = None
                    if isinstance(ts, datetime):
                        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                    elif ts is not None:
                        ts_str = str(ts) # Fallback to string conversion

                    # Log and return if price is valid
                    if price is not None:
                        logger.debug(f"[{func_name}] Found recent spot price in DB for {asset}: {price} at {ts_str}")
                        # Return dictionary with standard keys expected by other functions
                        return {"spot_price": price, "timestamp": ts_str}
                    else:
                        logger.warning(f"[{func_name}] Found row in DB for {asset} but failed to parse price from column '{spot_price_column}'. Result: {result}")
                        return None
                else:
                    # Log if no recent data found
                    logger.debug(f"[{func_name}] No recent spot price (<{max_age_minutes}min) found in DB for {asset}. Result: {result}")
                    return None
    except (ConnectionError, mysql.connector.Error) as db_err:
        # Log specific database connection/query errors
        logger.error(f"[{func_name}] DB Error fetching latest spot price for {asset}: {db_err}")
        # Propagate the error info upwards for better context if needed by adding 'raise' or returning specific error object
        # For now, returning None as per original logic
        return None
    except Exception as e:
        # Log any other unexpected errors during the process
        logger.error(f"[{func_name}] Unexpected error fetching spot price from DB for {asset}: {e}", exc_info=True)
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
        "success": success_status, "message": message,
        "metrics": results.get("metrics"), "charges": results.get("tax"),
        "greeks": results.get("greeks", []), "chart_html_content": results.get("chart_html")
    }
    logger.info(f"[{func_name}] Returning response. Success: {success_status}, Chart: {chart_generated}, Tax: {tax_calculated}, Greeks: {greeks_calculated}")
    return final_response




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