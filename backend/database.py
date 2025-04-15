import os
import logging
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import pooling
from contextlib2 import contextmanager
from .config import DB_CONFIG # Assuming DB_CONFIG is defined in config.py

# --- Logging Setup (can share logger instance if configured globally) ---
logger = logging.getLogger(__name__) # Get logger instance

# --- Global variable for the connection pool ---
cnxpool = None

# --- Initialization Function ---
def initialize_database_pool():
    global cnxpool
    if cnxpool:
        logger.warning("Database pool already initialized.")
        return

    logger.info("Initializing database connection pool...")
    load_dotenv() # Load .env file

    password = os.getenv("MYSQL_PASSWORD")
    if not password:
        logger.error("MYSQL_PASSWORD environment variable not set.")
        raise ValueError("MYSQL_PASSWORD environment variable is required but not set.")

    pool_config = DB_CONFIG.copy()
    pool_config["password"] = password

    try:
        cnxpool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="myapp_pool",
            pool_size=5, # Adjust as needed
            **pool_config
        )
        logger.info(f"Database connection pool '{cnxpool.pool_name}' created successfully with size {cnxpool.pool_size}.")
    except mysql.connector.Error as err:
        logger.error(f"FATAL: Failed to create database connection pool: {err}")
        raise RuntimeError(f"Database pool creation failed: {err}") from err

# --- Efficient Connection Function using the Pool ---
@contextmanager
def get_db_connection():
    global cnxpool
    if not cnxpool:
        logger.error("Database pool is not initialized. Call initialize_database_pool() first.")
        raise RuntimeError("Database connection pool is not available.")

    conn = None
    try:
        conn = cnxpool.get_connection()
        logger.debug(f"Acquired DB connection from pool '{cnxpool.pool_name}'.")
        yield conn
    except mysql.connector.Error as err:
        logger.error(f"Error obtaining connection from pool '{cnxpool.pool_name}': {err}")
        raise ConnectionError(f"Failed to get database connection from pool: {err}") from err
    finally:
        if conn and conn.is_connected():
            conn.close()
            logger.debug(f"Returned DB connection to pool '{cnxpool.pool_name}'.")
