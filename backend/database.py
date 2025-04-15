import os
import logging
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import pooling
from contextlib2 import contextmanager # Use standard contextlib

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Global variable for the connection pool ---
cnxpool = None

# --- Initialization Function ---
def initialize_database_pool():
    global cnxpool
    if cnxpool:
        logger.info("Database pool already initialized.") # Changed to INFO
        return

    logger.info("Initializing database connection pool...")
    # load_dotenv() # Optional: Keep if you NEED .env for local runs fallback,
                   # but Railway env vars should take precedence. Load once at app start if possible.

    # --- Get ALL credentials from environment variables ---
    db_host = os.getenv("MYSQLHOST")
    db_port_str = os.getenv("MYSQLPORT")
    db_user = os.getenv("MYSQLUSER")
    db_password = os.getenv("MYSQLPASSWORD") # Use the correct variable name from Railway screenshot
    db_name = os.getenv("MYSQLDATABASE")

    # --- Validate that all necessary variables are set ---
    missing_vars = []
    if not db_host: missing_vars.append("MYSQLHOST")
    if not db_port_str: missing_vars.append("MYSQLPORT")
    if not db_user: missing_vars.append("MYSQLUSER")
    if not db_password: missing_vars.append("MYSQLPASSWORD") # Check this one too
    if not db_name: missing_vars.append("MYSQLDATABASE")

    if missing_vars:
        error_message = f"CRITICAL: Missing database environment variables: {', '.join(missing_vars)}"
        logger.error(error_message)
        # Raise an error to prevent startup with invalid config
        raise ValueError(error_message)

    try:
        # --- Construct config SOLELY from environment variables ---
        pool_config = {
            "host": db_host,
            "port": int(db_port_str), # Convert port to integer
            "user": db_user,
            "password": db_password,
            "database": db_name,
            # Add other connection options if needed (e.g., ssl)
        }
        logger.info(f"Attempting DB pool creation with config: host={pool_config['host']}, port={pool_config['port']}, user={pool_config['user']}, database={pool_config['database']}")

        cnxpool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="myapp_pool",
            pool_size=5, # Adjust as needed
            **pool_config
        )
        logger.info(f"Database connection pool '{cnxpool.pool_name}' created successfully.")

        # Optional: Test connection immediately after pool creation
        try:
            logger.info("Attempting to get test connection from pool...")
            conn_test = cnxpool.get_connection()
            logger.info(f"Successfully got test connection to DB: {conn_test.database}")
            conn_test.close()
            logger.info("Test connection closed successfully.")
        except mysql.connector.Error as test_err:
            logger.error(f"Failed pool test connection: {test_err}")
            # Optional: You might want to destroy the pool or raise error if test fails
            # cnxpool = None # Or handle differently
            raise RuntimeError(f"Pool created but test connection failed: {test_err}") from test_err

    except ValueError as verr: # Catch potential int conversion error for port
         logger.error(f"Database configuration error (e.g., invalid port): {verr}", exc_info=True)
         raise RuntimeError(f"Invalid database configuration: {verr}") from verr
    except mysql.connector.Error as err:
        logger.error(f"FATAL: Failed to create database connection pool: {err}", exc_info=True)
        # Ensure pool is None if creation fails partway
        cnxpool = None
        raise RuntimeError(f"Database pool creation failed: {err}") from err
    except Exception as e:
         logger.error(f"FATAL: Unexpected error during database pool initialization: {e}", exc_info=True)
         cnxpool = None
         raise RuntimeError(f"Unexpected error initializing DB pool: {e}") from e

# --- Efficient Connection Function using the Pool ---
@contextmanager
def get_db_connection():
    global cnxpool
    if not cnxpool:
        logger.error("Database pool is not initialized. Call initialize_database_pool() first.")
        # Raising a specific error helps diagnose startup order issues
        raise ConnectionError("Database pool is not available.")

    conn = None
    try:
        # logger.debug(f"Attempting to acquire DB connection from pool '{cnxpool.pool_name}'...")
        conn = cnxpool.get_connection()
        # logger.debug(f"Acquired DB connection ID {conn.connection_id}.") # Debug level logging
        yield conn
    except mysql.connector.errors.PoolError as pool_err:
         logger.error(f"Pooling Error: Failed to get connection from pool '{cnxpool.pool_name}': {pool_err}", exc_info=True)
         # Specific error for pool issues (e.g., pool exhausted)
         raise ConnectionError(f"Could not get DB connection from pool: {pool_err}") from pool_err
    except mysql.connector.Error as err:
        # Catch other potential mysql errors during connection acquisition
        logger.error(f"Database Error: Failed to get connection from pool '{cnxpool.pool_name}': {err}", exc_info=True)
        raise ConnectionError(f"Failed to get database connection: {err}") from err
    except Exception as e:
         # Catch any other unexpected errors
         logger.error(f"Unexpected error in get_db_connection: {e}", exc_info=True)
         raise ConnectionError(f"Unexpected error getting DB connection: {e}") from e
    finally:
        if conn:
            if conn.is_connected():
                # logger.debug(f"Returning DB connection ID {conn.connection_id} to pool '{cnxpool.pool_name}'.")
                conn.close()
            else:
                 # Log if connection was already closed or lost before returning
                 logger.warning(f"DB connection ID {conn.connection_id} was not connected when returning to pool.")


print("--- backend/database.py imported/reloaded ---") # Changed message slightly




# For local hosting
# import os
# import logging
# from dotenv import load_dotenv
# import mysql.connector
# from mysql.connector import pooling
# from contextlib2 import contextmanager
# from .config import DB_CONFIG # Assuming DB_CONFIG is defined in config.py

# # --- Logging Setup (can share logger instance if configured globally) ---
# logger = logging.getLogger(__name__) # Get logger instance

# # --- Global variable for the connection pool ---
# cnxpool = None

# # --- Initialization Function ---
# def initialize_database_pool():
#     global cnxpool
#     if cnxpool:
#         logger.warning("Database pool already initialized.")
#         return

#     logger.info("Initializing database connection pool...")
#     load_dotenv() # Load .env file

#     password = os.getenv("MYSQL_PASSWORD")
#     if not password:
#         logger.error("MYSQL_PASSWORD environment variable not set.")
#         raise ValueError("MYSQL_PASSWORD environment variable is required but not set.")

#     pool_config = DB_CONFIG.copy()
#     pool_config["password"] = password

#     try:
#         cnxpool = mysql.connector.pooling.MySQLConnectionPool(
#             pool_name="myapp_pool",
#             pool_size=5, # Adjust as needed
#             **pool_config
#         )
#         logger.info(f"Database connection pool '{cnxpool.pool_name}' created successfully with size {cnxpool.pool_size}.")
#     except mysql.connector.Error as err:
#         logger.error(f"FATAL: Failed to create database connection pool: {err}")
#         raise RuntimeError(f"Database pool creation failed: {err}") from err

# # --- Efficient Connection Function using the Pool ---
# @contextmanager
# def get_db_connection():
#     global cnxpool
#     if not cnxpool:
#         logger.error("Database pool is not initialized. Call initialize_database_pool() first.")
#         raise RuntimeError("Database connection pool is not available.")

#     conn = None
#     try:
#         conn = cnxpool.get_connection()
#         logger.debug(f"Acquired DB connection from pool '{cnxpool.pool_name}'.")
#         yield conn
#     except mysql.connector.Error as err:
#         logger.error(f"Error obtaining connection from pool '{cnxpool.pool_name}': {err}")
#         raise ConnectionError(f"Failed to get database connection from pool: {err}") from err
#     finally:
#         if conn and conn.is_connected():
#             conn.close()
#             logger.debug(f"Returned DB connection to pool '{cnxpool.pool_name}'.")

# print("--- backend/database.py executed ---")
