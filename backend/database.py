from dotenv import load_dotenv

import mysql.connector
from config import DB_CONFIG

# Load from .env file (if using one)
load_dotenv()

password = os.getenv("MYSQL_PASSWORD")

def get_db_connection():
    DB_CONFIG["password"] = password
    return mysql.connector.connect(**DB_CONFIG)
