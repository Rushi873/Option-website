import mysql.connector
from .database import get_db_connection
from flask import jsonify

# Save new strategy
def save_strategy(user_id, strategy_name, positions):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO strategies (user_id, strategy_name) VALUES (%s, %s)", (user_id, strategy_name))
        strategy_id = cursor.lastrowid

        for pos in positions:
            cursor.execute("""
                INSERT INTO strategy_positions (strategy_id, strike_price, expiry_date, option_type, lots, entry_price)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (strategy_id, pos["strike_price"], pos["expiry_date"], pos["option_type"], pos["lots"], pos["entry_price"]))

        conn.commit()
        return jsonify({"status": "success", "message": "Strategy saved successfully!"})

    except mysql.connector.Error as err:
        return jsonify({"status": "error", "message": str(err)})

    finally:
        cursor.close()
        conn.close()

# Retrieve strategies for a user
def get_user_strategies(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT * FROM strategies WHERE user_id = %s", (user_id,))
        strategies = cursor.fetchall()
        return jsonify({"status": "success", "strategies": strategies})

    except mysql.connector.Error as err:
        return jsonify({"status": "error", "message": str(err)})

    finally:
        cursor.close()
        conn.close()
