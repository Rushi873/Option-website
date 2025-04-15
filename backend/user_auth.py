import bcrypt
import mysql.connector
from .database import get_db_connection
from flask import jsonify

# Register user
def register_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
        conn.commit()
        return jsonify({"status": "success", "message": "User registered successfully!"})

    except mysql.connector.Error as err:
        return jsonify({"status": "error", "message": str(err)})

    finally:
        cursor.close()
        conn.close()

# Login user
def login_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        if user and bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
            return jsonify({"status": "success", "message": "Login successful!", "user_id": user["id"]})
        else:
            return jsonify({"status": "error", "message": "Invalid credentials!"})

    except mysql.connector.Error as err:
        return jsonify({"status": "error", "message": str(err)})

    finally:
        cursor.close()
        conn.close()
