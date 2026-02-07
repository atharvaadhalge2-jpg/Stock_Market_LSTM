import sqlite3
import hashlib
from datetime import datetime

DB_NAME = "users.db"

# =========================
# USER AUTH
# =========================

def create_user_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hash_password(password))
        )
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()


def login_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hash_password(password))
    )
    user = c.fetchone()
    conn.close()
    return user


# =========================
# PREDICTIONS TABLE
# =========================

def create_prediction_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            stock TEXT,
            predicted_price REAL,
            signal TEXT,
            actual_price REAL,
            correct INTEGER,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()


# =========================
# SAVE PREDICTION
# =========================

def save_prediction(username, stock, price, signal):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions
        (username, stock, predicted_price, signal, actual_price, correct, date)
        VALUES (?, ?, ?, ?, NULL, NULL, datetime('now'))
    """, (username, stock, price, signal))
    conn.commit()
    conn.close()


# =========================
# FETCH USER HISTORY
# =========================

def get_user_predictions(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        SELECT stock, predicted_price, date
        FROM predictions
        WHERE username=?
        ORDER BY id DESC
    """, (username,))

    rows = c.fetchall()
    conn.close()

    cleaned = []
    for stock, price, date in rows:
        try:
            price = float(price)
        except:
            price = None
        cleaned.append((stock, price, date))

    return cleaned


# =========================
# DASHBOARD STATS
# =========================

def get_prediction_stats(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM predictions WHERE username=?", (username,))
    total = c.fetchone()[0]

    c.execute(
        "SELECT COUNT(*) FROM predictions WHERE username=? AND signal='BUY'",
        (username,)
    )
    buy_count = c.fetchone()[0]

    c.execute(
        "SELECT COUNT(*) FROM predictions WHERE username=? AND signal='SELL'",
        (username,)
    )
    sell_count = c.fetchone()[0]

    conn.close()
    return total, buy_count, sell_count


# =========================
# UPDATE ACTUAL PRICE
# =========================

def update_actual_price(stock, actual_price):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        UPDATE predictions
        SET actual_price=?
        WHERE stock=? AND actual_price IS NULL
    """, (actual_price, stock))
    conn.commit()
    conn.close()


# =========================
# EVALUATE ACCURACY
# =========================

def evaluate_predictions():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        UPDATE predictions
        SET correct = CASE
            WHEN signal='BUY'  AND actual_price > predicted_price THEN 1
            WHEN signal='SELL' AND actual_price < predicted_price THEN 1
            ELSE 0
        END
        WHERE actual_price IS NOT NULL
    """)
    conn.commit()
    conn.close()


def get_accuracy(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT COUNT(*), SUM(correct)
        FROM predictions
        WHERE username=? AND actual_price IS NOT NULL
    """, (username,))
    total, correct = c.fetchone()
    conn.close()

    if total == 0 or correct is None:
        return 0.0

    return (correct / total) * 100
