
import sqlite3
from datetime import datetime

DB_NAME = "uploads.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            contains_text INTEGER,
            ocr_text TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_upload(filename, filepath, contains_text, ocr_text):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO uploads (filename, filepath, uploaded_at, contains_text, ocr_text)
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, filepath, datetime.now().isoformat(), int(contains_text), ocr_text))
    conn.commit()
    conn.close()

def get_all_uploads():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM uploads ORDER BY uploaded_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows
