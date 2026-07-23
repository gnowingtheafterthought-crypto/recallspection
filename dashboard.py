import streamlit as st
import sqlite3
import pandas as pd
import time
import hashlib
import tempfile
import os
from datetime import datetime

st.set_page_config(page_title="Recallspection Admin", layout="wide")

# ---------- DATABASE SETUP ----------
def init_db():
    db_path = os.path.join(tempfile.gettempdir(), 'recallspection.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE, api_key TEXT, plan TEXT, created_at TIMESTAMP, is_active BOOLEAN)''')
    c.execute('''CREATE TABLE IF NOT EXISTS usage
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, query_type TEXT, timestamp TIMESTAMP, tokens_used INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS payments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, amount REAL, currency TEXT, status TEXT, timestamp TIMESTAMP)''')
    conn.commit()
    return conn

try:
    conn = init_db()
    st.sidebar.success("✅ Database connected")
except Exception as e:
    st.sidebar.error(f"❌ Database error: {e}")
    conn = None

st.sidebar.title("🧠 Recallspection")
st.sidebar.markdown("### Command Centre")

page = st.sidebar.radio("Navigate", ["Dashboard", "Users", "Usage", "Payments", "System"])

# ---------- DASHBOARD ----------
if page == "Dashboard":
    st.title("📊 Dashboard")

    if conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users WHERE is_active=1")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM users WHERE created_at > datetime('now', '-7 days')")
        new = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM usage WHERE timestamp > datetime('now', '-24 hours')")
        q24 = c.fetchone()[0]
        c.execute("SELECT SUM(amount) FROM payments WHERE status='completed' AND timestamp > datetime('now', '-30 days')")
        rev = c.fetchone()[0] or 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Active Users", total)
        col2.metric("New Users (7d)", new)
        col3.metric("Queries (24h)", q24)
        col4.metric("Revenue (30d)", f"${rev:.2f}")

        st.subheader("📋 Recent Activity")
        c.execute('''SELECT u.email, us.query_type, us.timestamp 
                     FROM usage us JOIN users u ON us.user_id = u.id 
                     ORDER BY us.timestamp DESC LIMIT 20''')
        recent = c.fetchall()
        if recent:
            st.dataframe(pd.DataFrame(recent, columns=["Email", "Query Type", "Timestamp"]), use_container_width=True)
        else:
            st.info("No recent activity. Click 'Add Demo User' to get started.")
    else:
        st.error("Database not connected. Please check the logs.")

# ---------- USERS ----------
elif page == "Users":
    st.title("👥 Users")
    if conn:
        c = conn.cursor()
        c.execute('''SELECT id, email, plan, created_at, is_active FROM users ORDER BY created_at DESC''')
        users = c.fetchall()
        if users:
            st.dataframe(pd.DataFrame(users, columns=["ID", "Email", "Plan", "Created", "Active"]), use_container_width=True)
        else:
            st.info("No users yet.")
    else:
        st.error("Database not connected.")

# ---------- USAGE ----------
elif page == "Usage":
    st.title("📈 Usage")
    if conn:
        days = st.number_input("Days", 1, 90, 7)
        c = conn.cursor()
        c.execute('''SELECT DATE(timestamp) as day, COUNT(*) as count 
                     FROM usage 
                     WHERE timestamp > datetime('now', ?) 
                     GROUP BY DATE(timestamp)''', (f'-{days} days',))
        data = c.fetchall()
        if data:
            st.line_chart(pd.DataFrame(data, columns=["Date", "Queries"]).set_index("Date"))
        else:
            st.info("No usage data yet.")
    else:
        st.error("Database not connected.")

# ---------- PAYMENTS ----------
elif page == "Payments":
    st.title("💰 Payments")
    if conn:
        c = conn.cursor()
        c.execute('''SELECT p.id, u.email, p.amount, p.status, p.timestamp 
                     FROM payments p JOIN users u ON p.user_id = u.id 
                     ORDER BY p.timestamp DESC LIMIT 100''')
        payments = c.fetchall()
        if payments:
            st.dataframe(pd.DataFrame(payments, columns=["ID", "User", "Amount", "Status", "Timestamp"]), use_container_width=True)
        else:
            st.info("No payments yet.")
    else:
        st.error("Database not connected.")

# ---------- SYSTEM ----------
elif page == "System":
    st.title("⚙️ System")
    if conn:
        st.success("✅ Database: Connected")
        st.write(f"Database path: {os.path.join(tempfile.gettempdir(), 'recallspection.db')}")
    else:
        st.error("❌ Database: Not connected")
    st.info("📊 Streamlit Cloud is running.")

# ---------- ADD DEMO USER ----------
if st.sidebar.button("➕ Add Demo User"):
    if conn:
        c = conn.cursor()
        email = f"demo_{int(time.time())}@example.com"
        api_key = hashlib.md5(email.encode()).hexdigest()[:16]
        try:
            c.execute("INSERT INTO users (email, api_key, plan, created_at, is_active) VALUES (?, ?, ?, ?, ?)",
                      (email, api_key, "free", datetime.now(), 1))
            conn.commit()
            st.sidebar.success(f"✅ Added: {email}")
            st.sidebar.info(f"🔑 API Key: {api_key}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    else:
        st.sidebar.error("Database not connected")
