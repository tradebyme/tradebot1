import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, render_template, jsonify, request, session, redirect, url_for

# Ensure ststegy2 directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from trading_bot import TradingBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("App2")

app = Flask(__name__, template_folder="templates")
app.secret_key = Config.SECRET_KEY

# Initialize Trading Bot for Strategy 2
bot = TradingBot()
bot.start_background()

@app.before_request
def check_auth():
    """Protect all dashboard views and APIs except /health (for keep-alive)."""
    if request.path == "/health":
        return None

    if not Config.DASHBOARD_PASSWORD:
        return None

    if request.path in ("/login", "/static") or request.path.startswith("/static/"):
        return None

    if session.get("authenticated"):
        return None

    if request.path.startswith("/api/"):
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == Config.DASHBOARD_USERNAME and password == Config.DASHBOARD_PASSWORD:
            session["authenticated"] = True
            logger.info(f"Successful Strategy 2 dashboard login by: {username}")
            return redirect(url_for("index"))
        else:
            logger.warning(f"Failed dashboard login attempt: {username}")
            error = "Invalid username or password. Please try again."

    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "strategy": "AdaptiveSuperSmoother + TrendTargetRibbon",
        "bot_running": bot.is_running,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200

@app.route("/api/status")
def get_status():
    try:
        data = bot.get_dashboard_data()
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/bot/scan", methods=["POST"])
def force_scan():
    try:
        res = bot.scan_once()
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/bot/toggle_pause", methods=["POST"])
def toggle_pause():
    try:
        current_status = bot.risk_manager.state.get("status")
        if current_status == "PAUSED":
            bot.risk_manager.set_pause(False)
            bot.log("INFO", "Bot resumed by user.")
        else:
            bot.risk_manager.set_pause(True)
            bot.log("INFO", "Bot paused by user.")
        return jsonify({"status": bot.risk_manager.state.get("status")}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/bot/close_trade/<trade_id>", methods=["POST"])
def close_trade(trade_id):
    try:
        res = bot.client.close_trade(trade_id)
        bot.log("INFO", f"Manually closed trade #{trade_id} via Web UI.")
        return jsonify(res), 200
    except Exception as e:
        bot.log("ERROR", f"Failed to close trade #{trade_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/bot/close_all", methods=["POST"])
def close_all_trades():
    try:
        open_trades = bot.client.get_open_trades()
        closed_count = 0
        for t in open_trades:
            bot.client.close_trade(t["id"])
            closed_count += 1
        bot.log("WARNING", f"EMERGENCY: Closed ALL {closed_count} open positions via Web UI.")
        return jsonify({"closed_trades_count": closed_count}), 200
    except Exception as e:
        bot.log("ERROR", f"Error in close_all: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/bot/reset_challenge", methods=["POST"])
def reset_challenge():
    try:
        bot.risk_manager.reset_challenge()
        bot.log("INFO", "Challenge tracking reset to $2,000 baseline by user.")
        return jsonify({"status": "reset_complete"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info(f"Starting Strategy 2 Trading Dashboard on {Config.HOST}:{Config.PORT}...")
    app.run(host=Config.HOST, port=Config.PORT, debug=False)
