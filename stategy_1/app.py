import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, render_template, jsonify, request, session, redirect, url_for

# Ensure stategy_1 directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from trading_bot import TradingBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("App")

app = Flask(__name__, template_folder="templates")
app.secret_key = Config.SECRET_KEY

# Initialize the Bot Engine
bot = TradingBot()

# Automatically start background bot thread
bot.start_background()

@app.before_request
def check_auth():
    """Protect all dashboard views and APIs except /health (needed for UptimeRobot)."""
    # 1. Always allow health check for Render keep-alive (UptimeRobot)
    if request.path == "/health":
        return None

    # 2. If no password is set, authentication is disabled
    if not Config.DASHBOARD_PASSWORD:
        return None

    # 3. Allow login page and static assets
    if request.path in ("/login", "/static") or request.path.startswith("/static/"):
        return None

    # 4. Check if session is already authenticated
    if session.get("authenticated"):
        return None

    # 5. Unauthorized handling
    if request.path.startswith("/api/"):
        return jsonify({"error": "Unauthorized. Please log in."}), 401
    
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    """Render login screen or verify credentials."""
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == Config.DASHBOARD_USERNAME and password == Config.DASHBOARD_PASSWORD:
            session["authenticated"] = True
            logger.info(f"Successful dashboard login by user: {username}")
            return redirect(url_for("index"))
        else:
            logger.warning(f"Failed dashboard login attempt with username: {username}")
            error = "Invalid username or password. Please try again."

    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    """Clear session and log out."""
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
def index():
    """Render the main trading dashboard."""
    return render_template("index.html")

@app.route("/health")
def health():
    """Health check endpoint for Render and UptimeRobot keep-alive."""
    return jsonify({
        "status": "healthy",
        "bot_running": bot.is_running,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200

@app.route("/api/status")
def get_status():
    """API endpoint consumed by the web dashboard frontend."""
    try:
        data = bot.get_dashboard_data()
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error fetching dashboard status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/bot/toggle_pause", methods=["POST"])
def toggle_pause():
    """Toggle Pause / Resume state of the trading bot."""
    current_status = bot.risk_manager.state.get("status")
    if current_status == "PAUSED":
        bot.risk_manager.set_pause(False)
        bot.log("INFO", "Bot resumed by user from dashboard.")
        msg = "Bot resumed."
    else:
        bot.risk_manager.set_pause(True)
        bot.log("WARNING", "Bot paused by user from dashboard.")
        msg = "Bot paused."
    return jsonify({"success": True, "message": msg, "status": bot.risk_manager.state.get("status")})

@app.route("/api/bot/scan", methods=["POST"])
def force_scan():
    """Force an immediate market scan."""
    bot.log("INFO", "Manual scan triggered from web dashboard.")
    res = bot.scan_once()
    return jsonify({"success": True, "result": res})

@app.route("/api/bot/close_all", methods=["POST"])
def close_all():
    """Emergency close all open trades."""
    results = bot.emergency_close_all()
    return jsonify({"success": True, "results": results})

@app.route("/api/bot/close_trade/<trade_id>", methods=["POST"])
def close_single_trade(trade_id):
    """Close a specific open trade."""
    try:
        res = bot.client.close_trade(trade_id)
        bot.sync_closed_trades()
        return jsonify({"success": True, "res": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/bot/reset_challenge", methods=["POST"])
def reset_challenge():
    """Reset the 3-day challenge state."""
    bot.risk_manager.reset_challenge()
    bot.log("WARNING", "Challenge state reset to initial $2,000 baseline.")
    return jsonify({"success": True, "message": "Challenge reset successfully."})

if __name__ == "__main__":
    Config.validate()
    port = int(os.environ.get("PORT", Config.PORT))
    logger.info(f"Starting Forex Trading Bot Web Server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

