import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, render_template, jsonify, request, session, redirect, url_for

# Ensure stategy5 directory is at head of sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from trading_bot import TradingBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("App5")

app = Flask(__name__, template_folder="templates")
app.secret_key = Config.SECRET_KEY

# Initialize the Bot Engine
bot = TradingBot()

# Automatically start background bot thread on launch
bot.start_background()


@app.before_request
def check_auth():
    """Protect dashboard views and APIs except /health (for Render keep-alive & UptimeRobot)."""
    # 1. Always allow health check for Render keep-alive
    if request.path == "/health":
        return None

    # 2. If no password is configured, authentication is disabled
    if not Config.DASHBOARD_PASSWORD:
        return None

    # 3. Allow login page and static assets
    if request.path in ("/login", "/static") or request.path.startswith("/static/"):
        return None

    # 4. Check if session is authenticated
    if session.get("authenticated"):
        return None

    # 5. Unauthorized handling for APIs vs page views
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
            logger.info("Admin dashboard login successful.")
            return redirect(url_for("index"))
        else:
            error = "Invalid username or password. Please try again."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    """Clear session and log out."""
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    """Render main Prop Trading Terminal Dashboard."""
    return render_template(
        "index.html",
        capital=Config.VIRTUAL_CAPITAL,
        instruments=",".join(Config.INSTRUMENTS),
        risk_pct=Config.RISK_PER_TRADE_PCT,
        circuit_breaker=Config.DAILY_CIRCUIT_BREAKER_PCT,
        target_pct=Config.TARGET_PROFIT_PCT,
        max_dd_pct=Config.MAX_DRAWDOWN_PCT
    )


@app.route("/health")
def health():
    """Health check endpoint for Render keep-alive and UptimeRobot."""
    return jsonify({
        "status": "healthy",
        "bot_running": bot.is_running,
        "strategy": "Institutional Trend-ATR Portfolio Engine (Strategy 5)",
        "instruments": Config.INSTRUMENTS,
        "timeframe_mapping": Config.TIMEFRAME_MAPPING,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200


@app.route("/api/status")
def api_status():
    """API endpoint providing live metrics, open trades, floating PnL, and logs."""
    return jsonify(bot.get_status())


@app.route("/api/toggle", methods=["POST"])
def api_toggle():
    """Toggle bot trading state (Active / Paused)."""
    if bot.is_running:
        bot.stop_background()
        bot.risk_manager.state["status"] = "PAUSED"
        bot.risk_manager.state["status_message"] = "Bot paused by user."
        bot.risk_manager._save_state()
        status_str = "PAUSED"
    else:
        bot.start_background()
        bot.risk_manager.state["status"] = "ACTIVE"
        bot.risk_manager.state["status_message"] = "Bot active and scanning."
        bot.risk_manager._save_state()
        status_str = "ACTIVE"

    return jsonify({"success": True, "is_running": bot.is_running, "status": status_str})


@app.route("/api/close_all", methods=["POST"])
def api_close_all():
    """Emergency button to close all open positions across all instruments."""
    logger.warning("Emergency Close All invoked via Web UI.")
    results = bot.client.close_all_trades()
    return jsonify({"success": True, "closed_trades": results})


@app.route("/api/reset_challenge", methods=["POST"])
def api_reset_challenge():
    """Reset challenge metrics for a fresh prop evaluation test."""
    logger.info("Challenge reset requested via Web UI.")
    bot.risk_manager.state = bot.risk_manager._default_state()
    bot.risk_manager._save_state()
    bot.trailing_states.clear()
    return jsonify({"success": True, "message": "Challenge reset successfully."})


if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=False)
