import logging
from collections import deque
from datetime import datetime, timezone
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from config import Config
from trading_bot import TradingBot

# In-memory log buffer for streaming UI
log_buffer = deque(maxlen=200)

class BufferLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_buffer.append({
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "level": record.levelname,
                "message": msg
            })
        except Exception:
            self.handleError(record)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
root_logger = logging.getLogger()
buffer_handler = BufferLogHandler()
buffer_handler.setFormatter(logging.Formatter("%(message)s"))
root_logger.addHandler(buffer_handler)

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

# Initialize Trading Bot
bot = TradingBot()
bot.start()

@app.route("/health")
def health():
    """Unauthenticated health check for Render keep-alive and uptime monitoring."""
    return jsonify({
        "status": "healthy",
        "bot_running": bot.is_running,
        "strategy": "ICT / SMC Asian Liquidity Sweep & Judas Swing",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username == Config.DASHBOARD_USERNAME and password == Config.DASHBOARD_PASSWORD:
            session["authenticated"] = True
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid credentials. Please try again.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("authenticated", None)
    return redirect(url_for("login"))

@app.route("/")
def index():
    if not session.get("authenticated"):
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    if not session.get("authenticated"):
        return jsonify({"error": "Unauthorized"}), 401

    risk_state = bot.risk_manager.state
    bot_info = bot.bot_status

    return jsonify({
        "risk": risk_state,
        "bot": bot_info,
        "logs": list(log_buffer)[-40:],
        "active_instrument": bot_info.get("active_traded_instrument", Config.INSTRUMENTS[0])
    })

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=False)

