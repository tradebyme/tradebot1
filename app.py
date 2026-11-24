"""
OANDA Multi-Bot Prop Firm Cloud Gateway
=======================================
Unified multi-page gateway orchestrating isolated trading engines:
- Page 1: Strategy 1 (M15 Scalper) -> / or /strategy1
- Page 2: Strategy 5 / APEX SNIPER (H4/H1 Institutional Trend-ATR) -> /apex or /strategy5

Architecture:
- 100% Process Isolation: Spawns independent sub-processes for each strategy.
- Complete Credential Isolation: Separate OANDA API keys and account IDs for Bot 1 & Bot 2.
- 1-Click Render.com Deployment: Deploys both bots in a single web service on the free tier.
"""

import os
import sys
import time
import atexit
import signal
import logging
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, Response, redirect, jsonify
import requests

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Gateway]: %(message)s"
)
logger = logging.getLogger("Gateway")
PORT_GATEWAY = int(os.environ.get("PORT", 5000))
PORT_STRAT1 = int(os.environ.get("PORT_STRAT1", 5001))
PORT_STRAT5 = int(os.environ.get("PORT_STRAT5", 5005))

app = Flask(__name__)

# Child processes
p1_proc = None
p5_proc = None


def spawn_subprocesses():
    """Spawn Bot 1 and Bot 2 in dedicated background subprocesses with isolated environments."""
    global p1_proc, p5_proc

    # 1. Environment for Bot 1 (Strategy 1)
    env1 = os.environ.copy()
    env1["PORT"] = str(PORT_STRAT1)
    if env1.get("OANDA_API_KEY_1"):
        env1["OANDA_API_KEY"] = env1["OANDA_API_KEY_1"]
    if env1.get("OANDA_ACCOUNT_ID_1"):
        env1["OANDA_ACCOUNT_ID"] = env1["OANDA_ACCOUNT_ID_1"]

    # 2. Environment for Bot 2 (APEX SNIPER / Strategy 5)
    env5 = os.environ.copy()
    env5["PORT"] = str(PORT_STRAT5)
    if env5.get("OANDA_API_KEY_APEX"):
        env5["OANDA_API_KEY"] = env5["OANDA_API_KEY_APEX"]
    elif env5.get("OANDA_API_KEY_5"):
        env5["OANDA_API_KEY"] = env5["OANDA_API_KEY_5"]

    if env5.get("OANDA_ACCOUNT_ID_APEX"):
        env5["OANDA_ACCOUNT_ID"] = env5["OANDA_ACCOUNT_ID_APEX"]
    elif env5.get("OANDA_ACCOUNT_ID_5"):
        env5["OANDA_ACCOUNT_ID"] = env5["OANDA_ACCOUNT_ID_5"]

    py_exe = sys.executable

    logger.info(f"Launching Strategy 1 on internal port {PORT_STRAT1}...")
    p1_proc = subprocess.Popen(
        [py_exe, str(ROOT_DIR / "stategy_1" / "app.py")],
        cwd=str(ROOT_DIR / "stategy_1"),
        env=env1
    )

    logger.info(f"Launching Strategy 5 (APEX SNIPER) on internal port {PORT_STRAT5}...")
    p5_proc = subprocess.Popen(
        [py_exe, str(ROOT_DIR / "stategy5" / "app.py")],
        cwd=str(ROOT_DIR / "stategy5"),
        env=env5
    )

    logger.info(f"Bots active (Bot 1 PID={p1_proc.pid}, Bot 2 PID={p5_proc.pid}). Waiting for initialization...")
    time.sleep(2.5)


def cleanup():
    """Terminate background bot processes cleanly on shutdown."""
    global p1_proc, p5_proc
    logger.info("Cleaning up bot background processes...")
    for p, name in [(p1_proc, "Strategy 1"), (p5_proc, "Strategy 5")]:
        if p and p.poll() is None:
            logger.info(f"Terminating {name} (PID={p.pid})...")
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()


atexit.register(cleanup)
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))


def proxy_request(target_port: int, target_path: str):
    """Universal HTTP reverse proxy forwarding requests to the target bot port."""
    target_url = f"http://127.0.0.1:{target_port}/{target_path.lstrip('/')}"
    query_string = request.query_string.decode("utf-8")
    if query_string:
        target_url = f"{target_url}?{query_string}"

    headers = {k: v for k, v in request.headers if k.lower() not in ("host", "content-length")}
    headers["X-Forwarded-For"] = request.remote_addr or "127.0.0.1"

    data = request.get_data()

    try:
        resp = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=data,
            cookies=request.cookies,
            allow_redirects=False,
            timeout=15
        )

        excluded_headers = ["content-encoding", "content-length", "transfer-encoding", "connection"]
        resp_headers = [(k, v) for k, v in resp.raw.headers.items() if k.lower() not in excluded_headers]

        return Response(resp.content, resp.status_code, resp_headers)

    except requests.exceptions.RequestException as e:
        logger.error(f"Proxy forwarding error to {target_url}: {e}")
        return jsonify({
            "error": "Bot engine starting up or unavailable.",
            "target": target_url,
            "details": str(e)
        }), 503


# =====================================================================
# ROUTES: HEALTH & MULTI-BOT GATEWAY
# =====================================================================

@app.route("/health")
def health():
    """Health check endpoint for Render keep-alive and uptime monitors."""
    s1_ok = False
    s5_ok = False
    try:
        r1 = requests.get(f"http://127.0.0.1:{PORT_STRAT1}/health", timeout=3)
        s1_ok = (r1.status_code == 200)
    except Exception:
        pass

    try:
        r5 = requests.get(f"http://127.0.0.1:{PORT_STRAT5}/health", timeout=3)
        s5_ok = (r5.status_code == 200)
    except Exception:
        pass

    overall = "healthy" if (s1_ok or s5_ok) else "starting"
    return jsonify({
        "status": overall,
        "strategy_1": {"port": PORT_STRAT1, "alive": s1_ok},
        "strategy_5_apex": {"port": PORT_STRAT5, "alive": s5_ok},
        "timestamp": time.time()
    }), (200 if overall == "healthy" else 503)


# Bot 2: APEX SNIPER / Strategy 5 routes
@app.route("/apex")
@app.route("/apex/")
@app.route("/strategy5")
@app.route("/strategy5/")
def apex_root():
    return proxy_request(PORT_STRAT5, "/")


@app.route("/apex/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
@app.route("/strategy5/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def apex_proxy(path):
    return proxy_request(PORT_STRAT5, path)


# Bot 1: Strategy 1 routes
@app.route("/strategy1")
@app.route("/strategy1/")
def strategy1_root():
    return proxy_request(PORT_STRAT1, "/")


@app.route("/strategy1/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def strategy1_proxy(path):
    return proxy_request(PORT_STRAT1, path)


# Root default route proxies to Strategy 1
@app.route("/", methods=["GET", "POST"])
def root():
    return proxy_request(PORT_STRAT1, "/")


# Fallback catches all other paths (e.g. /api/status when requested from /)
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def fallback_proxy(path):
    if path.startswith("apex") or path.startswith("strategy5"):
        sub = path.split("/", 1)[1] if "/" in path else ""
        return proxy_request(PORT_STRAT5, sub)
    return proxy_request(PORT_STRAT1, path)


if __name__ == "__main__":
    spawn_subprocesses()
    logger.info(f"Starting Multi-Bot Prop Firm Cloud Gateway on port {PORT_GATEWAY}...")
    app.run(host="0.0.0.0", port=PORT_GATEWAY, debug=False)
