# ============================================================
#  FOREX BOT v7.3 – PAPER TRADING + LIVE DASHBOARD
#  Fixes:
#   1. Thread-safe dashboard (reads from object, not file)
#   2. Backtest only every 10 cycles (not every cycle)
#   3. Market hours detection (skips weekends/off-hours)
#   4. Single epsilon decay (no double-decay)
#   5. Spread simulation for realistic PnL
#  Features:
#   - Flask-dashboard on /
#   - Simulated account with $10,000 starting balance
#   - Auto-saves Q-table every 10 cycles
#   - MACD + Bollinger + RSI + SMA
# ============================================================

import time
import random
import logging
import json
import os
import threading
import numpy as np
import yfinance as yf
from flask import Flask, jsonify
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()

PAIR              = "EURUSD=X"
SLEEP_SEC         = 60
PAPER_CAPITAL     = 10_000
TRADE_SIZE        = 1_000
STOP_LOSS         = 0.003
TAKE_PROFIT       = 0.006
SPREAD_PIPS       = 0.00015       # ~1.5 pip spread for EUR/USD
BACKTEST_EPISODES = 5000
BACKTEST_EVERY    = 10            # only backtest every N cycles
SAVE_FILE         = "qtable.json"
PAPER_FILE        = "paper_account.json"
SAVE_EVERY        = 10
PORT              = int(os.environ.get("PORT", 10000))

ALPHA         = 0.15
GAMMA         = 0.95
EPSILON       = 1.0
EPSILON_MIN   = 0.05
EPSILON_DECAY = 0.998
ACTIONS       = ["buy", "hold", "sell"]

bot_status = {
    "cycle":       0,
    "phase":       "Startar...",
    "last_price":  0.0,
    "last_signal": "-",
    "last_update": "-"
}

# ============================================================
#  FLASK DASHBOARD
# ============================================================
app = Flask(__name__)

# Global reference to paper account (set in main)
_paper_ref = None
_paper_lock = threading.Lock()

HTML = """<!DOCTYPE html>
<html lang="sv">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Forex Bot Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f; --panel: #111118; --border: #1e1e2e;
    --green: #00ff88; --red: #ff4466; --yellow: #ffd700;
    --text: #e0e0f0; --muted: #555570;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'Space Mono',monospace; min-height:100vh; padding:2rem; }
  h1 { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:var(--green); margin-bottom:0.25rem; }
  .subtitle { color:var(--muted); font-size:0.75rem; margin-bottom:2rem; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:1rem; margin-bottom:2rem; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:1.25rem; position:relative; overflow:hidden; }
  .card::before { content:''; position:absolute; top:0;left:0;right:0; height:2px; background:var(--green); opacity:0.4; }
  .card-label { font-size:0.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem; }
  .card-value { font-family:'Syne',sans-serif; font-size:1.75rem; font-weight:800; }
  .green{color:var(--green)} .red{color:var(--red)} .yellow{color:var(--yellow)} .white{color:var(--text)}
  .chart-wrap { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:1.25rem; margin-bottom:2rem; }
  .chart-title { font-size:0.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:1rem; }
  canvas { width:100% !important; }
  .trades-wrap { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:1.25rem; }
  .trades-title { font-size:0.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:1rem; }
  table { width:100%; border-collapse:collapse; font-size:0.75rem; }
  th { color:var(--muted); text-align:left; padding:0.4rem 0.5rem; border-bottom:1px solid var(--border); }
  td { padding:0.5rem 0.5rem; border-bottom:1px solid #1a1a28; }
  tr:last-child td { border-bottom:none; }
  .win{color:var(--green)} .loss{color:var(--red)}
  .status-bar { display:flex; align-items:center; gap:0.75rem; margin-bottom:2rem; font-size:0.75rem; color:var(--muted); }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--green); animation:pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
  .signal-buy{color:var(--green);font-weight:700} .signal-sell{color:var(--red);font-weight:700} .signal-hold{color:var(--yellow);font-weight:700}
  .market-closed { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:0.75rem 1.25rem; margin-bottom:1.5rem; font-size:0.75rem; }
  .market-closed .dot-off { width:8px; height:8px; border-radius:50%; background:var(--red); display:inline-block; margin-right:0.5rem; }
</style>
</head>
<body>
<h1>⚡ FOREX BOT v7.3</h1>
<p class="subtitle">EUR/USD · Paper Trading · Q-Learning · Spread: 1.5 pip</p>
<div class="status-bar"><div class="dot"></div><span id="status-text">Laddar...</span></div>
<div id="market-banner" class="market-closed" style="display:none;"><span class="dot-off"></span><span id="market-msg">Marknaden stängd</span></div>
<div class="grid">
  <div class="card"><div class="card-label">Saldo</div><div class="card-value green" id="balance">-</div></div>
  <div class="card"><div class="card-label">Total PnL</div><div class="card-value" id="pnl">-</div></div>
  <div class="card"><div class="card-label">Win Rate</div><div class="card-value yellow" id="winrate">-</div></div>
  <div class="card"><div class="card-label">Antal Trades</div><div class="card-value white" id="trades">-</div></div>
  <div class="card"><div class="card-label">Senaste Signal</div><div class="card-value" id="signal">-</div></div>
  <div class="card"><div class="card-label">EUR/USD Pris</div><div class="card-value white" id="price">-</div></div>
  <div class="card"><div class="card-label">Bot Cykel</div><div class="card-value white" id="cycle">-</div></div>
  <div class="card"><div class="card-label">Fas</div><div class="card-value yellow" id="phase" style="font-size:1rem;padding-top:0.4rem;">-</div></div>
</div>
<div class="chart-wrap">
  <div class="chart-title">Saldoutveckling</div>
  <canvas id="chart" height="120"></canvas>
</div>
<div class="trades-wrap">
  <div class="trades-title">Senaste Trades</div>
  <table>
    <thead><tr><th>Tid</th><th>Typ</th><th>Entry</th><th>Exit</th><th>PnL</th></tr></thead>
    <tbody id="trade-rows"><tr><td colspan="5" style="color:var(--muted);text-align:center;">Väntar på trades...</td></tr></tbody>
  </table>
</div>
<script>
let balanceHistory = [10000];
function drawChart() {
  const canvas = document.getElementById('chart');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * window.devicePixelRatio;
  canvas.height = 120 * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  const w = canvas.offsetWidth, h = 120;
  ctx.clearRect(0,0,w,h);
  if (balanceHistory.length < 2) return;
  const min = Math.min(...balanceHistory)-50, max = Math.max(...balanceHistory)+50;
  const range = max-min||1;
  const toX = i => (i/(balanceHistory.length-1))*w;
  const toY = v => h-((v-min)/range)*(h-10)-5;
  ctx.strokeStyle='#1e1e2e'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const y=(i/4)*h;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  const grad=ctx.createLinearGradient(0,0,0,h);
  grad.addColorStop(0,'rgba(0,255,136,0.3)');grad.addColorStop(1,'rgba(0,255,136,0)');
  ctx.beginPath();ctx.moveTo(toX(0),toY(balanceHistory[0]));
  for(let i=1;i<balanceHistory.length;i++)ctx.lineTo(toX(i),toY(balanceHistory[i]));
  ctx.lineTo(w,h);ctx.lineTo(0,h);ctx.closePath();ctx.fillStyle=grad;ctx.fill();
  ctx.beginPath();ctx.moveTo(toX(0),toY(balanceHistory[0]));
  for(let i=1;i<balanceHistory.length;i++)ctx.lineTo(toX(i),toY(balanceHistory[i]));
  ctx.strokeStyle='#00ff88';ctx.lineWidth=2;ctx.stroke();
}
async function update(){
  try{
    const r=await fetch('/api/status');const d=await r.json();
    const bal=d.balance??10000,pnl=d.total_pnl??0,wr=d.winrate??0;
    document.getElementById('balance').textContent='$'+bal.toLocaleString('sv-SE',{minimumFractionDigits:2,maximumFractionDigits:2});
    const pnlEl=document.getElementById('pnl');
    pnlEl.textContent=(pnl>=0?'+':'')+pnl.toFixed(2)+'$';
    pnlEl.className='card-value '+(pnl>=0?'green':'red');
    document.getElementById('winrate').textContent=wr.toFixed(1)+'%';
    document.getElementById('trades').textContent=d.trades??0;
    document.getElementById('cycle').textContent=d.cycle??0;
    document.getElementById('phase').textContent=d.phase??'-';
    document.getElementById('price').textContent=d.last_price?d.last_price.toFixed(5):'-';
    const sig=(d.last_signal??'-').toUpperCase();
    const sigEl=document.getElementById('signal');sigEl.textContent=sig;
    sigEl.className='card-value signal-'+sig.toLowerCase();
    document.getElementById('status-text').textContent='Senast uppdaterad: '+(d.last_update??'-')+'  |  Fas: '+(d.phase??'-');
    // Market status banner
    const banner=document.getElementById('market-banner');
    if(d.market_open===false){banner.style.display='block';document.getElementById('market-msg').textContent=d.market_msg||'Marknaden stängd';}
    else{banner.style.display='none';}
    if(d.balance_history&&d.balance_history.length>0)balanceHistory=d.balance_history;
    drawChart();
    if(d.trade_log&&d.trade_log.length>0){
      const tbody=document.getElementById('trade-rows');
      tbody.innerHTML=d.trade_log.slice(-10).reverse().map(t=>`
        <tr><td style="color:var(--muted)">${t.time}</td><td class="${t.type==='BUY'?'win':'loss'}">${t.type}</td>
        <td>${t.entry.toFixed(5)}</td><td>${t.exit.toFixed(5)}</td>
        <td class="${t.pnl>=0?'win':'loss'}">${t.pnl>=0?'+':''}${t.pnl.toFixed(2)}$</td></tr>`).join('');
    }
  }catch(e){console.error(e);}
}
update();setInterval(update,5000);window.addEventListener('resize',drawChart);
</script>
</body>
</html>"""

@app.route("/")
def dashboard():
    return HTML

@app.route("/api/status")
def api_status():
    """FIX #1: Read from live object with lock instead of file."""
    data = {**bot_status}
    with _paper_lock:
        if _paper_ref is not None:
            data["balance"]         = _paper_ref.balance
            data["total_pnl"]       = _paper_ref.total_pnl
            data["trades"]          = _paper_ref.trades
            data["winrate"]         = (_paper_ref.wins / _paper_ref.trades * 100) if _paper_ref.trades > 0 else 0
            data["balance_history"] = list(_paper_ref.balance_history[-200:])
            data["trade_log"]       = list(_paper_ref.trade_log[-50:])
    return jsonify(data)

def run_flask():
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)


# ============================================================
#  FIX #3: MARKET HOURS CHECK
# ============================================================
def is_market_open():
    """Forex market is open Sun 22:00 UTC to Fri 22:00 UTC.
    Returns (is_open: bool, message: str)."""
    now = datetime.now(timezone.utc)
    wd = now.weekday()  # Mon=0 ... Sun=6
    hour = now.hour

    # Saturday: always closed
    if wd == 5:
        return False, "Helgstängt (lördag) – nästa öppning söndag 22:00 UTC"
    # Sunday before 22:00: closed
    if wd == 6 and hour < 22:
        return False, f"Helgstängt (söndag) – öppnar 22:00 UTC (om {21 - hour}h)"
    # Friday after 22:00: closed
    if wd == 4 and hour >= 22:
        return False, "Helgstängt (fredag kväll) – nästa öppning söndag 22:00 UTC"

    return True, "Marknaden öppen"


# ============================================================
#  PAPER TRADING KONTO
# ============================================================
class PaperAccount:
    def __init__(self):
        self.balance = float(PAPER_CAPITAL)
        self.position = None
        self.entry = 0.0
        self.entry_type = None
        self.trades = 0
        self.wins = 0
        self.total_pnl = 0.0
        self.balance_history = [float(PAPER_CAPITAL)]
        self.trade_log = []
        self.load()

    def save(self):
        try:
            with _paper_lock:
                with open(PAPER_FILE, "w") as f:
                    json.dump({
                        "balance": self.balance,
                        "position": self.position,
                        "entry": self.entry,
                        "entry_type": self.entry_type,
                        "trades": self.trades,
                        "wins": self.wins,
                        "total_pnl": self.total_pnl,
                        "balance_history": self.balance_history[-200:],
                        "trade_log": self.trade_log[-50:]
                    }, f)
        except Exception as e:
            log.error(f"  Kunde inte spara paper account: {e}")

    def load(self):
        if not os.path.exists(PAPER_FILE):
            log.info("  Nytt paper-konto skapat med $10,000")
            return
        try:
            with open(PAPER_FILE) as f:
                data = json.load(f)
            self.balance = data.get("balance", PAPER_CAPITAL)
            self.position = data.get("position", None)
            self.entry = data.get("entry", 0.0)
            self.entry_type = data.get("entry_type", None)
            self.trades = data.get("trades", 0)
            self.wins = data.get("wins", 0)
            self.total_pnl = data.get("total_pnl", 0.0)
            self.balance_history = data.get("balance_history", [PAPER_CAPITAL])
            self.trade_log = data.get("trade_log", [])
            log.info(f"  Paper-konto laddat | Saldo: ${self.balance:,.2f} | Trades: {self.trades}")
        except Exception as e:
            log.error(f"  Kunde inte ladda: {e}")

    def open_long(self, price):
        """FIX #5: Apply spread – buy at ask (price + half spread)."""
        if self.position == "short":
            self._close(price)
        with _paper_lock:
            self.position = "long"
            self.entry = price + SPREAD_PIPS / 2   # buy at ask
            self.entry_type = "BUY"
        log.info(f"  📈 PAPER BUY  @ {self.entry:.5f} (mid: {price:.5f}, spread: {SPREAD_PIPS:.5f}) | Saldo: ${self.balance:,.2f}")

    def open_short(self, price):
        """FIX #5: Apply spread – sell at bid (price - half spread)."""
        if self.position == "long":
            self._close(price)
        with _paper_lock:
            self.position = "short"
            self.entry = price - SPREAD_PIPS / 2   # sell at bid
            self.entry_type = "SELL"
        log.info(f"  📉 PAPER SELL @ {self.entry:.5f} (mid: {price:.5f}, spread: {SPREAD_PIPS:.5f}) | Saldo: ${self.balance:,.2f}")

    def _close(self, price):
        """FIX #5: Close with spread applied to exit too."""
        with _paper_lock:
            if self.position == "long":
                exit_price = price - SPREAD_PIPS / 2   # sell at bid
                pnl = (exit_price - self.entry) / self.entry * TRADE_SIZE
            else:
                exit_price = price + SPREAD_PIPS / 2   # buy at ask
                pnl = (self.entry - exit_price) / self.entry * TRADE_SIZE

            self.balance += pnl
            self.total_pnl += pnl
            self.trades += 1
            self.balance_history.append(round(self.balance, 2))
            if pnl > 0:
                self.wins += 1
            self.trade_log.append({
                "time": datetime.now().strftime("%H:%M"),
                "type": self.entry_type,
                "entry": self.entry,
                "exit": exit_price,
                "pnl": round(pnl, 2)
            })
            wr = self.wins / self.trades * 100 if self.trades > 0 else 0
            log.info(f"  {'✅' if pnl > 0 else '❌'} STÄNGD | PnL: {pnl:+.2f}$ | Totalt: {self.total_pnl:+.2f}$ | Saldo: ${self.balance:,.2f} | WR: {wr:.1f}%")
            self.position = None
            self.entry_type = None
        self.save()

    def check_sl_tp(self, price):
        if self.position is None:
            return
        if self.position == "long":
            pnl_pct = (price - self.entry) / self.entry
        else:
            pnl_pct = (self.entry - price) / self.entry
        if pnl_pct <= -STOP_LOSS:
            log.info("  🛑 STOP LOSS!")
            self._close(price)
        elif pnl_pct >= TAKE_PROFIT:
            log.info("  🎯 TAKE PROFIT!")
            self._close(price)

    def status(self):
        wr = self.wins / self.trades * 100 if self.trades > 0 else 0
        log.info(f"  💰 Saldo: ${self.balance:,.2f} | PnL: {self.total_pnl:+.2f}$ | Trades: {self.trades} | WR: {wr:.1f}%")


# ============================================================
#  DATAFUNKTIONER
# ============================================================
def safe_close(df):
    try:
        arr = df["Close"].values
    except Exception:
        arr = df.iloc[:, 0].values
    arr = np.array(arr, dtype=np.float64).flatten()
    return arr[~np.isnan(arr)]

def compute_state(arr):
    arr = np.array(arr, dtype=np.float64).flatten()
    if len(arr) < 35:
        return (0, 1, 0, 0, 0)
    sma10 = float(np.mean(arr[-10:]))
    sma30 = float(np.mean(arr[-30:]))
    if sma10 == 0 or sma30 == 0 or np.isnan(sma10) or np.isnan(sma30):
        return (0, 1, 0, 0, 0)
    diffs = np.diff(arr[-15:])
    ag = float(np.mean(np.where(diffs > 0, diffs, 0))) or 1e-10
    al = float(np.mean(np.where(diffs < 0, -diffs, 0))) or 1e-10
    rsi = 100.0 - (100.0 / (1.0 + ag / al))

    def ema(data, n):
        k = 2 / (n + 1)
        e = float(data[0])
        for v in data[1:]:
            e = float(v) * k + e * (1 - k)
        return e

    macd = ema(arr[-26:], 12) - ema(arr[-26:], 26)
    mid = float(np.mean(arr[-20:]))
    std = float(np.std(arr[-20:]))
    cur = float(arr[-1])
    bb_pos = (1 if cur > mid + std else (-1 if cur < mid - std else 0)) if std > 0 else 0
    trend = 1 if sma10 > sma30 * 1.0002 else (-1 if sma10 < sma30 * 0.9998 else 0)
    rz = 2 if rsi > 70 else (0 if rsi < 30 else 1)
    move = 1 if float(arr[-1]) > float(arr[-2]) else -1
    return (int(trend), int(rz), int(move), int(1 if macd > 0 else -1), int(bb_pos))

def compute_reward(entry, current, position):
    pnl = (current - entry) / entry if position == "long" else (entry - current) / entry
    if pnl <= -STOP_LOSS:
        return -2.0
    if pnl >= TAKE_PROFIT * 2:
        return +3.0
    if pnl >= TAKE_PROFIT:
        return +1.5
    if pnl > 0:
        return round(float(pnl * 200), 4)
    return round(float(pnl * 100), 4)


# ============================================================
#  Q-TABLE
# ============================================================
class QTable:
    def __init__(self):
        self.t = {}
        self.cycle = 0
        self.epsilon = EPSILON

    def get(self, s, a):
        return float(self.t.get(str(s) + a, 0.0))

    def set(self, s, a, v):
        self.t[str(s) + a] = float(v)

    def best(self, s):
        return max(ACTIONS, key=lambda a: self.get(s, a))

    def vals(self, s):
        return {a: round(self.get(s, a), 3) for a in ACTIONS}

    def size(self):
        return len(self.t)

    def confidence(self, s):
        vals = [self.get(s, a) for a in ACTIONS]
        mx, mn = max(vals), min(vals)
        return round((mx - mn) / (abs(mx) + abs(mn) + 1e-10), 2)

    def update(self, s, a, r, ns):
        old = self.get(s, a)
        best = max(self.get(ns, x) for x in ACTIONS)
        self.set(s, a, old + ALPHA * (r + GAMMA * best - old))

    def save(self):
        try:
            with open(SAVE_FILE, "w") as f:
                json.dump({"table": self.t, "epsilon": self.epsilon, "cycle": self.cycle}, f)
            log.info(f"  SPARAT! {self.size()} states | ε={self.epsilon:.3f}")
        except Exception as e:
            log.error(f"  Kunde inte spara: {e}")

    def load(self):
        if not os.path.exists(SAVE_FILE):
            log.info("  Ingen sparad Q-tabell – börjar från scratch")
            return EPSILON
        try:
            with open(SAVE_FILE) as f:
                data = json.load(f)
            self.t = data.get("table", {})
            self.epsilon = data.get("epsilon", EPSILON)
            self.cycle = data.get("cycle", 0)
            log.info(f"  LADDAD! {self.size()} states | ε={self.epsilon:.3f} | cykel={self.cycle}")
            return self.epsilon
        except Exception as e:
            log.error(f"  Kunde inte ladda: {e}")
            return EPSILON


# ============================================================
#  FAS 1 – BACKTEST (FIX #4: epsilon only decays here)
# ============================================================
def backtest(qt, eps):
    bot_status["phase"] = "Backtest"
    log.info("=" * 50)
    log.info(f"  FAS 1 – TURBO BACKTEST (cykel {qt.cycle})")
    log.info(f"  {BACKTEST_EPISODES} episoder | MACD+BB+RSI+SMA")
    log.info("=" * 50)
    try:
        df = yf.download(PAIR, period="60d", interval="1h", progress=False, auto_adjust=True)
    except Exception as e:
        log.error(f"  Datafel: {e}")
        return eps
    if df.empty or len(df) < 50:
        log.warning("  För lite data")
        return eps
    close = safe_close(df)
    log.info(f"  {len(close)} ljus laddade")

    best_pnl = -99999.0
    best_wr = 0.0
    tot_wins = 0
    tot_trades = 0

    for ep in range(1, BACKTEST_EPISODES + 1):
        cap = float(PAPER_CAPITAL)
        pos = None
        entry = 0.0
        wins = trades = 0
        ps = pa = None
        start = random.randint(50, max(51, len(close) - 300))
        end = min(start + 300, len(close))

        for i in range(start, end):
            w = close[max(0, i - 60):i]
            if len(w) < 35:
                continue
            cur = float(close[i])
            state = compute_state(w)
            conf = qt.confidence(state)
            eff_e = eps * (1 - conf * 0.5)
            act = random.choice(ACTIONS) if random.random() < eff_e else qt.best(state)

            if ps and pos:
                qt.update(ps, pa, compute_reward(entry, cur, pos), state)

            if act == "buy" and pos != "long":
                if pos == "short":
                    pnl = (entry - cur) / entry * TRADE_SIZE
                    cap += pnl
                    trades += 1
                    if pnl > 0:
                        wins += 1
                pos = "long"
                entry = cur
            elif act == "sell" and pos != "short":
                if pos == "long":
                    pnl = (cur - entry) / entry * TRADE_SIZE
                    cap += pnl
                    trades += 1
                    if pnl > 0:
                        wins += 1
                pos = "short"
                entry = cur

            ps, pa = state, act

        ep_pnl = cap - PAPER_CAPITAL
        if ep_pnl > best_pnl:
            best_pnl = ep_pnl
        tot_wins += wins
        tot_trades += trades

        # FIX #4: Epsilon decay ONLY happens in backtest
        eps = max(EPSILON_MIN, eps * EPSILON_DECAY)

        if ep % 500 == 0:
            wr = (tot_wins / tot_trades * 100) if tot_trades > 0 else 0
            if wr > best_wr:
                best_wr = wr
            log.info(f"  Ep {ep:5d}/{BACKTEST_EPISODES} | PnL: {best_pnl:+.2f}$ | Win: {wr:.1f}% | ε={eps:.3f} | States: {qt.size()}")

    log.info(f"  BACKTEST KLAR! PnL: {best_pnl:+.2f}$ | Win: {best_wr:.1f}%")
    return eps


# ============================================================
#  FAS 2 – LIVE (FIX #4: no epsilon decay here)
# ============================================================
def live(qt, eps, paper):
    bot_status["phase"] = "Live Trading"
    log.info("  FAS 2 – LIVE PAPER TRADING | EUR/USD")

    # FIX #3: Check market hours
    market_open, market_msg = is_market_open()
    bot_status["market_open"] = market_open
    bot_status["market_msg"] = market_msg

    if not market_open:
        log.info(f"  ⏸️  {market_msg}")
        bot_status["phase"] = "Marknaden stängd"
        time.sleep(SLEEP_SEC * 5)  # Sleep longer when market closed
        return eps  # FIX #4: no decay

    try:
        df = yf.download(PAIR, period="1d", interval="1m", progress=False, auto_adjust=True)
        if df.empty or len(df) < 40:
            log.warning("  Ingen live-data – hoppar över")
            time.sleep(SLEEP_SEC)
            return eps  # FIX #4: no decay

        close = safe_close(df)
        cur = float(close[-1])
        state = compute_state(close)
        conf = qt.confidence(state)
        act = random.choice(ACTIONS) if random.random() < eps else qt.best(state)
        qv = qt.vals(state)

        bot_status["last_price"] = round(cur, 5)
        bot_status["last_signal"] = act
        bot_status["last_update"] = datetime.now().strftime("%H:%M:%S")
        bot_status["market_open"] = True

        log.info(f"  EUR/USD: {cur:.5f} | Signal: {act.upper()} | Säkerhet: {conf:.0%}")
        log.info(f"  Q → köp:{qv['buy']} håll:{qv['hold']} sälj:{qv['sell']}")

        paper.check_sl_tp(cur)

        if act == "buy":
            if paper.position != "long":
                paper.open_long(cur)
            else:
                log.info(f"  ⏸️  HÅLLER long @ {paper.entry:.5f}")
        elif act == "sell":
            if paper.position != "short":
                paper.open_short(cur)
            else:
                log.info(f"  ⏸️  HÅLLER short @ {paper.entry:.5f}")
        else:
            log.info("  ⏸️  HÅLLER – ingen ny position")

        paper.status()
        # FIX #4: NO eps decay here – only in backtest

    except Exception as e:
        log.error(f"  Live-fel: {e}")

    time.sleep(SLEEP_SEC)
    return eps


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║   FOREX BOT v7.3 – FIXED & IMPROVED         ║")
    log.info(f"║   Dashboard: port {PORT}                      ║")
    log.info("║   Fixes: thread-safe, spread, market hours   ║")
    log.info("╚══════════════════════════════════════════════╝")

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    log.info(f"  Dashboard live på port {PORT}")

    qt = QTable()
    eps = qt.load()
    paper = PaperAccount()

    # FIX #1: Set global reference for thread-safe dashboard reads
    _paper_ref = paper

    while True:
        try:
            qt.cycle += 1
            bot_status["cycle"] = qt.cycle

            log.info(f"\n>>> CYKEL {qt.cycle} STARTAR <<<")

            # FIX #2: Only backtest every N cycles
            if qt.cycle % BACKTEST_EVERY == 1 or qt.cycle == 1:
                eps = backtest(qt, eps)
                qt.epsilon = eps
            else:
                log.info(f"  Hoppar backtest (nästa vid cykel {((qt.cycle // BACKTEST_EVERY) + 1) * BACKTEST_EVERY + 1})")

            eps = live(qt, eps, paper)

            if qt.cycle % SAVE_EVERY == 0:
                qt.save()

            log.info(f">>> CYKEL {qt.cycle} KLAR <<<\n")

        except KeyboardInterrupt:
            log.info("Stoppar – sparar...")
            qt.save()
            paper.save()
            break
        except Exception as e:
            log.error(f"Fel: {e} – sparar och startar om om 60s")
            qt.save()
            paper.save()
            time.sleep(60)
