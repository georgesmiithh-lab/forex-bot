# ============================================================
#  FOREX BOT - EUR/USD Q-Learning Trading Bot
#  Fas 1: 1000 episoder backtest (så snabbt som möjligt)
#  Fas 2: Live var 60s (lär sig hela tiden)
#  Upprepar fas 1+2 för evigt
# ============================================================

import time
import random
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger()

# ── Inställningar ──────────────────────────────────────────
PAIR            = "EURUSD=X"
SLEEP_SEC       = 60        # Sekunder mellan live-beslut
CAPITAL         = 10_000
TRADE_SIZE      = 1_000
STOP_LOSS       = 0.002
TAKE_PROFIT     = 0.004
BACKTEST_EPISODES = 1000    # Episoder per backtest-runda

# ── Q-Learning ─────────────────────────────────────────────
ALPHA           = 0.1
GAMMA           = 0.9
EPSILON_START   = 1.0
EPSILON_MIN     = 0.05
EPSILON_DECAY   = 0.995
ACTIONS         = ["buy", "hold", "sell"]


# ============================================================
#  Q-TABLE
# ============================================================
class QTable:
    def __init__(self):
        self.table = {}

    def _key(self, state):
        return str(state)

    def get(self, state, action):
        return self.table.get((self._key(state), action), 0.0)

    def update(self, state, action, value):
        self.table[(self._key(state), action)] = value

    def best_action(self, state):
        vals = {a: self.get(state, a) for a in ACTIONS}
        return max(vals, key=vals.get)

    def q_values(self, state):
        return {a: round(self.get(state, a), 3) for a in ACTIONS}

    def size(self):
        return len(self.table)


# ============================================================
#  INDIKATORER → STATE
# ============================================================
def compute_state(close_arr):
    if len(close_arr) < 31:
        return (0, 1, 0)

    sma10 = np.mean(close_arr[-10:])
    sma30 = np.mean(close_arr[-30:])

    diffs = np.diff(close_arr[-15:])
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0)
    ag = np.mean(gains) if np.mean(gains) > 0 else 1e-10
    al = np.mean(losses) if np.mean(losses) > 0 else 1e-10
    rsi = 100 - (100 / (1 + ag / al))

    trend    = 1 if sma10 > sma30 * 1.0002 else (-1 if sma10 < sma30 * 0.9998 else 0)
    rsi_zone = 2 if rsi > 70 else (0 if rsi < 30 else 1)
    move     = 1 if close_arr[-1] > close_arr[-2] else -1

    return (trend, rsi_zone, move)


# ============================================================
#  REWARD
# ============================================================
def compute_reward(entry, current, position):
    if position == "long":
        pnl = (current - entry) / entry
    else:
        pnl = (entry - current) / entry

    if pnl <= -STOP_LOSS:
        return -1.0
    if pnl >= TAKE_PROFIT:
        return +1.0
    return round(pnl * 100, 4)


# ============================================================
#  FAS 1 – BACKTEST (1000 episoder, så snabbt som möjligt)
# ============================================================
def run_backtest_phase(qt, epsilon, cycle):
    log.info("╔══════════════════════════════════════════╗")
    log.info(f"║  FAS 1 – BACKTEST  (cykel {cycle})           ║")
    log.info(f"║  Kör {BACKTEST_EPISODES} episoder så snabbt som möjligt ║")
    log.info("╚══════════════════════════════════════════╝")

    # Hämta historisk data (60 dagar, 1h)
    log.info("  Hämtar historisk EUR/USD data...")
    df = yf.download(PAIR, period="60d", interval="1h",
                     progress=False, auto_adjust=True)
    if df.empty or len(df) < 50:
        log.warning("  Ingen historisk data – hoppar över backtest")
        return epsilon

    close = df["Close"].values
    log.info(f"  {len(close)} ljus laddade")

    best_pnl   = -99999
    total_wins = 0
    total_trades_all = 0

    for ep in range(1, BACKTEST_EPISODES + 1):
        capital  = CAPITAL
        position = None
        entry    = 0.0
        trades   = 0
        wins     = 0
        prev_state  = None
        prev_action = None

        # Slumpmässig startpunkt i historiken
        start = random.randint(50, max(51, len(close) - 200))
        end   = min(start + 200, len(close))

        for i in range(start, end):
            window = close[max(0, i-60):i]
            if len(window) < 31:
                continue

            state = compute_state(window)

            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = qt.best_action(state)

            # Reward från förra steget
            if prev_state is not None and position is not None:
                reward   = compute_reward(entry, close[i], position)
                old_q    = qt.get(prev_state, prev_action)
                best_nxt = max(qt.get(state, a) for a in ACTIONS)
                new_q    = old_q + ALPHA * (reward + GAMMA * best_nxt - old_q)
                qt.update(prev_state, prev_action, new_q)

            # Utför action
            if action == "buy" and position != "long":
                if position == "short":
                    pnl = (entry - close[i]) / entry * TRADE_SIZE
                    capital += pnl
                    trades += 1
                    if pnl > 0: wins += 1
                position = "long"
                entry = close[i]

            elif action == "sell" and position != "short":
                if position == "long":
                    pnl = (close[i] - entry) / entry * TRADE_SIZE
                    capital += pnl
                    trades += 1
                    if pnl > 0: wins += 1
                position = "short"
                entry = close[i]

            prev_state  = state
            prev_action = action

        ep_pnl = capital - CAPITAL
        if ep_pnl > best_pnl:
            best_pnl = ep_pnl

        total_wins        += wins
        total_trades_all  += trades

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # Logga var 100:e episod
        if ep % 100 == 0:
            wr = (total_wins / total_trades_all * 100) if total_trades_all > 0 else 0
            log.info(f"  Episod {ep:4d}/{BACKTEST_EPISODES}  |  "
                     f"Bästa PnL: {best_pnl:+.2f}$  |  "
                     f"Win-rate: {wr:.0f}%  |  "
                     f"ε={epsilon:.3f}  |  "
                     f"Q-states: {qt.size()}")

    log.info(f"  ✓ Backtest klar! Q-table har {qt.size()} states lärda")
    log.info(f"  ✓ Epsilon nu: {epsilon:.3f}")
    return epsilon


# ============================================================
#  FAS 2 – LIVE  (beslut var 60s, lär sig hela tiden)
# ============================================================
def run_live_phase(qt, epsilon, cycle):
    log.info("╔══════════════════════════════════════════╗")
    log.info(f"║  FAS 2 – LIVE  (cykel {cycle})                ║")
    log.info("║  Fattar beslut var 60s, lär sig hela tiden ║")
    log.info("╚══════════════════════════════════════════╝")

    capital     = CAPITAL
    position    = None
    entry_price = 0.0
    trades      = 0
    wins        = 0
    episode     = 0
    prev_state  = None
    prev_action = None

    # Kör live tills nästa backtest-cykel (500 live-beslut = ~8 timmar)
    LIVE_STEPS = 500

    for step in range(1, LIVE_STEPS + 1):
        try:
            df = yf.download(PAIR, period="1d", interval="1m",
                             progress=False, auto_adjust=True)
            if df.empty or len(df) < 35:
                log.warning("  Ingen live-data – väntar 60s...")
                time.sleep(SLEEP_SEC)
                continue

            close        = df["Close"].values
            current_price = float(close[-1])
            state        = compute_state(close)
            episode      += 1

            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
                mode   = f"utforskar (ε={epsilon:.3f})"
            else:
                action = qt.best_action(state)
                mode   = f"Q-beslut  (ε={epsilon:.3f})"

            qvals = qt.q_values(state)

            log.info(f"── Live steg {step}/{LIVE_STEPS} | Episod {episode} ──────────")
            log.info(f"  Kurs:     {current_price:.4f}")
            log.info(f"  State:    trend={state[0]} rsi={state[1]} move={state[2]}")
            log.info(f"  Q-värden: köp={qvals['buy']} håll={qvals['hold']} sälj={qvals['sell']}")
            log.info(f"  Beslut:   {action.upper()} ({mode})")

            # Reward + Q-update från förra steget
            if prev_state is not None and position is not None:
                reward   = compute_reward(entry_price, current_price, position)
                old_q    = qt.get(prev_state, prev_action)
                best_nxt = max(qt.get(state, a) for a in ACTIONS)
                new_q    = old_q + ALPHA * (reward + GAMMA * best_nxt - old_q)
                qt.update(prev_state, prev_action, new_q)
                log.info(f"  Reward:   {reward:+.4f}  →  Q uppdaterad")

            # Utför action
            if action == "buy" and position != "long":
                if position == "short":
                    pnl = (entry_price - current_price) / entry_price * TRADE_SIZE
                    capital += pnl
                    trades += 1
                    if pnl > 0: wins += 1
                    log.info(f"  Stänger SHORT → PnL: {pnl:+.2f}$")
                position    = "long"
                entry_price = current_price
                log.info(f"  Öppnar LONG @ {entry_price:.4f}")

            elif action == "sell" and position != "short":
                if position == "long":
                    pnl = (current_price - entry_price) / entry_price * TRADE_SIZE
                    capital += pnl
                    trades += 1
                    if pnl > 0: wins += 1
                    log.info(f"  Stänger LONG → PnL: {pnl:+.2f}$")
                position    = "short"
                entry_price = current_price
                log.info(f"  Öppnar SHORT @ {entry_price:.4f}")
            else:
                log.info(f"  Håller: {position or 'ingen position'}")

            total_pnl = capital - CAPITAL
            wr        = (wins / trades * 100) if trades > 0 else 0
            log.info(f"  Kapital: ${capital:,.2f} ({total_pnl:+.2f}$) | Win: {wr:.0f}%")

            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            prev_state  = state
            prev_action = action

        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.error(f"  Fel: {e} – försöker igen om 60s")

        time.sleep(SLEEP_SEC)

    return epsilon


# ============================================================
#  MAIN – KÖR FAS 1 + FAS 2 FÖR EVIGT
# ============================================================
if __name__ == "__main__":
    log.info("╔══════════════════════════════════════════╗")
    log.info("║      FOREX Q-LEARNING BOT  v2.0          ║")
    log.info("║  Backtest 1000 ep → Live 500 steg → om   ║")
    log.info("╚══════════════════════════════════════════╝")

    qt      = QTable()       # Delar Q-table mellan backtest och live!
    epsilon = EPSILON_START
    cycle   = 1

    while True:
        try:
            log.info(f"\n{'='*44}")
            log.info(f"  CYKEL {cycle} STARTAR")
            log.info(f"{'='*44}")

            # FAS 1: Backtest 1000 episoder
            epsilon = run_backtest_phase(qt, epsilon, cycle)

            # FAS 2: Live 500 steg (~8 timmar)
            epsilon = run_live_phase(qt, epsilon, cycle)

            cycle += 1
            log.info(f"  ✓ Cykel {cycle-1} klar – börjar om!")

        except KeyboardInterrupt:
            log.info("Bot stoppad.")
            break
        except Exception as e:
            log.error(f"Oväntat fel: {e} – startar om om 60s")
            time.sleep(60)
