# ============================================================
#  FOREX BOT v6.0 – MAXIMAL AVKASTNING
#  - 5000 backtest-episoder per cykel
#  - MACD + Bollinger + RSI + SMA indikatorer
#  - Dynamisk position sizing
#  - Belönar stora vinster extra
#  - Pappershandel live
# ============================================================

import time
import random
import logging
import numpy as np
import yfinance as yf

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
BACKTEST_EPISODES = 5000
LIVE_STEPS        = 500

ALPHA         = 0.15
GAMMA         = 0.95
EPSILON       = 1.0
EPSILON_MIN   = 0.05
EPSILON_DECAY = 0.998
ACTIONS       = ["buy", "hold", "sell"]


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

    # SMA
    sma10 = float(np.mean(arr[-10:]))
    sma30 = float(np.mean(arr[-30:]))
    if sma10 == 0 or sma30 == 0 or np.isnan(sma10) or np.isnan(sma30):
        return (0, 1, 0, 0, 0)

    # RSI
    diffs  = np.diff(arr[-15:])
    ag     = float(np.mean(np.where(diffs > 0, diffs, 0)))
    al     = float(np.mean(np.where(diffs < 0, -diffs, 0)))
    ag     = ag if ag > 0 else 1e-10
    al     = al if al > 0 else 1e-10
    rsi    = 100.0 - (100.0 / (1.0 + ag / al))

    # MACD (12/26 EMA)
    def ema(data, n):
        k = 2/(n+1)
        e = float(data[0])
        for v in data[1:]:
            e = float(v)*k + e*(1-k)
        return e
    macd = ema(arr[-26:], 12) - ema(arr[-26:], 26)
    macd_signal = 1 if macd > 0 else -1

    # Bollinger Bands
    mid  = float(np.mean(arr[-20:]))
    std  = float(np.std(arr[-20:]))
    cur  = float(arr[-1])
    if std > 0:
        bb_pos = 1 if cur > mid + std else (-1 if cur < mid - std else 0)
    else:
        bb_pos = 0

    # State
    trend  = 1 if sma10 > sma30 * 1.0002 else (-1 if sma10 < sma30 * 0.9998 else 0)
    rz     = 2 if rsi > 70 else (0 if rsi < 30 else 1)
    move   = 1 if float(arr[-1]) > float(arr[-2]) else -1

    return (int(trend), int(rz), int(move), int(macd_signal), int(bb_pos))


def compute_reward(entry, current, position, capital):
    """Belönar stora vinster extra – maximerar avkastning."""
    pnl = (current - entry) / entry if position == "long" else (entry - current) / entry

    if pnl <= -STOP_LOSS:
        return -2.0  # Dubbel straff för förlust

    if pnl >= TAKE_PROFIT * 2:
        return +3.0  # Trippel belöning för stor vinst

    if pnl >= TAKE_PROFIT:
        return +1.5  # Extra belöning för take profit

    if pnl > 0:
        return round(float(pnl * 200), 4)  # Proportionell belöning

    return round(float(pnl * 100), 4)


# ============================================================
#  Q-TABLE
# ============================================================
class QTable:
    def __init__(self):
        self.t = {}

    def get(self, s, a):
        return float(self.t.get(str(s) + a, 0.0))

    def set(self, s, a, v):
        self.t[str(s) + a] = float(v)

    def best(self, s):
        return max(ACTIONS, key=lambda a: self.get(s, a))

    def vals(self, s):
        return {a: round(self.get(s, a), 3) for a in ACTIONS}

    def update(self, s, a, r, ns):
        old  = self.get(s, a)
        best = max(self.get(ns, x) for x in ACTIONS)
        self.set(s, a, old + ALPHA * (r + GAMMA * best - old))

    def size(self):
        return len(self.t)

    def confidence(self, s):
        """Hur säker är boten på sitt beslut (0-1)."""
        vals = [self.get(s, a) for a in ACTIONS]
        mx   = max(vals)
        mn   = min(vals)
        return round((mx - mn) / (abs(mx) + abs(mn) + 1e-10), 2)


# ============================================================
#  FAS 1 – TURBO BACKTEST (5000 episoder)
# ============================================================
def backtest(qt, eps):
    log.info("=" * 50)
    log.info("  FAS 1 – TURBO BACKTEST (maximal avkastning)")
    log.info(f"  {BACKTEST_EPISODES} episoder | MACD+BB+RSI+SMA")
    log.info("=" * 50)

    try:
        df = yf.download(PAIR, period="60d", interval="1h",
                         progress=False, auto_adjust=True)
    except Exception as e:
        log.error(f"  Datafel: {e}")
        return eps

    if df.empty or len(df) < 50:
        log.warning("  För lite data")
        return eps

    close = safe_close(df)
    log.info(f"  {len(close)} ljus laddade – kör {BACKTEST_EPISODES} episoder!")

    best_pnl   = -99999.0
    best_wr    = 0.0
    tot_wins   = 0
    tot_trades = 0

    for ep in range(1, BACKTEST_EPISODES + 1):
        cap   = float(PAPER_CAPITAL)
        pos   = None
        entry = 0.0
        wins  = trades = 0
        ps = pa = None

        start = random.randint(50, max(51, len(close) - 300))
        end   = min(start + 300, len(close))

        for i in range(start, end):
            w = close[max(0, i - 60):i]
            if len(w) < 35:
                continue

            cur   = float(close[i])
            state = compute_state(w)

            # Epsilon-greedy med confidence boost
            conf = qt.confidence(state)
            eff_eps = eps * (1 - conf * 0.5)  # Lägre epsilon när boten är säker
            act = random.choice(ACTIONS) if random.random() < eff_eps else qt.best(state)

            if ps and pos:
                r = compute_reward(entry, cur, pos, cap)
                qt.update(ps, pa, r, state)

            if act == "buy" and pos != "long":
                if pos == "short":
                    pnl = (entry - cur) / entry * TRADE_SIZE
                    cap += pnl; trades += 1
                    if pnl > 0: wins += 1
                pos = "long"; entry = cur

            elif act == "sell" and pos != "short":
                if pos == "long":
                    pnl = (cur - entry) / entry * TRADE_SIZE
                    cap += pnl; trades += 1
                    if pnl > 0: wins += 1
                pos = "short"; entry = cur

            ps, pa = state, act

        ep_pnl = cap - PAPER_CAPITAL
        if ep_pnl > best_pnl:
            best_pnl = ep_pnl
        tot_wins   += wins
        tot_trades += trades
        eps = max(EPSILON_MIN, eps * EPSILON_DECAY)

        if ep % 500 == 0:
            wr = (tot_wins / tot_trades * 100) if tot_trades > 0 else 0
            if wr > best_wr:
                best_wr = wr
            log.info(
                f"  Ep {ep:5d}/{BACKTEST_EPISODES} | "
                f"Bästa PnL: {best_pnl:+.2f}$ | "
                f"Win: {wr:.1f}% | "
                f"ε={eps:.3f} | "
                f"States: {qt.size()}"
            )

    log.info(f"  BACKTEST KLAR!")
    log.info(f"  Bästa PnL: {best_pnl:+.2f}$ | Bästa win-rate: {best_wr:.1f}%")
    log.info(f"  Q-states: {qt.size()} | ε={eps:.3f}")
    log.info("=" * 50)
    return eps


# ============================================================
#  FAS 2 – LIVE PAPPERSHANDEL
# ============================================================
def live(qt, eps, cycle):
    log.info("=" * 50)
    log.info(f"  FAS 2 – LIVE PAPPERSHANDEL (cykel {cycle})")
    log.info(f"  Startkapital: ${PAPER_CAPITAL:,} | Beslut var {SLEEP_SEC}s")
    log.info("=" * 50)

    cap    = float(PAPER_CAPITAL)
    pos    = None
    entry  = 0.0
    trades = wins = 0
    ps = pa = None
    best_cap = float(PAPER_CAPITAL)

    for step in range(1, LIVE_STEPS + 1):
        try:
            df = yf.download(PAIR, period="1d", interval="1m",
                             progress=False, auto_adjust=True)

            if df.empty or len(df) < 40:
                log.warning("  Ingen data – väntar...")
                time.sleep(SLEEP_SEC)
                continue

            close = safe_close(df)
            cur   = float(close[-1])
            state = compute_state(close)
            conf  = qt.confidence(state)

            if random.random() < eps:
                act  = random.choice(ACTIONS)
                mode = f"utforskar"
            else:
                act  = qt.best(state)
                mode = f"Q-beslut (säkerhet: {conf:.0%})"

            qv = qt.vals(state)
            log.info(f"  [{step:3d}/{LIVE_STEPS}] EUR/USD: {cur:.5f}")
            log.info(f"  Beslut: {act.upper()} | {mode}")
            log.info(f"  Q → köp:{qv['buy']} håll:{qv['hold']} sälj:{qv['sell']}")

            if ps and pos:
                r = compute_reward(entry, cur, pos, cap)
                qt.update(ps, pa, r, state)
                log.info(f"  Reward: {r:+.4f} → Q uppdaterad")

            if act == "buy" and pos != "long":
                if pos == "short":
                    pnl = (entry - cur) / entry * TRADE_SIZE
                    cap += pnl; trades += 1
                    if pnl > 0: wins += 1
                    log.info(f"  ↩ Stänger SHORT | PnL: {pnl:+.2f}$")
                pos = "long"; entry = cur
                log.info(f"  ↗ LONG öppnad @ {cur:.5f}")

            elif act == "sell" and pos != "short":
                if pos == "long":
                    pnl = (cur - entry) / entry * TRADE_SIZE
                    cap += pnl; trades += 1
                    if pnl > 0: wins += 1
                    log.info(f"  ↩ Stänger LONG | PnL: {pnl:+.2f}$")
                pos = "short"; entry = cur
                log.info(f"  ↘ SHORT öppnad @ {cur:.5f}")

            else:
                log.info(f"  = Håller ({pos or 'ingen position'})")

            if cap > best_cap:
                best_cap = cap

            wr  = (wins / trades * 100) if trades > 0 else 0
            pnl_total = cap - PAPER_CAPITAL
            log.info(
                f"  Kapital: ${cap:,.2f} ({pnl_total:+.2f}$) | "
                f"Max: ${best_cap:,.2f} | "
                f"Win: {wr:.1f}% | "
                f"Trades: {trades}"
            )
            log.info("-" * 40)

            eps = max(EPSILON_MIN, eps * EPSILON_DECAY)
            ps, pa = state, act

        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.error(f"  Fel: {e}")

        time.sleep(SLEEP_SEC)

    final_pnl = cap - PAPER_CAPITAL
    wr = (wins / trades * 100) if trades > 0 else 0
    log.info(f"  LIVE KLAR! PnL: {final_pnl:+.2f}$ | Win: {wr:.1f}%")
    return eps


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║   FOREX BOT v6.0 – MAXIMAL AVKASTNING        ║")
    log.info("║   5000 ep backtest → Live pappershandel       ║")
    log.info("║   MACD + Bollinger + RSI + SMA                ║")
    log.info("╚══════════════════════════════════════════════╝")

    qt    = QTable()
    eps   = EPSILON
    cycle = 1

    while True:
        try:
            log.info(f"\n{'>'*10} CYKEL {cycle} STARTAR {'<'*10}")
            eps = backtest(qt, eps)
            eps = live(qt, eps, cycle)
            cycle += 1
            log.info(f"{'>'*10} CYKEL {cycle-1} KLAR – BÖRJAR OM {'<'*10}\n")

        except KeyboardInterrupt:
            log.info("Bot stoppad.")
            break
        except Exception as e:
            log.error(f"Fel: {e} – startar om om 60s")
            time.sleep(60)
