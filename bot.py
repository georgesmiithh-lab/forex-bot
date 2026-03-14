import time
import random
import logging
import numpy as np
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger()

PAIR="EURUSD=X"; SLEEP_SEC=60; CAPITAL=10_000; TRADE_SIZE=1_000
STOP_LOSS=0.002; TAKE_PROFIT=0.004; BACKTEST_EPISODES=1000
ALPHA=0.1; GAMMA=0.9; EPSILON_START=1.0; EPSILON_MIN=0.05; EPSILON_DECAY=0.995
ACTIONS=["buy","hold","sell"]

def get_close(df):
    try:
        c = df["Close"].values.flatten()
    except:
        c = df.iloc[:,0].values.flatten()
    return np.array(c, dtype=np.float64)

class QTable:
    def __init__(self): self.table={}
    def _key(self,s): return str(s)
    def get(self,s,a): return float(self.table.get((self._key(s),a),0.0))
    def update(self,s,a,v): self.table[(self._key(s),a)]=float(v)
    def best_action(self,s): return max(ACTIONS,key=lambda a:self.get(s,a))
    def q_values(self,s): return {a:round(self.get(s,a),3) for a in ACTIONS}
    def size(self): return len(self.table)

def compute_state(arr):
    arr=np.array(arr,dtype=np.float64).flatten()
    if len(arr)<31: return (0,1,0)
    sma10=float(np.mean(arr[-10:])); sma30=float(np.mean(arr[-30:]))
    if sma10==0 or sma30==0 or np.isnan(sma10) or np.isnan(sma30): return (0,1,0)
    diffs=np.diff(arr[-15:])
    ag=float(np.mean(np.where(diffs>0,diffs,0))); al=float(np.mean(np.where(diffs<0,-diffs,0)))
    ag=ag if ag>0 else 1e-10; al=al if al>0 else 1e-10
    rsi=100.0-(100.0/(1.0+ag/al))
    trend=1 if sma10>sma30*1.0002 else(-1 if sma10<sma30*0.9998 else 0)
    rsi_zone=2 if rsi>70 else(0 if rsi<30 else 1)
    move=1 if float(arr[-1])>float(arr[-2]) else -1
    return (int(trend),int(rsi_zone),int(move))

def compute_reward(entry,current,position):
    entry=float(entry); current=float(current)
    pnl=(current-entry)/entry if position=="long" else(entry-current)/entry
    if pnl<=-STOP_LOSS: return -1.0
    if pnl>=TAKE_PROFIT: return +1.0
    return round(float(pnl*100),4)

def run_backtest_phase(qt,epsilon,cycle):
    log.info(f"FAS 1 - BACKTEST cykel {cycle} - {BACKTEST_EPISODES} episoder")
    df=yf.download(PAIR,period="60d",interval="1h",progress=False,auto_adjust=True)
    if df.empty or len(df)<50:
        log.warning("Ingen data"); return epsilon
    close=get_close(df)
    log.info(f"{len(close)} ljus laddade")
    best_pnl=-99999.0; total_wins=0; total_trades=0
    for ep in range(1,BACKTEST_EPISODES+1):
        capital=float(CAPITAL); position=None; entry=0.0; trades=0; wins=0
        prev_state=None; prev_action=None
        start=random.randint(50,max(51,len(close)-200))
        end=min(start+200,len(close))
        for i in range(start,end):
            window=close[max(0,i-60):i]
            if len(window)<31: continue
            cur=float(close[i]); state=compute_state(window)
            action=random.choice(ACTIONS) if random.random()<epsilon else qt.best_action(state)
            if prev_state is not None and position is not None:
                r=compute_reward(entry,cur,position)
                old_q=qt.get(prev_state,prev_action)
                new_q=old_q+ALPHA*(r+GAMMA*max(qt.get(state,a) for a in ACTIONS)-old_q)
                qt.update(prev_state,prev_action,new_q)
            if action=="buy" and position!="long":
                if position=="short":
                    pnl=(entry-cur)/entry*TRADE_SIZE; capital+=pnl; trades+=1
                    if pnl>0: wins+=1
                position="long"; entry=cur
            elif action=="sell" and position!="short":
                if position=="long":
                    pnl=(cur-entry)/entry*TRADE_SIZE; capital+=pnl; trades+=1
                    if pnl>0: wins+=1
                position="short"; entry=cur
            prev_state=state; prev_action=action
        ep_pnl=capital-CAPITAL
        if ep_pnl>best_pnl: best_pnl=ep_pnl
        total_wins+=wins; total_trades+=trades
        epsilon=max(EPSILON_MIN,epsilon*EPSILON_DECAY)
        if ep%100==0:
            wr=(total_wins/total_trades*100) if total_trades>0 else 0
            log.info(f"Episod {ep}/{BACKTEST_EPISODES} | Bästa PnL: {best_pnl:+.2f}$ | Win: {wr:.0f}% | ε={epsilon:.3f} | Q-states: {qt.size()}")
    log.info(f"Backtest klar! Q-states: {qt.size()} | ε={epsilon:.3f}")
    return epsilon

def run_live_phase(qt,epsilon,cycle):
    log.info(f"FAS 2 - LIVE cykel {cycle} - beslut var {SLEEP_SEC}s")
    capital=float(CAPITAL); position=None; entry_price=0.0
    trades=0; wins=0; prev_state=None; prev_action=None; LIVE_STEPS=500
    for step in range(1,LIVE_STEPS+1):
        try:
            df=yf.download(PAIR,period="1d",interval="1m",progress=False,auto_adjust=True)
            if df.empty or len(df)<35:
                log.warning("Ingen live-data"); time.sleep(SLEEP_SEC); continue
            close=get_close(df); current_price=float(close[-1]); state=compute_state(close)
            action=random.choice(ACTIONS) if random.random()<epsilon else qt.best_action(state)
            mode="utforskar" if random.random()<epsilon else "Q-beslut"
            qvals=qt.q_values(state)
            log.info(f"Live {step}/{LIVE_STEPS} | Kurs: {current_price:.4f} | {action.upper()} ({mode}) | Q: {qvals}")
            if prev_state is not None and position is not None:
                r=compute_reward(entry_price,current_price,position)
                old_q=qt.get(prev_state,prev_action)
                new_q=old_q+ALPHA*(r+GAMMA*max(qt.get(state,a) for a in ACTIONS)-old_q)
                qt.update(prev_state,prev_action,new_q)
                log.info(f"Reward: {r:+.4f} Q uppdaterad")
            if action=="buy" and position!="long":
                if position=="short":
                    pnl=(entry_price-current_price)/entry_price*TRADE_SIZE; capital+=pnl; trades+=1
                    if pnl>0: wins+=1
                    log.info(f"Stänger SHORT PnL: {pnl:+.2f}$")
                position="long"; entry_price=current_price; log.info(f"Öppnar LONG @ {entry_price:.4f}")
            elif action=="sell" and position!="short":
                if position=="long":
                    pnl=(current_price-entry_price)/entry_price*TRADE_SIZE; capital+=pnl; trades+=1
                    if pnl>0: wins+=1
                    log.info(f"Stänger LONG PnL: {pnl:+.2f}$")
                position="short"; entry_price=current_price; log.info(f"Öppnar SHORT @ {entry_price:.4f}")
            else:
                log.info(f"Håller: {position or 'ingen'}")
            wr=(wins/trades*100) if trades>0 else 0
            log.info(f"Kapital: ${capital:,.2f} ({capital-CAPITAL:+.2f}$) | Win: {wr:.0f}%")
            epsilon=max(EPSILON_MIN,epsilon*EPSILON_DECAY)
            prev_state=state; prev_action=action
        except KeyboardInterrupt: raise
        except Exception as e: log.error(f"Fel: {e}")
        time.sleep(SLEEP_SEC)
    return epsilon

if __name__=="__main__":
    log.info("FOREX Q-LEARNING BOT v4.0 - Backtest 1000ep + Live 500 steg - för evigt")
    qt=QTable(); epsilon=EPSILON_START; cycle=1
    while True:
        try:
            log.info(f"CYKEL {cycle} STARTAR")
            epsilon=run_backtest_phase(qt,epsilon,cycle)
            epsilon=run_live_phase(qt,epsilon,cycle)
            cycle+=1; log.info(f"Cykel {cycle-1} klar - börjar om!")
        except KeyboardInterrupt: log.info("Stoppad."); break
        except Exception as e: log.error(f"Fel: {e} - startar om 60s"); time.sleep(60)
