"""Fetch + cache HL MAINNET history for the live crypto universe, so the
cadence/lookback validation runs on the universe we ACTUALLY trade (HL top-40
crypto) with REAL per-coin funding — not the BinanceUS top-25 / flat-20% proxy.

  fetch_hl_data.py ohlcv     # 4h candles per coin (fast, ~36 calls)
  fetch_hl_data.py funding   # hourly funding per coin, paginated to 2024 (slow)

Caches to research/data/hl/<COIN>_{4h,fund}.json. Re-run skips coins already
cached. Point-in-time by construction: newer coins simply have shorter history.
"""
import sys, os, json, time, ccxt

CACHE = "research/data/hl"
os.makedirs(CACHE, exist_ok=True)
START_MS = 1704067200000  # 2024-01-01


def universe():
    hl = ccxt.hyperliquid({"enableRateLimit": True}); hl.load_markets()
    tk = hl.fetch_tickers()
    def vol(c):
        t = tk.get(f"{c}/USDC:USDC"); return (t.get("quoteVolume") or 0) if t else 0
    uni = json.load(open("research/universe.json"))
    crypto = [s.split("/")[0] for s in uni if vol(s.split("/")[0]) > 0]
    return hl, sorted(crypto, key=lambda c: -vol(c))[:40]


def fetch_ohlcv(hl, coins):
    for c in coins:
        p = f"{CACHE}/{c}_4h.json"
        if os.path.exists(p):
            print(f"  {c}: cached"); continue
        try:
            o = hl.fetch_ohlcv(f"{c}/USDC:USDC", "4h", limit=5000)
            json.dump(o, open(p, "w"))
            print(f"  {c}: {len(o)} bars")
        except Exception as e:
            print(f"  {c}: ERR {e}")
        time.sleep(0.15)


def fetch_funding(hl, coins):
    for c in coins:
        p = f"{CACHE}/{c}_fund.json"
        if os.path.exists(p):
            print(f"  {c}: cached"); continue
        out = []; since = START_MS; last = None
        try:
            while True:
                batch = hl.fetch_funding_rate_history(f"{c}/USDC:USDC", since=since, limit=500)
                if not batch:
                    break
                out.extend((b["timestamp"], b["fundingRate"]) for b in batch)
                end = batch[-1]["timestamp"]
                if end == last or len(batch) < 500:
                    break
                last = end; since = end + 1
                time.sleep(0.08)
            json.dump(out, open(p, "w"))
            print(f"  {c}: {len(out)} funding pts")
        except Exception as e:
            print(f"  {c}: ERR {e} (saved {len(out)})")
            if out:
                json.dump(out, open(p, "w"))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ohlcv"
    hl, coins = universe()
    print(f"universe ({len(coins)}): {', '.join(coins)}\n{mode}:")
    (fetch_ohlcv if mode == "ohlcv" else fetch_funding)(hl, coins)
    print("done")
