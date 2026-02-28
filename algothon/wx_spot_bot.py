"""WX_SPOT Competitive Trading Bot v2

Trades WX_SPOT = round(Temperature_F) x Humidity at Sunday 12:00 London.

Architecture:
  1. Multi-model fair value — averages ECMWF, GFS, UKMO, GEM forecasts
  2. Confidence-weighted sizing — bigger when models agree, smaller when they don't
  3. Inventory-skewed quoting — lean prices to offload position risk
  4. Multi-level quotes — capture spread at several depths
  5. Time-aware aggressiveness — tighter quotes closer to settlement
  6. IOC for directional — don't leave stale aggressive orders in the book
  7. Rounding-aware — temp is rounded before multiplication, creating discrete jumps
"""

import math
import time
import statistics
import warnings
from datetime import datetime, timedelta
from threading import Thread

import pandas as pd
import requests

# Suppress SSL warnings when verify=False is needed
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

from bot_template import (
    BaseBot, OrderBook, Order, OrderRequest, OrderResponse, Side, Trade, Product
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — Edit these
# ═══════════════════════════════════════════════════════════════════════════════
TEST_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
CHALLENGE_URL = "REPLACE_WITH_CHALLENGE_URL"

EXCHANGE_URL = TEST_URL  # switch to CHALLENGE_URL when competing

USERNAME = "EmploymentSeekers"      # TODO
PASSWORD = "aryan"  # TODO

PRODUCT = "WX_SPOT"

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Position limits
MAX_POSITION = 80          # hard limit (exchange is +/-100, we leave buffer)
SOFT_POSITION = 50         # above this, start being cautious

# Quoting
BASE_HALF_WIDTH = 8        # base half-spread around fair value
MIN_HALF_WIDTH = 3         # minimum half-spread (tight when confident)
NUM_QUOTE_LEVELS = 3       # number of price levels on each side
LEVEL_STEP = 5             # price gap between quote levels
VOLUME_PER_LEVEL = 3       # volume at each quote level

# Directional
EDGE_THRESHOLD_AGGRESSIVE = 50   # take big directional position
EDGE_THRESHOLD_MODERATE = 25     # take small directional position
DIRECTIONAL_VOL_BIG = 8
DIRECTIONAL_VOL_SMALL = 3

# Timing
LOOP_INTERVAL = 8          # seconds between iterations
DATA_REFRESH_INTERVAL = 180  # re-fetch weather every 3 minutes
RATE_LIMIT_PAUSE = 1.0     # pause between API calls

# Inventory skew: how much to shift quotes per unit of position
SKEW_PER_LOT = 0.15        # shift theo this much per lot of position

# ═══════════════════════════════════════════════════════════════════════════════
# WEATHER DATA — Multi-model forecasting
# ═══════════════════════════════════════════════════════════════════════════════

LONDON_LAT, LONDON_LON = 51.5074, -0.1278

WEATHER_MODELS = [
    "ecmwf_ifs025",   # ECMWF high-res (best for Europe, 0.25 deg)
    "gfs_seamless",   # NOAA GFS
    "ukmo_seamless",  # UK Met Office (best for UK!)
    "gem_seamless",   # Canadian GEM
]

# Model weights — UK Met Office and ECMWF are best for London
MODEL_WEIGHTS = {
    "ecmwf_ifs025": 1.5,   # excellent for Europe
    "ukmo_seamless": 2.0,  # best for UK specifically
    "gfs_seamless": 1.0,   # decent global model
    "gem_seamless": 0.8,   # decent but less accurate for UK
}


def celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def get_settlement_target() -> pd.Timestamp:
    """Return the next Sunday 12:00 London time."""
    now = pd.Timestamp.now(tz="Europe/London")
    days_until_sunday = (6 - now.weekday()) % 7
    if days_until_sunday == 0 and now.hour >= 12:
        days_until_sunday = 7
    return now.normalize() + pd.Timedelta(days=days_until_sunday, hours=12)


def hours_to_settlement() -> float:
    """Hours remaining until settlement."""
    now = pd.Timestamp.now(tz="Europe/London")
    target = get_settlement_target()
    return max(0, (target - now).total_seconds() / 3600)


def fetch_model_forecast(model: str, target: pd.Timestamp) -> dict | None:
    """Fetch forecast from a single model and extract the settlement-time reading.

    Returns dict with temp_c, temp_f, temp_f_rounded, humidity, fair_value
    or None on failure.
    """
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": LONDON_LAT,
                "longitude": LONDON_LON,
                "minutely_15": "temperature_2m,relative_humidity_2m",
                "forecast_minutely_15": 384,   # ~4 days forward
                "past_minutely_15": 96,        # 24h backward
                "timezone": "Europe/London",
                "models": model,
            },
            timeout=15,
            verify=False,  # fallback for SSL cert issues
        )
        resp.raise_for_status()
        m = resp.json()["minutely_15"]

        times = pd.to_datetime(m["time"]).tz_localize("Europe/London")
        temps = m["temperature_2m"]
        humids = m["relative_humidity_2m"]

        # Find closest reading to settlement
        diffs = [abs(t - target) for t in times]
        idx = diffs.index(min(diffs))

        if temps[idx] is None or humids[idx] is None:
            return None

        temp_c = temps[idx]
        temp_f = celsius_to_fahrenheit(temp_c)
        temp_f_rounded = round(temp_f)
        humidity = humids[idx]
        fair_value = temp_f_rounded * humidity

        return {
            "model": model,
            "temp_c": temp_c,
            "temp_f": temp_f,
            "temp_f_rounded": temp_f_rounded,
            "humidity": humidity,
            "fair_value": fair_value,
        }
    except Exception as e:
        print("  [WARN] Model %s failed: %s" % (model, e))
        return None


class FairValueEstimate:
    """Multi-model fair value with confidence metrics."""

    def __init__(self):
        self.fair_value: float = 0
        self.confidence: float = 0       # 0-1, higher = more agreement
        self.model_spread: float = 0     # range of model fair values
        self.model_std: float = 0        # std deviation
        self.num_models: int = 0
        self.estimates: list[dict] = []
        self.timestamp: float = 0

    def __repr__(self):
        return ("FV=%.0f conf=%.2f spread=%.0f models=%d"
                % (self.fair_value, self.confidence, self.model_spread, self.num_models))


def compute_multi_model_fair_value() -> FairValueEstimate:
    """Query all weather models and produce a weighted fair value with confidence."""
    target = get_settlement_target()
    result = FairValueEstimate()
    result.timestamp = time.monotonic()

    estimates = []
    for model in WEATHER_MODELS:
        est = fetch_model_forecast(model, target)
        if est is not None:
            estimates.append(est)

    if not estimates:
        return result

    result.estimates = estimates
    result.num_models = len(estimates)

    # Weighted average fair value
    total_weight = 0
    weighted_fv = 0
    fvs = []
    for est in estimates:
        w = MODEL_WEIGHTS.get(est["model"], 1.0)
        weighted_fv += est["fair_value"] * w
        total_weight += w
        fvs.append(est["fair_value"])

    result.fair_value = weighted_fv / total_weight if total_weight > 0 else 0

    # Confidence: based on model agreement
    if len(fvs) >= 2:
        result.model_std = statistics.stdev(fvs)
        result.model_spread = max(fvs) - min(fvs)
        # Confidence scaling: std=0 -> 1.0, std=300 -> 0.5, std=600 -> 0.25
        result.confidence = max(0.1, 1.0 / (1.0 + result.model_std / 300.0))
    else:
        result.confidence = 0.5  # single model, moderate confidence

    # Boost confidence as we get closer to settlement (forecasts converge)
    hours_left = hours_to_settlement()
    if hours_left < 6:
        result.confidence = min(1.0, result.confidence * 1.5)
    elif hours_left < 12:
        result.confidence = min(1.0, result.confidence * 1.2)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BOT
# ═══════════════════════════════════════════════════════════════════════════════

class WxSpotBot(BaseBot):
    """Competitive WX_SPOT trading bot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimate: FairValueEstimate | None = None
        self.last_data_fetch: float = 0
        self.fill_count = 0
        self._last_ob: dict[str, OrderBook] = {}

    # ── SSE callbacks ───────────────────────────────────────────────────────

    def on_orderbook(self, orderbook: OrderBook):
        self._last_ob[orderbook.product] = orderbook

    def on_trades(self, trade: Trade):
        self.fill_count += 1
        if trade.buyer == self.username:
            side_str = "BOUGHT"
        else:
            side_str = "SOLD"
        print("  >>> FILL #%d: %s %dx %s @ %s"
              % (self.fill_count, side_str, trade.volume, trade.product, trade.price))

    # ── Data refresh ────────────────────────────────────────────────────────

    def refresh_fair_value(self, force=False) -> bool:
        """Re-fetch weather data if stale. Returns True if updated."""
        now = time.monotonic()
        if not force and self.estimate is not None:
            if (now - self.last_data_fetch) < DATA_REFRESH_INTERVAL:
                return False

        print("\n  Fetching multi-model weather forecasts...")
        self.estimate = compute_multi_model_fair_value()
        self.last_data_fetch = now

        if self.estimate.num_models == 0:
            print("  [ERROR] No models returned data!")
            return False

        print("  [FV] %s" % self.estimate)
        for est in self.estimate.estimates:
            label = est["model"].ljust(20)
            print("    %s: %.1f C = %.1f F (rd:%d) x %s%% = %d"
                  % (label, est["temp_c"], est["temp_f"],
                     est["temp_f_rounded"], est["humidity"], est["fair_value"]))
        return True

    # ── Market reading ──────────────────────────────────────────────────────

    def get_market_mid(self) -> float | None:
        """Get mid price from orderbook, ignoring own orders."""
        ob = self.get_orderbook(PRODUCT)
        bids = [o.price for o in ob.buy_orders if o.volume - o.own_volume > 0]
        asks = [o.price for o in ob.sell_orders if o.volume - o.own_volume > 0]
        if bids and asks:
            return (max(bids) + min(asks)) / 2
        elif bids:
            return max(bids)
        elif asks:
            return min(asks)
        return None

    # ── IOC helper ──────────────────────────────────────────────────────────

    def send_ioc(self, order: OrderRequest) -> OrderResponse | None:
        """Send order and immediately cancel remainder (IOC simulation)."""
        resp = self.send_order(order)
        if resp and resp.volume > 0:
            self.cancel_order(resp.id)
        return resp

    # ── Quoting logic ───────────────────────────────────────────────────────

    def compute_skewed_theo(self, raw_theo: float, position: int) -> float:
        """Shift theoretical value to lean away from inventory.

        If long -> lower theo -> sell side quotes become more attractive.
        If short -> raise theo -> buy side quotes become more attractive.
        """
        return raw_theo - position * SKEW_PER_LOT

    def compute_half_width(self) -> float:
        """Dynamic half-width based on confidence and time to settlement."""
        if self.estimate is None:
            return BASE_HALF_WIDTH

        confidence = self.estimate.confidence
        hours_left = hours_to_settlement()

        # Base width inversely proportional to confidence
        width = BASE_HALF_WIDTH / max(confidence, 0.2)

        # Tighten as we approach settlement (forecast becomes more accurate)
        if hours_left < 3:
            width *= 0.5
        elif hours_left < 6:
            width *= 0.7
        elif hours_left < 12:
            width *= 0.85

        return max(MIN_HALF_WIDTH, min(width, 40))  # clamp to reasonable range

    def compute_quote_volume(self, position: int, side: Side, level: int) -> int:
        """Volume per quote level, adjusted for position and level depth."""
        # Reduce volume at deeper levels
        base_vol = max(1, VOLUME_PER_LEVEL - level)

        # Reduce volume on the side that would increase position risk
        if side == Side.BUY and position > SOFT_POSITION:
            scale = max(0, (MAX_POSITION - position) / (MAX_POSITION - SOFT_POSITION))
            base_vol = max(1, int(base_vol * scale))
        elif side == Side.SELL and position < -SOFT_POSITION:
            scale = max(0, (MAX_POSITION + position) / (MAX_POSITION - SOFT_POSITION))
            base_vol = max(1, int(base_vol * scale))

        # Hard cap: never exceed remaining capacity
        if side == Side.BUY:
            base_vol = min(base_vol, MAX_POSITION - position)
        else:
            base_vol = min(base_vol, MAX_POSITION + position)

        return max(0, base_vol)

    def generate_quotes(self, theo: float, tick: float, position: int) -> list[OrderRequest]:
        """Generate multi-level bid/ask quotes around skewed theo."""
        skewed_theo = self.compute_skewed_theo(theo, position)
        half_width = self.compute_half_width()

        orders = []
        for level in range(NUM_QUOTE_LEVELS):
            offset = half_width + level * LEVEL_STEP

            # Bid
            bid_price = math.floor((skewed_theo - offset) / tick) * tick
            bid_vol = self.compute_quote_volume(position, Side.BUY, level)
            if bid_vol > 0 and bid_price > 0:
                orders.append(OrderRequest(PRODUCT, bid_price, Side.BUY, bid_vol))

            # Ask
            ask_price = math.ceil((skewed_theo + offset) / tick) * tick
            ask_vol = self.compute_quote_volume(position, Side.SELL, level)
            if ask_vol > 0 and ask_price > 0:
                orders.append(OrderRequest(PRODUCT, ask_price, Side.SELL, ask_vol))

        return orders

    # ── Directional logic ───────────────────────────────────────────────────

    def maybe_take_directional(self, theo: float, market_mid: float,
                                position: int, tick: float) -> bool:
        """Take directional position if strong edge exists. Returns True if acted."""
        edge = theo - market_mid
        abs_edge = abs(edge)

        if abs_edge < EDGE_THRESHOLD_MODERATE:
            return False

        # Scale volume by confidence and edge size
        confidence = self.estimate.confidence if self.estimate else 0.5

        if abs_edge >= EDGE_THRESHOLD_AGGRESSIVE:
            base_vol = DIRECTIONAL_VOL_BIG
        else:
            base_vol = DIRECTIONAL_VOL_SMALL

        vol = max(1, int(base_vol * confidence))

        if edge > 0:
            # Theo above market -> BUY
            remaining = MAX_POSITION - position
            vol = min(vol, remaining)
            if vol <= 0:
                return False
            # Price at or slightly above best ask to get filled
            price = math.ceil(market_mid / tick) * tick + tick
            print("  [DIRECTIONAL] BUY %d@%s (edge=%+.0f, conf=%.2f)"
                  % (vol, price, edge, confidence))
            self.send_ioc(OrderRequest(PRODUCT, price, Side.BUY, vol))
        else:
            # Theo below market -> SELL
            remaining = MAX_POSITION + position
            vol = min(vol, remaining)
            if vol <= 0:
                return False
            price = math.floor(market_mid / tick) * tick - tick
            print("  [DIRECTIONAL] SELL %d@%s (edge=%+.0f, conf=%.2f)"
                  % (vol, price, edge, confidence))
            self.send_ioc(OrderRequest(PRODUCT, price, Side.SELL, vol))

        return True

    # ── Main loop ───────────────────────────────────────────────────────────

    def run_loop(self):
        """Main trading loop."""
        products = {p.symbol: p for p in self.get_products()}
        if PRODUCT not in products:
            print("Product %s not found! Available: %s" % (PRODUCT, list(products.keys())))
            return

        product = products[PRODUCT]
        tick = product.tickSize

        self.start()  # start SSE stream
        print("\n" + "=" * 60)
        print("  WX_SPOT Bot v2 started")
        print("  Product: %s (tick=%s)" % (PRODUCT, tick))
        print("  Hours to settlement: %.1f" % hours_to_settlement())
        print("=" * 60)

        iteration = 0
        try:
            while True:
                iteration += 1

                # 1. Refresh fair value
                self.refresh_fair_value()
                if self.estimate is None or self.estimate.num_models == 0:
                    print("  Waiting for data...")
                    time.sleep(LOOP_INTERVAL)
                    continue

                theo = self.estimate.fair_value

                # 2. Cancel all existing orders
                self.cancel_all_orders()
                time.sleep(RATE_LIMIT_PAUSE)

                # 3. Read position and market
                position = self.get_positions().get(PRODUCT, 0)
                time.sleep(RATE_LIMIT_PAUSE)

                market_mid = self.get_market_mid()
                time.sleep(RATE_LIMIT_PAUSE)

                hours_left = hours_to_settlement()
                half_width = self.compute_half_width()

                print("\n  [%d] theo=%.0f mid=%s pos=%+d hw=%.1f conf=%.2f hrs=%.1f"
                      % (iteration, theo, market_mid, position, half_width,
                         self.estimate.confidence, hours_left))

                # 4. Directional: take position if strong edge
                if market_mid is not None:
                    took_directional = self.maybe_take_directional(
                        theo, market_mid, position, tick)
                    if took_directional:
                        time.sleep(RATE_LIMIT_PAUSE)
                        # Re-read position after directional trade
                        position = self.get_positions().get(PRODUCT, 0)
                        time.sleep(RATE_LIMIT_PAUSE)

                # 5. Market making: quote around fair value
                quotes = self.generate_quotes(theo, tick, position)
                if quotes:
                    self.send_orders(quotes)
                    buy_quotes = [q for q in quotes if q.side == Side.BUY]
                    sell_quotes = [q for q in quotes if q.side == Side.SELL]
                    if buy_quotes and sell_quotes:
                        print("  [QUOTE] %d bids (%.0f-%.0f) / %d asks (%.0f-%.0f)"
                              % (len(buy_quotes), buy_quotes[0].price,
                                 buy_quotes[-1].price, len(sell_quotes),
                                 sell_quotes[0].price, sell_quotes[-1].price))

                # 6. Show PnL
                try:
                    pnl = self.get_pnl()
                    if pnl:
                        if isinstance(pnl, dict):
                            total = pnl.get("totalProfit", 0)
                            # Extract per-product WX_SPOT PnL
                            wx_pnl = 0
                            for d in pnl.get("details", []):
                                if d.get("product") == PRODUCT:
                                    wx_pnl = d.get("profit", 0)
                            print("  [PNL] total=%+.0f  %s=%+.0f" % (total, PRODUCT, wx_pnl))
                        elif isinstance(pnl, list):
                            total = sum(p.get("totalProfit", p.get("profit", 0)) for p in pnl)
                            print("  [PNL] %+.0f" % total)
                except Exception as e:
                    print("  [PNL] error: %s" % e)

                time.sleep(LOOP_INTERVAL)

        except KeyboardInterrupt:
            print("\n\nStopping bot...")
            self.cancel_all_orders()
            self.stop()
            print("\nFinal state:")
            for prod, pos in self.get_positions().items():
                print("  %s: %+d" % (prod, pos))
            try:
                pnl = self.get_pnl()
                if isinstance(pnl, dict):
                    print("  Total PnL: %+.0f" % pnl.get("totalProfit", 0))
                    for d in pnl.get("details", []):
                        print("    %s: %+.0f" % (d.get("product", "?"), d.get("profit", 0)))
                else:
                    print("  PnL: %s" % pnl)
            except Exception as e:
                print("  PnL error: %s" % e)
            print("Bot stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  WX_SPOT Bot v2 — Multi-Model Weather Trading")
    print("=" * 60)

    # Preview
    print("\nHours to settlement: %.1f" % hours_to_settlement())
    print("Computing multi-model fair value...\n")
    est = compute_multi_model_fair_value()
    print("\n  >>> Fair Value: %.0f" % est.fair_value)
    print("  >>> Confidence: %.2f" % est.confidence)
    print("  >>> Model spread: %.0f" % est.model_spread)
    print("  >>> Model std: %.0f" % est.model_std)

    print("\nStarting bot...\n")
    bot = WxSpotBot(EXCHANGE_URL, USERNAME, PASSWORD)
    bot.run_loop()
