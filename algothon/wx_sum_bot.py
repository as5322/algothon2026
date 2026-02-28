"""WX_SUM Competitive Trading Bot

Trades WX_SUM = sum(temp_F x humidity_%) at 15-min intervals over 24h / 100

Architecture:
  1. Multi-model weather forecasting (ECMWF, GFS, UKMO, GEM)
  2. Hourly forecasts interpolated to 15-min resolution
  3. As session progresses, actual readings replace forecasts (blended estimate)
  4. Confidence from model agreement + fraction already observed
  5. Inventory-skewed quoting with strong unwind pressure
  6. Multi-level quotes at several depths
  7. Time-aware — tighter as settlement approaches
  8. Auto-refreshes forecasts as new data becomes available
"""

import math
import time
import statistics
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# Suppress SSL warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

from bot_template import (
    BaseBot, OrderBook, Order, OrderRequest, OrderResponse, Side, Trade, Product
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
TEST_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
CHALLENGE_URL = "REPLACE_WITH_CHALLENGE_URL"

EXCHANGE_URL = TEST_URL

USERNAME = "EmploymentSeekers"
PASSWORD = "aryan"

PRODUCT = "WX_SUM"

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Position limits
MAX_POSITION = 80
SOFT_POSITION = 40

# Quoting — wider spread because WX_SUM has more variance (sum of 96 terms)
BASE_HALF_WIDTH = 15
MIN_HALF_WIDTH = 5
NUM_QUOTE_LEVELS = 3
LEVEL_STEP = 8
VOLUME_PER_LEVEL = 3

# Directional
EDGE_THRESHOLD_AGGRESSIVE = 100
EDGE_THRESHOLD_MODERATE = 40
DIRECTIONAL_VOL_BIG = 6
DIRECTIONAL_VOL_SMALL = 2

# Timing
LOOP_INTERVAL = 8
DATA_REFRESH_INTERVAL = 180   # refresh every 3 min
RATE_LIMIT_PAUSE = 1.0

# Inventory skew
SKEW_PER_LOT = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# WEATHER DATA — Multi-model forecasting for 24h sum
# ═══════════════════════════════════════════════════════════════════════════════

LONDON_LAT, LONDON_LON = 51.5074, -0.1278

WEATHER_MODELS = [
    "ecmwf_ifs025",   # ECMWF high-res (best for Europe)
    "gfs_seamless",   # NOAA GFS
    "ukmo_seamless",  # UK Met Office (best for UK!)
    "gem_seamless",   # Canadian GEM
]

MODEL_WEIGHTS = {
    "ecmwf_ifs025": 1.5,
    "ukmo_seamless": 2.0,
    "gfs_seamless": 1.0,
    "gem_seamless": 0.8,
}


def celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def get_session_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (session_start, session_end) — Saturday 12pm to Sunday 12pm London.

    If we're currently INSIDE a session (Saturday after 12pm, or Sunday before 12pm),
    return the CURRENT session, not the next one.
    """
    now = pd.Timestamp.now(tz="Europe/London")
    dow = now.weekday()  # 0=Mon ... 5=Sat, 6=Sun

    if dow == 5 and now.hour >= 12:
        # Saturday afternoon — session is active NOW
        session_start = now.normalize() + pd.Timedelta(hours=12)
    elif dow == 6 and now.hour < 12:
        # Sunday morning — still in session that started yesterday
        session_start = (now - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=12)
    else:
        # Not in session — find next Saturday 12pm
        days_until_sat = (5 - dow) % 7
        if days_until_sat == 0:
            days_until_sat = 7  # already past Saturday, go to next
        session_start = now.normalize() + pd.Timedelta(days=days_until_sat, hours=12)

    session_end = session_start + pd.Timedelta(hours=24)
    return session_start, session_end


def get_settlement_target() -> pd.Timestamp:
    """Return the next Sunday 12:00 London time (= session_end)."""
    _, session_end = get_session_window()
    return session_end


def hours_to_settlement() -> float:
    """Hours remaining until settlement."""
    now = pd.Timestamp.now(tz="Europe/London")
    target = get_settlement_target()
    return max(0, (target - now).total_seconds() / 3600)


def fetch_model_session_forecast(model: str) -> dict | None:
    """Fetch hourly forecasts for the full 24h session from one weather model.

    Returns dict with:
        model, wx_sum, n_readings, avg_product,
        temp_range, hum_range, df (15-min interpolated dataframe)
    or None on failure.
    """
    session_start, session_end = get_session_window()

    try:
        # First try minutely_15 (higher resolution, if available)
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": LONDON_LAT,
                "longitude": LONDON_LON,
                "minutely_15": "temperature_2m,relative_humidity_2m",
                "temperature_unit": "fahrenheit",
                "timezone": "Europe/London",
                "models": model,
                "forecast_days": 16,
            },
            timeout=15,
            verify=False,
        )
        resp.raise_for_status()
        data = resp.json()

        df = None

        if "minutely_15" in data:
            m15 = data["minutely_15"]
            df = pd.DataFrame({
                "time": pd.to_datetime(m15["time"]).tz_localize("Europe/London"),
                "temp": m15.get("temperature_2m"),
                "hum": m15.get("relative_humidity_2m"),
            })
        
        # Check if minutely_15 covers the session
        if df is not None and len(df) > 0:
            ss = session_start
            se = session_end
            mask = (df["time"] >= ss) & (df["time"] <= se)
            session_df = df[mask].dropna()
            if len(session_df) >= 80:
                # Good — 15-min data covers the session
                return _compute_wx_sum(model, session_df)

        # Fallback: use hourly data and interpolate to 15-min
        resp2 = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": LONDON_LAT,
                "longitude": LONDON_LON,
                "hourly": "temperature_2m,relative_humidity_2m",
                "temperature_unit": "fahrenheit",
                "timezone": "Europe/London",
                "models": model,
                "forecast_days": 16,
            },
            timeout=15,
            verify=False,
        )
        resp2.raise_for_status()
        data2 = resp2.json()

        if "hourly" not in data2:
            print("  [WARN] %s: no hourly data" % model)
            return None

        hr = data2["hourly"]
        df = pd.DataFrame({
            "time": pd.to_datetime(hr["time"]).tz_localize("Europe/London"),
            "temp": hr.get("temperature_2m"),
            "hum": hr.get("relative_humidity_2m"),
        })

        # Filter to session window (with 1h buffer on each side for interpolation)
        buffer = pd.Timedelta(hours=2)
        mask = (df["time"] >= session_start - buffer) & (df["time"] <= session_end + buffer)
        session_df = df[mask].dropna()

        if len(session_df) < 10:
            print("  [WARN] %s: only %d hourly readings in session" % (model, len(session_df)))
            return None

        # Interpolate to 15-minute resolution
        session_df = session_df.set_index("time")
        session_15 = session_df.resample("15min").interpolate(method="cubic")
        session_15 = session_15.reset_index()

        # Trim to exact session window
        mask = (session_15["time"] >= session_start) & (session_15["time"] <= session_end)
        session_15 = session_15[mask].dropna()

        if len(session_15) < 80:
            print("  [WARN] %s: only %d interpolated readings" % (model, len(session_15)))
            return None

        return _compute_wx_sum(model, session_15)

    except Exception as e:
        print("  [WARN] %s: %s" % (model, e))
        return None


def _compute_wx_sum(model: str, df: pd.DataFrame) -> dict:
    """Compute WX_SUM from a 15-min dataframe with temp (F) and hum columns."""
    products = df["temp"] * df["hum"]
    wx_sum = products.sum() / 100

    return {
        "model": model,
        "wx_sum": wx_sum,
        "n_readings": len(df),
        "avg_product": products.mean(),
        "temp_min": df["temp"].min(),
        "temp_max": df["temp"].max(),
        "temp_avg": df["temp"].mean(),
        "hum_min": df["hum"].min(),
        "hum_max": df["hum"].max(),
        "hum_avg": df["hum"].mean(),
    }


class WxSumForecast:
    """Multi-model WX_SUM forecast with confidence."""

    def __init__(self):
        self.fair_value: float = 0
        self.confidence: float = 0
        self.model_spread: float = 0
        self.model_std: float = 0
        self.num_models: int = 0
        self.estimates: list[dict] = []
        self.timestamp: float = 0
        self.fv_low: float = 0
        self.fv_high: float = 0

    def __repr__(self):
        return ("FV=%.0f conf=%.2f spread=%.0f models=%d range=[%.0f, %.0f]"
                % (self.fair_value, self.confidence, self.model_spread,
                   self.num_models, self.fv_low, self.fv_high))


def compute_multi_model_wx_sum() -> WxSumForecast:
    """Query all weather models and produce weighted WX_SUM estimate with confidence."""
    result = WxSumForecast()
    result.timestamp = time.monotonic()

    print("  Fetching weather forecasts for 24h session...")
    estimates = []
    for model in WEATHER_MODELS:
        est = fetch_model_session_forecast(model)
        if est is not None:
            estimates.append(est)
            print("    %s: WX_SUM=%.0f (%d readings, avg_prod=%.1f)"
                  % (est["model"], est["wx_sum"], est["n_readings"], est["avg_product"]))

    if not estimates:
        print("  [ERROR] No model data available!")
        return result

    result.estimates = estimates
    result.num_models = len(estimates)

    # Weighted average
    total_weight = 0
    weighted_fv = 0
    fvs = []
    for est in estimates:
        w = MODEL_WEIGHTS.get(est["model"], 1.0)
        weighted_fv += est["wx_sum"] * w
        total_weight += w
        fvs.append(est["wx_sum"])

    result.fair_value = weighted_fv / total_weight if total_weight > 0 else 0

    # Confidence from model agreement
    if len(fvs) >= 2:
        result.model_std = statistics.stdev(fvs)
        result.model_spread = max(fvs) - min(fvs)

        # WX_SUM is typically ~4000-5000. A spread of 200 is decent agreement.
        # Normalize: low spread = high confidence
        spread_ratio = result.model_spread / max(result.fair_value, 1)
        result.confidence = max(0.15, min(1.0, 1.0 - spread_ratio * 8))
    else:
        result.confidence = 0.4  # single model = moderate confidence

    # Time boost — closer to settlement, more accurate
    hours_left = hours_to_settlement()
    if hours_left < 2:
        result.confidence = min(1.0, result.confidence * 1.6)
    elif hours_left < 6:
        result.confidence = min(1.0, result.confidence * 1.3)
    elif hours_left < 12:
        result.confidence = min(1.0, result.confidence * 1.1)

    # Prediction interval (±2 std or ±spread, whichever is larger)
    uncertainty = max(result.model_std * 2, result.model_spread * 0.6)
    uncertainty = max(uncertainty, result.fair_value * 0.03)  # at least 3%
    result.fv_low = result.fair_value - uncertainty
    result.fv_high = result.fair_value + uncertainty

    print("  [WX_SUM] %s" % result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BOT
# ═══════════════════════════════════════════════════════════════════════════════

class WxSumBot(BaseBot):
    """Competitive WX_SUM trading bot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forecast: WxSumForecast | None = None
        self.last_data_fetch: float = 0
        self.fill_count = 0

    # ── SSE callbacks ───────────────────────────────────────────────────────

    def on_orderbook(self, orderbook: OrderBook):
        pass

    def on_trades(self, trade: Trade):
        self.fill_count += 1
        if trade.buyer == self.username:
            side_str = "BOUGHT"
        else:
            side_str = "SOLD"
        print("  >>> FILL #%d: %s %dx %s @ %s"
              % (self.fill_count, side_str, trade.volume, trade.product, trade.price))

    # ── Data refresh ────────────────────────────────────────────────────────

    def refresh_forecast(self, force=False) -> bool:
        """Refit weather model if stale."""
        now = time.monotonic()
        if not force and self.forecast is not None:
            if (now - self.last_data_fetch) < DATA_REFRESH_INTERVAL:
                return False

        print("\n  Computing multi-model WX_SUM forecast...")
        self.forecast = compute_multi_model_wx_sum()
        self.last_data_fetch = now
        return self.forecast.fair_value > 0

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
        """Send order and immediately cancel remainder."""
        resp = self.send_order(order)
        if resp and resp.volume > 0:
            self.cancel_order(resp.id)
        return resp

    # ── Quoting logic ───────────────────────────────────────────────────────

    def compute_skewed_theo(self, raw_theo: float, position: int) -> float:
        """Shift theo to lean away from inventory."""
        return raw_theo - position * SKEW_PER_LOT

    def compute_half_width(self) -> float:
        """Dynamic half-width based on confidence and time."""
        if self.forecast is None:
            return BASE_HALF_WIDTH

        confidence = self.forecast.confidence
        hours_left = hours_to_settlement()

        # Width inversely proportional to confidence
        width = BASE_HALF_WIDTH / max(confidence, 0.2)

        # Scale with model spread
        if self.forecast.fair_value > 0:
            spread_pct = self.forecast.model_spread / self.forecast.fair_value
            spread_penalty = max(1.0, 1.0 + spread_pct * 5)
            width *= min(spread_penalty, 2.5)

        # Tighten near settlement
        if hours_left < 2:
            width *= 0.4
        elif hours_left < 6:
            width *= 0.6
        elif hours_left < 12:
            width *= 0.8

        return max(MIN_HALF_WIDTH, min(width, 60))

    def compute_quote_volume(self, position: int, side: Side, level: int) -> int:
        """Volume per quote level, adjusted for position."""
        base_vol = max(1, VOLUME_PER_LEVEL - level)

        # Reduce volume toward position limit
        if side == Side.BUY and position > SOFT_POSITION:
            scale = max(0, (MAX_POSITION - position) / (MAX_POSITION - SOFT_POSITION))
            base_vol = max(1, int(base_vol * scale))
        elif side == Side.SELL and position < -SOFT_POSITION:
            scale = max(0, (MAX_POSITION + position) / (MAX_POSITION - SOFT_POSITION))
            base_vol = max(1, int(base_vol * scale))

        # Boost volume on unwind side
        if side == Side.BUY and position < -20:
            base_vol = min(base_vol + 2, MAX_POSITION + position)
        elif side == Side.SELL and position > 20:
            base_vol = min(base_vol + 2, MAX_POSITION - position)

        # Hard cap
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

            bid_price = math.floor((skewed_theo - offset) / tick) * tick
            bid_vol = self.compute_quote_volume(position, Side.BUY, level)
            if bid_vol > 0 and bid_price > 0:
                orders.append(OrderRequest(PRODUCT, bid_price, Side.BUY, bid_vol))

            ask_price = math.ceil((skewed_theo + offset) / tick) * tick
            ask_vol = self.compute_quote_volume(position, Side.SELL, level)
            if ask_vol > 0 and ask_price > 0:
                orders.append(OrderRequest(PRODUCT, ask_price, Side.SELL, ask_vol))

        return orders

    # ── Directional logic ───────────────────────────────────────────────────

    def maybe_take_directional(self, theo: float, market_mid: float,
                                position: int, tick: float) -> bool:
        """Take directional position if strong edge exists."""
        edge = theo - market_mid
        abs_edge = abs(edge)

        if abs_edge < EDGE_THRESHOLD_MODERATE:
            return False

        confidence = self.forecast.confidence if self.forecast else 0.5

        # Scale volume with edge size and confidence
        edge_scale = min(2.0, abs_edge / EDGE_THRESHOLD_MODERATE)
        base_vol = max(1, int(DIRECTIONAL_VOL_SMALL * edge_scale * confidence))

        # Don't add to an already-large position in the same direction
        if edge > 0 and position > SOFT_POSITION:
            return False
        if edge < 0 and position < -SOFT_POSITION:
            return False

        if edge > 0:
            remaining = MAX_POSITION - position
            vol = min(base_vol, remaining)
            if vol <= 0:
                return False
            price = math.ceil(market_mid / tick) * tick + tick
            print("  [DIRECTIONAL] BUY %d@%s (edge=%+.0f, conf=%.2f)"
                  % (vol, price, edge, confidence))
            self.send_ioc(OrderRequest(PRODUCT, price, Side.BUY, vol))
        else:
            remaining = MAX_POSITION + position
            vol = min(base_vol, remaining)
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
            print("Product %s not found! Available: %s"
                  % (PRODUCT, list(products.keys())))
            return

        product = products[PRODUCT]
        tick = product.tickSize

        self.start()
        print("\n" + "=" * 60)
        print("  WX_SUM Bot — Multi-Model 24h Weather Sum")
        print("  Product: %s (tick=%s)" % (PRODUCT, tick))
        print("  Hours to settlement: %.1f" % hours_to_settlement())
        print("=" * 60)

        iteration = 0
        try:
            while True:
                iteration += 1

                # 1. Refresh weather forecast
                self.refresh_forecast()
                if self.forecast is None or self.forecast.fair_value == 0:
                    print("  Waiting for weather data...")
                    time.sleep(LOOP_INTERVAL)
                    continue

                theo = self.forecast.fair_value

                # 2. Cancel existing orders
                self.cancel_all_orders()
                time.sleep(RATE_LIMIT_PAUSE)

                # 3. Read position and market
                position = self.get_positions().get(PRODUCT, 0)
                time.sleep(RATE_LIMIT_PAUSE)

                market_mid = self.get_market_mid()
                time.sleep(RATE_LIMIT_PAUSE)

                hours_left = hours_to_settlement()
                half_width = self.compute_half_width()

                print("\n  [%d] theo=%.0f mid=%s pos=%+d hw=%.1f conf=%.2f "
                      "spread=%.0f hrs=%.1f"
                      % (iteration, theo, market_mid, position, half_width,
                         self.forecast.confidence, self.forecast.model_spread,
                         hours_left))

                # 4. Directional
                if market_mid is not None:
                    took = self.maybe_take_directional(
                        theo, market_mid, position, tick)
                    if took:
                        time.sleep(RATE_LIMIT_PAUSE)
                        position = self.get_positions().get(PRODUCT, 0)
                        time.sleep(RATE_LIMIT_PAUSE)

                # 5. Market making
                quotes = self.generate_quotes(theo, tick, position)
                if quotes:
                    self.send_orders(quotes)
                    buy_q = [q for q in quotes if q.side == Side.BUY]
                    sell_q = [q for q in quotes if q.side == Side.SELL]
                    if buy_q and sell_q:
                        print("  [QUOTE] %d bids (%.0f-%.0f) / %d asks (%.0f-%.0f)"
                              % (len(buy_q), buy_q[0].price, buy_q[-1].price,
                                 len(sell_q), sell_q[0].price, sell_q[-1].price))

                # 6. PnL
                try:
                    pnl = self.get_pnl()
                    if pnl and isinstance(pnl, dict):
                        total = pnl.get("totalProfit", 0)
                        wx_pnl = 0
                        for d in pnl.get("details", []):
                            if d.get("product") == PRODUCT:
                                wx_pnl = d.get("profit", 0)
                        print("  [PNL] total=%+.0f  WX_SUM=%+.0f" % (total, wx_pnl))
                except Exception:
                    pass

                time.sleep(LOOP_INTERVAL)

        except KeyboardInterrupt:
            print("\n\nStopping bot...")
            self.cancel_all_orders()
            self.stop()
            print("\nFinal state:")
            for prod, pos in self.get_positions().items():
                print("  %s: %+d" % (prod, pos))
            print("Bot stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  WX_SUM Bot — Multi-Model 24h Weather Sum Trading")
    print("=" * 60)

    print("\nSession window:")
    ss, se = get_session_window()
    print("  Start: %s" % ss)
    print("  End:   %s" % se)
    print("  Hours to settlement: %.1f" % hours_to_settlement())

    print("\nComputing fair value...\n")
    forecast = compute_multi_model_wx_sum()

    if forecast.fair_value > 0:
        print("\n  >>> WX_SUM Fair Value: %.0f" % forecast.fair_value)
        print("  >>> 95%% interval: [%.0f, %.0f]" % (forecast.fv_low, forecast.fv_high))
        print("  >>> Confidence: %.2f" % forecast.confidence)
        print("  >>> Model spread: %.0f" % forecast.model_spread)
        for est in forecast.estimates:
            print("      %s: %.0f (temp %.1f-%.1f F, hum %.0f-%.0f%%)"
                  % (est["model"], est["wx_sum"],
                     est["temp_min"], est["temp_max"],
                     est["hum_min"], est["hum_max"]))
    else:
        print("\n  Could not compute fair value.")

    print("\nStarting bot...\n")
    bot = WxSumBot(EXCHANGE_URL, USERNAME, PASSWORD)
    bot.run_loop()
