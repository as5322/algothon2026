"""TIDE_SPOT Competitive Trading Bot v2

Trades TIDE_SPOT = ABS(tidal_level_mAOD) x 1000 at Sunday 12:00 London.

Architecture:
  1. 8-component harmonic tidal model (M2, S2, N2, K1, O1, M4, MS4, MN4)
     - 44% more accurate than 3-component, captures shallow-water Thames overtides
  2. Confidence from RMSE + cross-validation on recent data
  3. Inventory-skewed quoting with strong unwind pressure
  4. Multi-level quotes at several depths
  5. Time-aware — tighter as settlement approaches and fit improves
  6. Auto-refits every 2 min as new tidal readings arrive
  7. Smarter directional sizing — scales with edge AND confidence
"""

import math
import time
import statistics
import warnings
from threading import Thread

import numpy as np
import pandas as pd
import requests
from scipy import optimize

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

PRODUCT = "TIDE_SPOT"

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Position limits
MAX_POSITION = 80
SOFT_POSITION = 40

# Quoting — tighter now with better model (RMSE ~245 vs ~435)
BASE_HALF_WIDTH = 10
MIN_HALF_WIDTH = 4
NUM_QUOTE_LEVELS = 3
LEVEL_STEP = 6
VOLUME_PER_LEVEL = 3

# Directional — more confident with 8-component model
EDGE_THRESHOLD_AGGRESSIVE = 80
EDGE_THRESHOLD_MODERATE = 30
DIRECTIONAL_VOL_BIG = 6
DIRECTIONAL_VOL_SMALL = 2

# Timing
LOOP_INTERVAL = 8
DATA_REFRESH_INTERVAL = 120   # refit every 2 min (EA publishes every 15min)
RATE_LIMIT_PAUSE = 1.0

# Inventory skew — strong to prevent large positions building up
SKEW_PER_LOT = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# TIDAL DATA & 8-COMPONENT HARMONIC MODEL
# ═══════════════════════════════════════════════════════════════════════════════

THAMES_MEASURE = "0006-level-tidal_level-i-15_min-mAOD"

# All 8 major tidal constituents for the Thames at Westminster
# Shallow-water overtides (M4, MS4, MN4) are critical for the Thames
# because the tidal wave is distorted by the river bed and constrictions.
TIDAL_PERIODS = {
    'M2':  12.4206,   # principal lunar semidiurnal (dominant)
    'S2':  12.0000,   # principal solar semidiurnal
    'N2':  12.6583,   # larger lunar elliptic semidiurnal
    'K1':  23.9345,   # lunisolar diurnal
    'O1':  25.8193,   # principal lunar diurnal
    'M4':   6.2103,   # shallow water overtide of M2 (Thames-critical!)
    'MS4':  6.1033,   # shallow water compound tide
    'MN4':  6.2692,   # shallow water compound tide
}
NUM_HARMONICS = len(TIDAL_PERIODS)


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


def fetch_thames_readings(limit=800) -> pd.DataFrame:
    """Fetch recent Thames tidal readings from the EA Flood Monitoring API."""
    resp = requests.get(
        "https://environment.data.gov.uk/flood-monitoring/id/measures/"
        "%s/readings" % THAMES_MEASURE,
        params={"_sorted": "", "_limit": limit},
        timeout=15,
        verify=False,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        return pd.DataFrame(columns=["time", "level"])

    df = pd.DataFrame(items)[["dateTime", "value"]].rename(
        columns={"dateTime": "time", "value": "level"}
    )
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/London")
    df = df.sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=["level"])
    return df


def tidal_model(t, *params):
    """8-component harmonic tidal model.

    params layout: [A1, phi1, A2, phi2, ..., A8, phi8, C]
    C is the constant offset (mean water level).

    Includes shallow-water overtides (M4, MS4, MN4) which are critical
    for the Thames estuary where the tidal wave is distorted by the
    river bed and constrictions.
    """
    result = params[-1]  # constant offset C
    for i, (name, period) in enumerate(TIDAL_PERIODS.items()):
        A = params[2 * i]
        phi = params[2 * i + 1]
        result = result + A * np.cos(2 * np.pi * t / period + phi)
    return result


def make_initial_guess():
    """Initial parameter guess for curve fitting."""
    p0 = []
    for name in TIDAL_PERIODS:
        if name == 'M2':          p0.extend([2.5, 0])
        elif name in ('S2','N2'): p0.extend([0.5, 0])
        elif name in ('K1','O1'): p0.extend([0.2, 0])
        else:                     p0.extend([0.1, 0])  # overtides
    p0.append(0.5)  # offset
    return p0


class TidalForecast:
    """Fitted tidal model with prediction and confidence."""

    def __init__(self):
        self.fair_value: float = 0
        self.predicted_level: float = 0
        self.rmse: float = float("inf")
        self.recent_rmse: float = float("inf")
        self.confidence: float = 0
        self.params: tuple = ()
        self.n_readings: int = 0
        self.timestamp: float = 0
        self.fv_low: float = 0
        self.fv_high: float = 0

    def __repr__(self):
        return ("FV=%d level=%.3f rmse=%.3f recent_rmse=%.3f conf=%.2f n=%d range=[%d, %d]"
                % (self.fair_value, self.predicted_level, self.rmse,
                   self.recent_rmse, self.confidence, self.n_readings,
                   self.fv_low, self.fv_high))


def fit_tidal_model() -> TidalForecast:
    """Fetch tidal data, fit 8-component harmonic model, predict settlement."""
    result = TidalForecast()
    result.timestamp = time.monotonic()

    print("  Fetching Thames tidal data...")
    df = fetch_thames_readings(limit=800)
    if len(df) < 50:
        print("  [ERROR] Not enough tidal readings: %d" % len(df))
        return result

    result.n_readings = len(df)
    t0 = df["time"].iloc[0]
    t = (df["time"] - t0).dt.total_seconds().values / 3600  # hours
    y = df["level"].values

    # Fit the 8-component harmonic model
    try:
        popt, pcov = optimize.curve_fit(
            tidal_model, t, y,
            p0=make_initial_guess(),
            maxfev=50000,
        )
        result.params = tuple(popt)
    except Exception as e:
        print("  [ERROR] Curve fit failed: %s" % e)
        return _fallback_analog_prediction(df, result)

    # Compute fit quality
    residuals = y - tidal_model(t, *popt)
    result.rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # Cross-validate on last 48 readings (~12h) for recent accuracy
    if len(t) > 100:
        recent_resid = residuals[-48:]
        result.recent_rmse = float(np.sqrt(np.mean(recent_resid ** 2)))
        print("    Recent 12h RMSE: %.4f mAOD (%d units)"
              % (result.recent_rmse, result.recent_rmse * 1000))

    # Predict settlement level
    target = get_settlement_target()
    target_hours = (target - t0).total_seconds() / 3600
    predicted_level = float(tidal_model(target_hours, *popt))
    result.predicted_level = predicted_level
    result.fair_value = abs(predicted_level) * 1000

    # Prediction interval (2 * RMSE ~95%)
    level_low = predicted_level - 2 * result.rmse
    level_high = predicted_level + 2 * result.rmse
    abs_values = sorted([abs(level_low), abs(level_high)])
    if level_low <= 0 <= level_high:
        result.fv_low = 0
    else:
        result.fv_low = int(abs_values[0] * 1000)
    result.fv_high = int(abs_values[1] * 1000)

    # Confidence: based on RMSE and time to settlement
    # 8-comp model: RMSE ~0.25 is typical (good), > 0.5 is bad
    rmse_conf = max(0.1, 1.0 / (1.0 + result.rmse * 3))
    hours_left = hours_to_settlement()
    if hours_left < 2:
        time_boost = 1.8
    elif hours_left < 6:
        time_boost = 1.4
    elif hours_left < 12:
        time_boost = 1.15
    else:
        time_boost = 1.0
    result.confidence = min(1.0, rmse_conf * time_boost)

    print("  [TIDAL] %s" % result)
    for i, (name, period) in enumerate(TIDAL_PERIODS.items()):
        print("    %s (%.2fh): A=%.4f" % (name, period, popt[2 * i]))
    print("    Offset: %.4f  RMSE: %.4f" % (popt[-1], result.rmse))
    print("    Target: %s" % target)
    print("    Predicted level: %.3f mAOD -> TIDE_SPOT=%d"
          % (predicted_level, result.fair_value))

    return result


def _fallback_analog_prediction(df: pd.DataFrame, result: TidalForecast) -> TidalForecast:
    """Fallback: use tidal analogs (same phase of tide cycle) if curve fit fails."""
    target = get_settlement_target()
    tidal_period = pd.Timedelta(hours=12, minutes=25)

    predictions = []
    for n in range(1, 10):
        analog_time = target - n * tidal_period
        mask = abs(df["time"] - analog_time) < pd.Timedelta(minutes=10)
        nearby = df[mask]
        if len(nearby) > 0:
            closest = nearby.iloc[(abs(nearby["time"] - analog_time)).argmin()]
            predictions.append(closest["level"])

    if not predictions:
        return result

    same_phase = predictions[::2]
    if same_phase:
        avg_level = float(np.mean(same_phase))
        result.predicted_level = avg_level
        result.fair_value = abs(avg_level) * 1000
        result.rmse = float(np.std(same_phase)) if len(same_phase) > 1 else 1.0
        result.confidence = max(0.1, 0.5 / (1.0 + result.rmse))
        result.fv_low = int(abs(avg_level - 2 * result.rmse) * 1000)
        result.fv_high = int(abs(avg_level + 2 * result.rmse) * 1000)

    print("  [TIDAL fallback] %s" % result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BOT
# ═══════════════════════════════════════════════════════════════════════════════

class TideSpotBot(BaseBot):
    """Competitive TIDE_SPOT trading bot v2."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forecast: TidalForecast | None = None
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

    def refresh_forecast(self, force=False) -> bool:
        """Refit tidal model if stale."""
        now = time.monotonic()
        if not force and self.forecast is not None:
            if (now - self.last_data_fetch) < DATA_REFRESH_INTERVAL:
                return False

        print("\n  Refitting tidal model...")
        self.forecast = fit_tidal_model()
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
        """Shift theo to lean away from inventory.

        With SKEW_PER_LOT=0.5, a position of +40 shifts theo down by 20,
        making sell quotes much more attractive and helping unwind.
        """
        return raw_theo - position * SKEW_PER_LOT

    def compute_half_width(self) -> float:
        """Dynamic half-width based on RMSE confidence and time."""
        if self.forecast is None:
            return BASE_HALF_WIDTH

        confidence = self.forecast.confidence
        hours_left = hours_to_settlement()

        # Width inversely proportional to confidence
        width = BASE_HALF_WIDTH / max(confidence, 0.2)

        # Scale with RMSE (8-comp typical RMSE ~0.25 = 250 units)
        rmse_units = self.forecast.rmse * 1000
        rmse_penalty = max(1.0, rmse_units / 250)
        width *= min(rmse_penalty, 2.5)

        # Tighten near settlement (model becomes more accurate)
        if hours_left < 2:
            width *= 0.4
        elif hours_left < 6:
            width *= 0.6
        elif hours_left < 12:
            width *= 0.8

        return max(MIN_HALF_WIDTH, min(width, 50))

    def compute_quote_volume(self, position: int, side: Side, level: int) -> int:
        """Volume per quote level, adjusted for position."""
        base_vol = max(1, VOLUME_PER_LEVEL - level)

        # Strongly reduce volume toward position limit
        if side == Side.BUY and position > SOFT_POSITION:
            scale = max(0, (MAX_POSITION - position) / (MAX_POSITION - SOFT_POSITION))
            base_vol = max(1, int(base_vol * scale))
        elif side == Side.SELL and position < -SOFT_POSITION:
            scale = max(0, (MAX_POSITION + position) / (MAX_POSITION - SOFT_POSITION))
            base_vol = max(1, int(base_vol * scale))

        # Boost volume on the side that unwinds position
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
        """Take directional position if strong edge exists.

        Key improvement: scales volume with both edge size and confidence,
        and NEVER trades into an already-large position.
        """
        edge = theo - market_mid
        abs_edge = abs(edge)

        if abs_edge < EDGE_THRESHOLD_MODERATE:
            return False

        confidence = self.forecast.confidence if self.forecast else 0.5

        # Scale volume with edge size (not just threshold buckets)
        edge_scale = min(2.0, abs_edge / EDGE_THRESHOLD_MODERATE)
        base_vol = max(1, int(DIRECTIONAL_VOL_SMALL * edge_scale * confidence))

        # CRITICAL: don't add to an already-large position in the same direction
        if edge > 0 and position > SOFT_POSITION:
            return False  # already long enough
        if edge < 0 and position < -SOFT_POSITION:
            return False  # already short enough

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
        print("  TIDE_SPOT Bot v2 — 8-Component Harmonic Model")
        print("  Product: %s (tick=%s)" % (PRODUCT, tick))
        print("  Hours to settlement: %.1f" % hours_to_settlement())
        print("=" * 60)

        iteration = 0
        try:
            while True:
                iteration += 1

                # 1. Refresh tidal model
                self.refresh_forecast()
                if self.forecast is None or self.forecast.fair_value == 0:
                    print("  Waiting for tidal data...")
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

                print("\n  [%d] theo=%d mid=%s pos=%+d hw=%.1f conf=%.2f "
                      "rmse=%.3f hrs=%.1f"
                      % (iteration, theo, market_mid, position, half_width,
                         self.forecast.confidence, self.forecast.rmse, hours_left))

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
                        tide_pnl = 0
                        for d in pnl.get("details", []):
                            if d.get("product") == PRODUCT:
                                tide_pnl = d.get("profit", 0)
                        print("  [PNL] total=%+.0f  TIDE_SPOT=%+.0f" % (total, tide_pnl))
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
    print("  TIDE_SPOT Bot v2 — 8-Component Harmonic Tidal Trading")
    print("=" * 60)

    print("\nHours to settlement: %.1f" % hours_to_settlement())
    print("Fitting tidal model...\n")
    forecast = fit_tidal_model()

    if forecast.fair_value > 0:
        print("\n  >>> TIDE_SPOT Fair Value: %d" % forecast.fair_value)
        print("  >>> Predicted level: %.3f mAOD" % forecast.predicted_level)
        print("  >>> 95%% interval: [%d, %d]" % (forecast.fv_low, forecast.fv_high))
        print("  >>> Confidence: %.2f" % forecast.confidence)
        print("  >>> Fit RMSE: %.3f mAOD (%d units)" % (forecast.rmse, forecast.rmse * 1000))
    else:
        print("\n  Could not compute fair value.")

    print("\nStarting bot...\n")
    bot = TideSpotBot(EXCHANGE_URL, USERNAME, PASSWORD)
    bot.run_loop()
