"""Toolbox for the rates market-maker simulation.

Pricing and DV01 of a bond, rate simulation, quoting rules (fixed spread and
inventory skew), a probabilistic client model, the strategy simulator and the
summary statistics. The experiments themselves live in the notebook.

Conventions: prices are per 100 of face value; position = face value held
(positive = long); DV01 = loss in currency units when rates rise by 1 bp.
"""
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Bond pricing and DV01
# --------------------------------------------------------------------------- #

def bond_price(market_rate, nominal, coupon, maturity):
    """Price of a bond paying annual coupons.

    market_rate : yield as a decimal (0.04 = 4%); a number or an array of any shape
    nominal     : face value repaid at maturity (e.g. 100)
    coupon      : annual coupon rate as a decimal
    maturity    : remaining years to maturity (integer)

    Returns the price, with the same shape as market_rate.
    """
    market_rate = np.asarray(market_rate, dtype=float)

    years = np.arange(1, maturity + 1)
    cash_flows = np.full(maturity, coupon * nominal)
    cash_flows[-1] += nominal                       # face value repaid with the last coupon

    rate = market_rate[..., None]                   # extra axis to cross rates with years
    return (cash_flows / (1 + rate) ** years).sum(axis=-1)


def bond_price_loop(market_rate, nominal, coupon, maturity):
    """Slow reference implementation, used to test bond_price."""
    result = 0
    for i in range(1, maturity + 1):
        result += coupon * nominal / (1 + market_rate) ** i
    result += nominal / (1 + market_rate) ** maturity
    return result


def calculation_DV01(market_rate, nominal, coupon, maturity, bump=1e-4):
    """DV01 per `nominal` of face value: price loss when the yield rises by `bump` (1 bp).

    Returns a positive number, with the same shape as market_rate.
    """
    market_rate = np.asarray(market_rate, dtype=float)
    price_now = bond_price(market_rate, nominal, coupon, maturity)
    price_bumped = bond_price(market_rate + bump, nominal, coupon, maturity)
    return price_now - price_bumped


def position_dv01(position, dv01_per_100):
    """DV01 of a position, in currency units per bp (positive when long)."""
    position = np.asarray(position, dtype=float)
    dv01_per_100 = np.asarray(dv01_per_100, dtype=float)
    return position / 100 * dv01_per_100


# --------------------------------------------------------------------------- #
# Market simulation
# --------------------------------------------------------------------------- #

def simulate_rates(initial_rate, n_paths, n_steps, sigma_bp, seed=None):
    """Rate paths as an arithmetic random walk: rate[t+1] = rate[t] + sigma * epsilon.

    sigma_bp : standard deviation of one step, in basis points.
    Returns an array of shape (n_paths, n_steps); column 0 equals initial_rate.
    """
    sigma = sigma_bp * 1e-4
    rng = np.random.default_rng(seed)

    shocks = rng.normal(0, sigma, size=(n_paths, n_steps))
    shocks[:, 0] = 0                                # the path starts at initial_rate
    return initial_rate + shocks.cumsum(axis=1)


def draw_client_uniforms(n_paths, n_steps, seed=None):
    """Uniform draws deciding whether a client trades: a trade occurs if u < p.

    Returns (u_buy, u_sell), independent arrays of shape (n_paths, n_steps):
    u_buy drives client purchases (at our ask), u_sell client sales (at our bid).
    Draw them once and share them across the strategies being compared
    (common random numbers).
    """
    rng = np.random.default_rng(seed)
    u_buy = rng.random(size=(n_paths, n_steps))
    u_sell = rng.random(size=(n_paths, n_steps))
    return u_buy, u_sell


def simulate_market(params, seed_rates, seed_clients):
    """Rates, fair prices, DV01 per 100 and client draws for one simulation."""
    rates = simulate_rates(params["initial_rate"], params["n_paths"], params["n_steps"],
                           params["sigma_bp"], seed_rates)
    prices = bond_price(rates, params["nominal"], params["coupon"], params["maturity"])
    dv01_per_100 = calculation_DV01(rates, params["nominal"], params["coupon"], params["maturity"])
    u_buy, u_sell = draw_client_uniforms(params["n_paths"], params["n_steps"], seed_clients)
    return {"rates": rates, "prices": prices, "dv01_per_100": dv01_per_100,
            "u_buy": u_buy, "u_sell": u_sell}


# --------------------------------------------------------------------------- #
# Quotes and clients
# --------------------------------------------------------------------------- #

def quote_fixed_spread(fair_price, half_spread):
    """Bid and ask placed symmetrically around the fair price (strategy A)."""
    fair_price = np.asarray(fair_price, dtype=float)
    return fair_price - half_spread, fair_price + half_spread


def quote_skewed(fair_price, half_spread, shift):
    """Bid and ask shifted together by `shift` (price points per 100) to steer inventory.

    shift > 0 when we are long: both quotes move down, which attracts buyers and
    repels sellers. shift < 0 when short. The spread stays 2 * half_spread.
    """
    fair_price = np.asarray(fair_price, dtype=float)
    shift = np.asarray(shift, dtype=float)
    return fair_price - half_spread - shift, fair_price + half_spread - shift


def client_probabilities(fair_price, bid, ask, flow, sensitivity):
    """Probability that a client buys from us / sells to us at each step.

        p = min(1, flow * exp(-sensitivity * gap))

    where gap is how much worse than the fair price the quote is for the client
    (ask - fair on the buy side, fair - bid on the sell side). A negative gap
    (quote better than fair) pushes p above `flow`, up to 1.

    Returns (p_buy, p_sell), with the same shape as fair_price.
    """
    fair_price = np.asarray(fair_price, dtype=float)
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)

    gap_buy = ask - fair_price
    gap_sell = fair_price - bid
    p_buy = np.minimum(flow * np.exp(-sensitivity * gap_buy), 1.0)
    p_sell = np.minimum(flow * np.exp(-sensitivity * gap_sell), 1.0)
    return p_buy, p_sell


# --------------------------------------------------------------------------- #
# Strategy simulator
# --------------------------------------------------------------------------- #

def run_strategy(fair_price, u_buy, u_sell, half_spread, flow, sensitivity, dv01_per_100,
                 trade_size=1_000_000, skew=0.0, dv01_limit=None):
    """Simulate a market maker on many paths at once, step by step.

    fair_price, u_buy, u_sell, dv01_per_100 : arrays of shape (n_paths, n_steps)
    skew       : shift applied to both quotes per 1,000 of position DV01, in price
                 points per 100 (0 = fixed spread, i.e. strategy A). Long -> quotes
                 move down, short -> up.
    dv01_limit : hard limit on the absolute position DV01 (None = no limit). A trade
                 that would push the position beyond the limit is refused; a trade
                 that reduces the risk is always accepted. The limit is checked at
                 trade time only.

    At each step at most one client buys (at our ask) and one sells (at our bid).
    P&L is marked to market: cash + position / 100 * fair price.

    Returns a dict of arrays of shape (n_paths, n_steps), recorded after the trades
    of each step: 'position', 'cash', 'pnl', 'bid', 'ask'.
    """
    fair_price = np.asarray(fair_price, dtype=float)
    dv01_per_100 = np.asarray(dv01_per_100, dtype=float)
    n_paths, n_steps = fair_price.shape

    position = np.zeros(n_paths)
    cash = np.zeros(n_paths)

    position_hist = np.zeros((n_paths, n_steps))
    cash_hist = np.zeros((n_paths, n_steps))
    pnl_hist = np.zeros((n_paths, n_steps))
    bid_hist = np.zeros((n_paths, n_steps))
    ask_hist = np.zeros((n_paths, n_steps))

    for t in range(n_steps):
        fair = fair_price[:, t]

        current_dv01 = position / 100 * dv01_per_100[:, t]      # risk before this step's trades
        shift = skew * current_dv01 / 1000
        bid, ask = quote_skewed(fair, half_spread, shift)
        p_buy, p_sell = client_probabilities(fair, bid, ask, flow, sensitivity)

        client_buys = u_buy[:, t] < p_buy           # a client buys from us (we sell)
        client_sells = u_sell[:, t] < p_sell        # a client sells to us (we buy)

        if dv01_limit is not None:
            dv01_trade = trade_size / 100 * dv01_per_100[:, t]
            client_sells = client_sells & (current_dv01 + dv01_trade <= dv01_limit)
            client_buys = client_buys & (current_dv01 - dv01_trade >= -dv01_limit)

        position -= trade_size * client_buys
        cash += ask * trade_size / 100 * client_buys
        position += trade_size * client_sells
        cash -= bid * trade_size / 100 * client_sells

        position_hist[:, t] = position
        cash_hist[:, t] = cash
        pnl_hist[:, t] = cash + position / 100 * fair
        bid_hist[:, t] = bid
        ask_hist[:, t] = ask

    return {"position": position_hist, "cash": cash_hist, "pnl": pnl_hist,
            "bid": bid_hist, "ask": ask_hist}


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def summarise(results, dv01_position):
    """One value per path: final P&L, P&L volatility, max and mean absolute DV01."""
    pnl = results["pnl"]
    abs_dv01 = np.abs(dv01_position)
    return {
        "final_pnl": pnl[:, -1],
        "pnl_vol": np.diff(pnl, axis=1).std(axis=1),
        "max_abs_dv01": abs_dv01.max(axis=1),
        "mean_abs_dv01": abs_dv01.mean(axis=1),
    }


def describe_summary(summary):
    """Key statistics of a strategy over all paths, from the output of summarise()."""
    final = summary["final_pnl"]
    max_dv01 = summary["max_abs_dv01"]
    return pd.Series({
        "mean final P&L": final.mean(),
        "std of final P&L": final.std(),
        "5th percentile of final P&L": np.percentile(final, 5),
        "share of losing paths": (final < 0).mean(),
        "mean of max |DV01|": max_dv01.mean(),
        "95th percentile of max |DV01|": np.percentile(max_dv01, 95),
        "mean of mean |DV01|": summary["mean_abs_dv01"].mean(),
    })


SHORT_NAMES = {
    "mean final P&L": "mean P&L",
    "std of final P&L": "std P&L",
    "5th percentile of final P&L": "5th pct P&L",
    "share of losing paths": "% losing",
    "mean of max |DV01|": "mean max DV01",
    "95th percentile of max |DV01|": "95th pct max DV01",
    "mean of mean |DV01|": "mean DV01",
}


def evaluate_variants(variants, params, seed_pairs):
    """Run several strategy variants on independent simulations.

    variants   : {name: dict(skew=..., dv01_limit=...)}
    params     : dict of market and strategy parameters (see the notebook)
    seed_pairs : list of (seed_rates, seed_clients), one per simulation

    Within a simulation, all variants see the same rates and the same client draws.
    Returns a long DataFrame with one row per (simulation, variant) and the
    statistics of describe_summary (long column names).
    """
    rows = []
    for i, (seed_r, seed_c) in enumerate(seed_pairs):
        m = simulate_market(params, seed_r, seed_c)
        for name, kwargs in variants.items():
            res = run_strategy(m["prices"], m["u_buy"], m["u_sell"], params["half_spread"],
                               params["flow"], params["sensitivity"], m["dv01_per_100"],
                               trade_size=params["trade_size"], **kwargs)
            stats = describe_summary(summarise(res, position_dv01(res["position"], m["dv01_per_100"])))
            rows.append({"simulation": i, "variant": name, **stats})
    return pd.DataFrame(rows)


def average_over_simulations(table):
    """Mean and std across simulations, one row per variant, short column names, % losing in %."""
    grouped = (table.drop(columns="simulation").rename(columns=SHORT_NAMES)
               .groupby("variant", sort=False))
    mean, std = grouped.mean(), grouped.std()
    for t in (mean, std):
        t["% losing"] *= 100
    return mean, std
