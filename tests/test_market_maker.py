import numpy as np
import pytest

from market_maker import (bond_price, bond_price_loop, calculation_DV01, position_dv01,
                          simulate_rates, draw_client_uniforms, quote_fixed_spread,
                          quote_skewed, client_probabilities, run_strategy, summarise)

DV01_100 = calculation_DV01(0.04, 100, 0.04, 10)     # ~0.0811 per 100 of face value


def test_vectorised_pricer_matches_loop():
    rates = np.linspace(0.01, 0.08, 50)
    slow = np.array([bond_price_loop(r, 100, 0.04, 10) for r in rates])
    assert np.allclose(bond_price(rates, 100, 0.04, 10), slow)


def test_par_bond_prices_at_par():
    assert bond_price(0.04, 100, 0.04, 10) == pytest.approx(100.0)


def test_dv01_values_and_shape():
    assert DV01_100 == pytest.approx(0.0811, abs=1e-4)
    assert calculation_DV01(np.full((100, 50), 0.04), 100, 0.04, 10).shape == (100, 50)
    assert position_dv01(1_000_000, DV01_100) == pytest.approx(811, abs=1)
    assert position_dv01(-1_000_000, DV01_100) == pytest.approx(-811, abs=1)


def test_simulate_rates_starts_at_initial_rate_and_is_reproducible():
    a = simulate_rates(0.04, 10, 20, 0.15, seed=1)
    b = simulate_rates(0.04, 10, 20, 0.15, seed=1)
    assert a.shape == (10, 20) and np.all(a[:, 0] == 0.04) and np.array_equal(a, b)


def test_client_uniforms():
    u_buy, u_sell = draw_client_uniforms(1000, 1000, seed=2)
    assert u_buy.shape == (1000, 1000) and 0 <= u_buy.min() and u_buy.max() < 1
    assert not np.array_equal(u_buy, u_sell)
    assert np.mean(u_buy < 0.165) == pytest.approx(0.165, abs=0.002)
    a, _ = draw_client_uniforms(5, 10, seed=3)
    c, _ = draw_client_uniforms(5, 10, seed=3)
    e, _ = draw_client_uniforms(5, 10, seed=4)
    assert np.array_equal(a, c) and not np.array_equal(a, e)


def test_fixed_spread_quotes():
    bid, ask = quote_fixed_spread([99.9, 100.1, 100.0], 0.02)
    assert np.allclose(bid, [99.88, 100.08, 99.98]) and np.allclose(ask, [99.92, 100.12, 100.02])


def test_skewed_quotes_keep_spread_and_move_against_inventory():
    bid, ask = quote_skewed(100.0, 0.02, 0.05)       # long: both quotes move down
    assert ask - bid == pytest.approx(0.04) and (bid + ask) / 2 == pytest.approx(99.95)
    bid, ask = quote_skewed(100.0, 0.02, -0.05)      # short: both quotes move up
    assert (bid + ask) / 2 == pytest.approx(100.05)


def test_client_probabilities():
    fair = np.full(4, 100.0)
    bid = fair - np.array([0, 0.02, 0.1, 0.05])
    ask = fair + np.array([0, 0.02, 0.1, -0.05])     # last ask is below the fair price
    p_buy, p_sell = client_probabilities(fair, bid, ask, flow=0.3, sensitivity=30)
    assert np.allclose(p_buy, [0.3, 0.165, 0.015, 1.0], atol=1e-3)
    assert np.allclose(p_sell, [0.3, 0.165, 0.015, 0.067], atol=1e-3)


@pytest.fixture(scope="module")
def market():
    rates = simulate_rates(0.04, 200, 50, 0.15, seed=1)
    prices = bond_price(rates, 100, 0.04, 10)
    dv01 = calculation_DV01(rates, 100, 0.04, 10)
    u_buy, u_sell = draw_client_uniforms(200, 50, seed=2)
    return prices, dv01, u_buy, u_sell


def test_no_clients_no_pnl(market):
    prices, dv01, u_buy, u_sell = market
    r = run_strategy(prices, u_buy, u_sell, 0.02, 0.0, 30, dv01)
    assert np.all(r["position"] == 0) and np.all(r["pnl"] == 0)


def test_hand_example_sell_then_no_trade():
    fair = np.array([[100.0, 99.95]])
    ub = np.array([[0.0, 1.0]])
    us = np.ones((1, 2))                              # u = 0 forces a trade, u = 1 forbids it
    r = run_strategy(fair, ub, us, 0.02, 1.0, 0.0, np.full((1, 2), DV01_100))
    assert np.allclose(r["position"], [[-1e6, -1e6]])
    assert np.allclose(r["cash"], [[1_000_200, 1_000_200]])
    assert np.allclose(r["pnl"], [[200, 700]])


def test_constant_price_pnl_is_spread_times_trades(market):
    _, dv01, u_buy, u_sell = market
    fair = np.full((200, 50), 100.0)
    r = run_strategy(fair, u_buy, u_sell, 0.02, 0.3, 30, dv01)
    p = 0.3 * np.exp(-30 * 0.02)
    n_trades = (u_buy < p).sum(axis=1) + (u_sell < p).sum(axis=1)
    assert np.allclose(r["pnl"][:, -1], n_trades * 0.02 * 1_000_000 / 100)


def test_hard_limit_is_respected_at_constant_dv01(market):
    _, _, u_buy, u_sell = market
    fair = np.full((200, 50), 100.0)
    dv01 = np.full((200, 50), DV01_100)
    r = run_strategy(fair, u_buy, u_sell, 0.02, 0.3, 30, dv01, dv01_limit=2000)
    assert np.abs(position_dv01(r["position"], dv01)).max() <= 2000
    assert np.abs(r["position"]).max() == 2_000_000


def test_skew_shifts_quotes_by_inventory_dv01():
    fair = np.full((1, 2), 100.0)
    d = np.full((1, 2), 0.0810686)
    ub = np.ones((1, 2))
    us = np.array([[0.0, 1.0]])                       # we buy 1M at step 0, then no trade
    r = run_strategy(fair, ub, us, 0.02, 1.0, 0.0, d, skew=0.1)
    assert np.allclose(r["position"], [[1e6, 1e6]])
    assert np.allclose(r["bid"], [[99.98, 99.8989]], atol=1e-4)    # shift = 0.1 * 810.7 / 1000
    assert np.allclose(r["ask"], [[100.02, 99.9389]], atol=1e-4)


def test_skew_zero_with_and_without_limit_is_consistent(market):
    prices, dv01, u_buy, u_sell = market
    free = run_strategy(prices, u_buy, u_sell, 0.02, 0.3, 30, dv01, skew=0.0)
    limited = run_strategy(prices, u_buy, u_sell, 0.02, 0.3, 30, dv01, dv01_limit=10**9)
    assert all(np.allclose(free[k], limited[k]) for k in free)    # a non-binding limit changes nothing


def test_summarise_by_hand():
    res = {"pnl": np.array([[0, 100, 50], [0, -20, -70]])}
    dv01 = np.array([[0, -5, 3], [10, 2, -40]])
    s = summarise(res, dv01)
    assert np.allclose(s["final_pnl"], [50, -70])
    assert np.allclose(s["pnl_vol"], [75, 15])
    assert np.allclose(s["max_abs_dv01"], [5, 40])
    assert np.allclose(s["mean_abs_dv01"], [8 / 3, 52 / 3])
