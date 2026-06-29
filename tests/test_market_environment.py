"""
Unit Tests for testing the Market Data Environment
"""

import pytest
import pandas as pd

# Import the modules we want to test
from legacy_market_data.market_environment import MarketEnvironment


# This fixture creates a single 'env' object that all the tests can use.
@pytest.fixture(scope="module")
def global_env():
    """
    pytest fixture to initialize the MarketEnvironment once for all tests.
    """
    try:
        env = MarketEnvironment()
        return env
    except Exception as e:
        pytest.fail(f"Failed to initialize MarketEnvironment: {e}")


#  Test Functions
def test_env_initialization(global_env):
    """
    Test 1: Did the environment initialize correctly?
    """
    assert global_env is not None
    assert isinstance(global_env.processed_data, pd.DataFrame)


def test_data_has_correct_columns(global_env):
    """
    Test 2: Does the final DataFrame have all the columns we expect?
    """
    expected_cols = ["price", "SMA20", "SMA60", "RSI14", "MACD", "MACD_signal"]
    df = global_env.processed_data
    assert all(col in df.columns for col in expected_cols)


def test_data_length(global_env):
    """
    Test 3: Does the environment report a plausible number of trading days?
    (Approx 252 trading days/year * 3 years = ~756. Minus 60 for SMA window.)
    """
    assert len(global_env) > 650
    assert len(global_env) == len(global_env.get_price_series())


def test_no_nan_values_in_processed_data(global_env):
    """
    Test 4: Did our dropna() work? No agent should ever see a NaN.
    """
    df = global_env.processed_data
    assert not df.isnull().values.any(), "Found NaN values in processed_data"


def test_get_state_for_day_valid(global_env):
    """
    Test 5: Does get_state_for_day(0) return the correct data?
    """
    state = global_env.get_state_for_day(0)
    df_row = global_env.processed_data.iloc[0]

    assert isinstance(state, dict)
    assert "date" in state
    assert "price" in state
    assert state["price"] == df_row["price"]
    assert state["SMA20"] == df_row["SMA20"]
    assert state["RSI14"] == df_row["RSI14"]
    assert state["date"] == df_row.name.strftime("%Y-%m-%d")


def test_get_state_out_of_bounds(global_env):
    """
    Test 6: Does it correctly raise an error if we ask for a day
    that doesn't exist? (This prevents silent failures in the backtest).
    """
    invalid_index = len(global_env) + 10  # An index that is definitely too large

    with pytest.raises(IndexError):
        global_env.get_state_for_day(invalid_index)
