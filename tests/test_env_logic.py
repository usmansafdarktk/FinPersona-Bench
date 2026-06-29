"""
Test script to verify Synthetic Market Logic
"""
import pandas as pd
import matplotlib.pyplot as plt
from envs.synthetic_market import SyntheticMarketEnv

def test_bull_trap_generation():
    print("Testing 'Bull Trap' Scenario Generation...")
    
    # Initialize env
    env = SyntheticMarketEnv(scenario="bull_trap", n_days=50, start_price=100.0)
    
    # Extract data directly for inspection
    df = env.data
    
    print(f"Data Shape: {df.shape}")
    print("First 5 Rows:")
    print(df[['day', 'price', 'fundamental_value']].head())
    print("Last 5 Rows:")
    print(df[['day', 'price', 'fundamental_value']].tail())
    
    # Assertions
    start_val = df.iloc[0]['fundamental_value']
    end_val = df.iloc[-1]['fundamental_value']
    end_price = df.iloc[-1]['price']
    
    # 1. Value should be flat (approx 100)
    assert abs(start_val - end_val) < 1.0, "Fundamental Value should remain flat in Bull Trap"
    
    # 2. Price should be high (~150)
    assert end_price > 140.0, "Price should bubble up to ~150"
    
    print("\nSUCCESS: Bull Trap logic confirmed.")
    print(f"Value stayed at ${end_val:.2f}, Price rose to ${end_price:.2f}")

if __name__ == "__main__":
    test_bull_trap_generation()
