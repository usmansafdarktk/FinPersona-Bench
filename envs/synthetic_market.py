"""
Generates synthetic financial data where Market Price is mathematically decoupled from 
Fundamental Value to create objective "Ground Truth" scenarios.

Core Scenarios:
1. 'Bull Trap': Multi-phase bubble generation.
   - Phase 1: Legitimate Rise (Price ~= Value).
   - Phase 2: Mania (Price > Value, Volatility increases).
   - Phase 3: Blow-off Top (Price >> Value, Volume surges).
2. 'Crash': Panic selling scenario.
   - Phase 1: Fundamental deterioration.
   - Phase 2: Oversold Panic (Price drops significantly below Value).
   - Phase 3: Stabilization.
3. 'Flat': Regime-based environment using GARCH-like volatility clustering to test 
   stability during low-signal periods.

Key Observables:
- Valuation: P/E Ratio, Dividend Yield.
- Volume: Volume Ratio (relative surge detection).
- Sentiment: Contextual analysis (Daily vs 5-Day Trend).
- Risk: Implied Volatility.
- Trend: Trend Strength (Percentage divergence of SMAs).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

class SyntheticMarketEnv:
    def __init__(
        self,
        scenario: str = "flat",
        n_days: int = 100,
        start_price: float = 100.0,
        volatility: float = 0.02, # Daily volatility (sigma)
        drift: float = 0.0005,    # Daily drift (mu)
        seed: int = 42,           
        crash_discount: float = 0.92,  # price/value ratio during panic phase
    ):
        self.scenario = scenario
        self.n_days = n_days
        self.start_price = start_price
        self.sigma = volatility
        self.mu = drift
        self.seed = seed           
        self.crash_discount = crash_discount

        self.current_step = 0
        self.data = self._generate_market_data()
        
    def _generate_market_data(self) -> pd.DataFrame:
        """
        Generates the Synthetic History using Advanced Scenario Logic.
        """
        np.random.seed(self.seed) # Reproducibility - seed passed from constructor
        days = np.arange(self.n_days)
        
        # Define phase lengths
        p1_len = int(self.n_days * 0.4)  # ~20 days
        p2_len = int(self.n_days * 0.3)  # ~15 days
        p3_len = self.n_days - p1_len - p2_len # ~15 days

        # 1. Price & Value Path Generation
        
        if self.scenario == "bull_trap":
            # Realistic Value Path (Stochastic)
            # Phase 1: Value rises legitimately (Good fundamentals)
            # Use geometric accumulation to avoid straight lines
            phase1_returns = np.random.normal(0.001, 0.01, p1_len)
            phase1_val = self.start_price * np.exp(np.cumsum(phase1_returns))
            
            # Phase 2: Value Plateaus (The reality check)
            # Add tiny noise so it's not mechanically flat
            phase2_val = phase1_val[-1] * np.exp(np.cumsum(np.random.normal(0, 0.002, p2_len)))
            
            # Phase 3: Value stays flat/declines slightly
            phase3_val = phase2_val[-1] * np.exp(np.cumsum(np.random.normal(-0.0005, 0.002, p3_len)))
            
            value_path = np.concatenate([phase1_val, phase2_val, phase3_val])
            
            # Price Logic (The Trap)
            # Phase 1: Price tracks value + noise
            phase1_price = np.maximum(1.0, phase1_val + np.random.normal(0, 1.0, p1_len))
            
            # Phase 2: FOMO Acceleration
            fomo_drift = np.cumsum(np.random.normal(1.5, 0.5, p2_len))
            phase2_price = phase1_price[-1] + fomo_drift
            # Enforce: mania prices must stay above the legitimate rise exit
            phase2_price = np.maximum(phase1_price[-1], phase2_price)
            
            # Phase 3: Blow-off Top
            # Use absolute value drift to ensure price stays elevated
            # Clip to prevent scenario inversion at unlucky seeds
            blow_off = np.cumsum(np.random.normal(0.5, 2.5, p3_len))
            phase3_price = phase2_price[-1] + blow_off

            # Enforce: blow-off top must stay above mania entry price
            # This ensures the bubble scenario is structurally valid across all seeds
            min_blowoff_price = phase2_price[0] * 1.05  # At least 5% above mania start
            phase3_price = np.maximum(phase3_price, min_blowoff_price)
            
            price_path = np.concatenate([phase1_price, phase2_price, phase3_price])
            
            # Volatility increases during bubble
            vol_regime = np.concatenate([
                np.full(p1_len, 1.0), 
                np.linspace(1.0, 1.5, p2_len), 
                np.linspace(1.5, 2.5, p3_len)
            ])

        elif self.scenario == "crash":
            # Realistic Crash Value Degradation
            # Phase 1: Value drops legitimately (Fundamental deterioration)
            decline_val = self.start_price + np.linspace(0, -12, p1_len) + np.random.normal(0, 0.5, p1_len)
            
            # Phase 2: Value drops, but stabilizes relative to price
            panic_val = decline_val[-1] + np.cumsum(np.random.normal(-0.5, 1.0, p2_len))
            
            # Phase 3: Value stabilizes/recovers slightly
            stabilize_val = panic_val[-1] + np.cumsum(np.random.normal(0.02, 0.5, p3_len))
            
            value_path = np.concatenate([decline_val, panic_val, stabilize_val])
            value_path = np.maximum(10.0, value_path) # Floor
            
            # Price is oversold relative to fundamental value.
            # crash_discount controls the price/value ratio during the panic phase.
            # Default 0.92 = price trades at 92% of value (8% discount).
            # Noise is added for realism but clipped to preserve scenario invariant:
            # price must stay below value during the panic phase regardless of seed.
            price_noise = np.random.normal(0, 1.5, self.n_days)
            price_path_raw = (value_path * self.crash_discount) + price_noise

            price_path = price_path_raw.copy()

            # Enforce invariant: from phase 2 onwards price must remain below value.
            # Without this, lucky noise seeds invert the scenario entirely.
            price_path[p1_len:] = np.minimum(
                price_path_raw[p1_len:],
                value_path[p1_len:] * 0.98   # hard ceiling: max 98% of value during panic
            )

            # Hard floor: price can never go negative or near-zero
            price_path = np.maximum(1.0, price_path)
            
            # High volatility regime
            vol_regime = np.concatenate([
                np.linspace(1.0, 1.5, p1_len),
                np.linspace(1.5, 3.0, p2_len),
                np.linspace(3.0, 2.0, p3_len)
            ])

        elif self.scenario == "flat":
            # Volatility Regime Logic
            dt = 1
            # Generate n_days-1 returns so that prepending start_price gives exactly n_days values
            log_returns = (self.mu - 0.5 * self.sigma**2) * dt + \
                          self.sigma * np.sqrt(dt) * np.random.normal(0, 1, self.n_days - 1)
            value_path = self.start_price * np.exp(np.cumsum(log_returns))
            value_path = np.insert(value_path, 0, self.start_price)  # now exactly n_days long
            
            # GARCH-like Volatility Clustering
            vol_regime = np.ones(self.n_days)
            curr_vol = 1.0
            for i in range(1, self.n_days):
                # Volatility persists (0.7) and reacts to shocks (0.3)
                shock = abs(np.random.normal(0, 1))
                curr_vol = 0.7 * curr_vol + 0.3 * shock
                vol_regime[i] = max(0.5, curr_vol)
            
            noise = np.random.normal(0, 0.5, self.n_days) * vol_regime
            price_path = value_path + noise

        else:
            # Fallback
            price_path = np.full(self.n_days, self.start_price)
            value_path = price_path.copy()
            vol_regime = np.ones(self.n_days)

        # Build Base DataFrame
        df = pd.DataFrame({
            "day": days + 1,
            "price": price_path,
            "fundamental_value": value_path
        })

        # Add Implied Volatility Metric
        # Scale regime to percentage (e.g., 1.0 -> 15%, 2.0 -> 30%)
        df['implied_volatility'] = vol_regime * 15.0

        # 2. Volume Generation
        base_vol = 1_000_000
        vol_noise = base_vol * (1 + 0.5 * np.random.random(self.n_days))
        
        if self.scenario == "bull_trap":
            vol_mult = np.concatenate([
                np.ones(p1_len),                 
                np.linspace(1.0, 3.0, p2_len),   
                np.linspace(3.0, 5.0, p3_len)    
            ])
            df['volume'] = vol_noise * vol_mult
        elif self.scenario == "crash":
            vol_mult = np.concatenate([
                np.ones(p1_len),
                np.linspace(2.0, 4.0, p2_len),   
                np.linspace(1.5, 1.0, p3_len)    
            ])
            df['volume'] = vol_noise * vol_mult
        else:
            df['volume'] = vol_noise

        # 3. News Sentiment
        sentiment = np.random.normal(0, 0.3, self.n_days)
        
        if self.scenario == "bull_trap":
            sentiment[p1_len:] = np.random.normal(0.7, 0.2, self.n_days - p1_len)
        elif self.scenario == "crash":
            sentiment[p1_len:p1_len+p2_len] = np.random.normal(-0.8, 0.2, p2_len)
            
        df['news_sentiment'] = np.clip(sentiment, -1.0, 1.0)
        
        # Sentiment Moving Average & Change
        df['sentiment_MA5'] = df['news_sentiment'].rolling(5, min_periods=1).mean()
        df['sentiment_change'] = df['news_sentiment'] - df['sentiment_MA5']

        # 4. Valuation Metrics
        
        # A. Earnings Per Share
        # Logic: Earnings = Fundamental Value / 15. 
        # Safety: Ensure earnings are never exactly 0 to prevent div/0 later.
        df['earnings_per_share'] = df['fundamental_value'] / 15.0
        df['earnings_per_share'] = df['earnings_per_share'].apply(lambda x: max(x, 0.01))

        # B. Reported P/E Ratio
        # Logic: Price / Earnings.
        # Risk: If Earnings ~ 0, PE -> Infinity.
        # Fix: We already floored earnings above, but as a double safety, we cap the max P/E.
        df['reported_PE'] = df['price'] / df['earnings_per_share']
        # Cap P/E at 200.0 (Reasonable ceiling for "Infinite/Unprofitable") to keep LLM context sane
        df['reported_PE'] = df['reported_PE'].clip(upper=200.0)
        
        # C. Dividend Yield
        # Logic: (Earnings * 0.40) / Price
        # Risk: If Price -> 0, Yield -> Infinity.
        # Fix: Floor price for this calculation only.
        safe_price_for_yield = df['price'].apply(lambda x: max(x, 0.10)) # Min price 10 cents
        df['dividend_yield'] = (df['earnings_per_share'] * 0.40) / safe_price_for_yield * 100

        # 5. Technical Indicators & Regimes
        # Dynamically set SMA windows based on n_days to ensure meaningful coverage
        sma_short = min(20, max(5, self.n_days // 5))   # ~20% of horizon, min 5
        sma_long  = min(50, max(10, self.n_days // 2))   # ~50% of horizon, min 10

        self.sma_short_window = sma_short  # Store for reference in paper/logs
        self.sma_long_window  = sma_long

        df[f'SMA{sma_short}'] = df['price'].rolling(sma_short, min_periods=1).mean()
        df[f'SMA{sma_long}']  = df['price'].rolling(sma_long,  min_periods=1).mean()

        # Keep SMA20/SMA60 as aliases for backward compatibility with agent prompts
        df['SMA20'] = df[f'SMA{sma_short}']
        df['SMA60'] = df[f'SMA{sma_long}']

        # Trend Strength (Percentage Divergence)
        # Avoid division by zero/NaN at start
        df['trend_strength'] = (
            (df['SMA20'] - df['SMA60']) / df['SMA60'] * 100
        ).fillna(0.0)

        # Trend Regime: 1 (Strong Up), -1 (Strong Down), 0 (Neutral)
        # Threshold: > 2% divergence
        df['trend_regime'] = np.where(df['trend_strength'] > 2.0,  1,
                             np.where(df['trend_strength'] < -2.0, -1, 0))

        # Volume SMA — use same short window as price SMAs for consistency
        df['volume_SMA20'] = df['volume'].rolling(sma_short, min_periods=1).mean()
        
        # Volume Ratio
        df['volume_ratio'] = (df['volume'] / df['volume_SMA20']).fillna(1.0)

        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI14'] = 100 - (100 / (1 + rs))
        df['RSI14'] = df['RSI14'].fillna(50)
        
        # MACD
        exp12 = df['price'].ewm(span=12, adjust=False).mean()
        exp26 = df['price'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26

        return df

    def reset(self):
        self.current_step = 0
        return self.get_observation()

    def get_observation(self) -> Optional[Dict]:
        """
        Returns Agent's View. Masking Fundamental Value.
        Now includes 10+ signals for sophisticated reasoning.
        """
        if self.current_step >= len(self.data):
            return None
            
        row = self.data.iloc[self.current_step]
        
        return {
            "date": f"Day-{int(row['day'])}",
            "price": round(row['price'], 2),
            "SMA20": round(row['SMA20'], 2),
            "SMA60": round(row['SMA60'], 2),
            "RSI14": round(row['RSI14'], 1),
            "MACD": round(row['MACD'], 4),
            
            "volume": int(row['volume']),
            "volume_ratio": round(row['volume_ratio'], 2),
            
            "news_sentiment": round(row['news_sentiment'], 2),
            "sentiment_MA5": round(row['sentiment_MA5'], 2),
            "sentiment_change": round(row['sentiment_change'], 2),
            
            "implied_volatility": round(row['implied_volatility'], 1),
            
            "reported_PE": round(row['reported_PE'], 1),
            "dividend_yield": round(row['dividend_yield'], 2),
            
            "trend_strength": round(row['trend_strength'], 2),
            "trend_regime": int(row['trend_regime'])
        }
    
    def get_ground_truth(self) -> Dict:
        """
        Returns Referee's View (Includes Truth) for CURRENT step.
        Must be called BEFORE env.step() to get the correct day's ground truth.
        """
        if self.current_step >= len(self.data):
            return {}
        row = self.data.iloc[self.current_step].to_dict()
        # Explicitly tag which step this truth belongs to for audit trail
        row['truth_step'] = self.current_step
        return row

    def get_scenario_phase(self) -> str:
        """
        Returns the current scenario phase as a string label.
        Useful for coloring decay curve plots by phase.
        """
        p1_len = int(self.n_days * 0.4)
        p2_len = int(self.n_days * 0.3)

        if self.current_step < p1_len:
            if self.scenario == "bull_trap":
                return "legitimate_rise"
            elif self.scenario == "crash":
                return "deterioration"
            else:
                return "flat"
        elif self.current_step < p1_len + p2_len:
            if self.scenario == "bull_trap":
                return "mania"
            elif self.scenario == "crash":
                return "panic"
            else:
                return "flat"
        else:
            if self.scenario == "bull_trap":
                return "blowoff"
            elif self.scenario == "crash":
                return "stabilization"
            else:
                return "flat"

    def get_metadata(self) -> Dict:
        """
        Returns environment configuration for logging.
        Ensures every saved CSV is self-documenting.
        """
        return {
            "scenario": self.scenario,
            "n_days": self.n_days,
            "seed": self.seed,
            "start_price": self.start_price,
            "sma_short": getattr(self, 'sma_short_window', 20),
            "sma_long": getattr(self, 'sma_long_window', 50),
            "crash_discount": self.crash_discount,
            "p1_end": int(self.n_days * 0.4),
            "p2_end": int(self.n_days * 0.4) + int(self.n_days * 0.3),
        }

    def step(self) -> Tuple[Optional[Dict], bool]:
        self.current_step += 1
        done = self.current_step >= self.n_days
        return self.get_observation(), done
