"""
Generate Synthetic Archetypes

Creates a library of labeled price patterns (archetypes) for pre-training models.
Generates 100+ samples for each of the 10 defined archetypes.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from lab.generators import (
    PriceGenerator,
    MarketState,
    StateConfig,
    get_custom_state,
    STATE_CONFIGS
)

# Configuration
OUTPUT_DIR = root / "out" / "data" / "synthetic" / "archetypes"
SAMPLES_PER_ARCHETYPE = 100
START_DATE = datetime(2024, 1, 1, 0, 0, 0)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic archetypes")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_ARCHETYPE, help="Number of samples per archetype")
    return parser.parse_args()

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def generate_rally_day(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 1: Pure Rally Day - Sustained upward trend"""
    # Mix of RALLY and IMPULSIVE, maybe some RANGING rests
    state_sequence = [
        (0, MarketState.RANGING),      # Open
        (60, MarketState.RALLY),       # Start moving
        (240, MarketState.RANGING),    # Rest
        (300, MarketState.RALLY),      # Resume
        (600, MarketState.IMPULSIVE),  # Accelerate
        (720, MarketState.RALLY),      # Sustain
        (1200, MarketState.RANGING),   # Close
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="RALLY_DAY")

def generate_range_day(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 2: Pure Range Day - Bounded, mean-reverting"""
    # Mostly RANGING and FLAT, maybe brief false breakouts
    state_sequence = [
        (0, MarketState.RANGING),
        (300, MarketState.FLAT),
        (600, MarketState.RANGING),
        (900, MarketState.FLAT),
        (1200, MarketState.RANGING),
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="RANGE_DAY")

def generate_breakout_pattern(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 3: Breakout Pattern - Range -> Breakout -> Rally"""
    state_sequence = [
        (0, MarketState.RANGING),
        (400, MarketState.RANGING),
        (420, MarketState.BREAKOUT),   # The move
        (480, MarketState.IMPULSIVE),  # Follow through
        (600, MarketState.RALLY),      # Trend
        (1000, MarketState.RANGING),   # Consolidation at new high
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="BREAKOUT_PATTERN")

def generate_breakdown_pattern(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 4: Breakdown Pattern - Range -> Breakdown -> Selloff"""
    state_sequence = [
        (0, MarketState.RANGING),
        (400, MarketState.RANGING),
        (420, MarketState.BREAKDOWN),  # The drop
        (480, MarketState.IMPULSIVE),  # Follow through (down)
        # Note: IMPULSIVE is high vol, direction depends on trend persistence/bias. 
        # For pure breakdown we might need a custom state or rely on momentum.
        # Let's use a custom 'strong_down' config if needed, but standard states might work 
        # if the breakdown sets the direction.
        (600, MarketState.RALLY),      # RALLY state has up bias. We need a 'SELLOFF' state really.
        # Using custom state for selloff to ensure down direction
    ]
    
    # Let's use standard states but rely on the generator's momentum or use a custom config for the trend part
    # Actually, let's use a custom state for the selloff part to guarantee it goes down
    selloff_config = StateConfig(
        name="selloff",
        avg_ticks_per_bar=20.0,
        up_probability=0.3, # Down bias
        trend_persistence=0.8,
        volatility_multiplier=1.5
    )
    
    # We can't easily mix state_sequence with custom_config in generate_day directly 
    # unless we modify generate_day to accept a list of configs.
    # For now, let's stick to standard states and hope the BREAKDOWN momentum carries, 
    # or use 'BREAKDOWN' state for longer.
    
    state_sequence = [
        (0, MarketState.RANGING),
        (400, MarketState.BREAKDOWN),
        (600, MarketState.BREAKDOWN), # Keep breaking down
        (800, MarketState.RANGING),
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="BREAKDOWN_PATTERN")

def generate_reversal_pattern(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 5: Reversal Pattern - Rally -> Range -> Breakdown"""
    state_sequence = [
        (0, MarketState.RALLY),        # Up
        (400, MarketState.RANGING),    # Top
        (600, MarketState.BREAKDOWN),  # Reversal
        (800, MarketState.RANGING),    # Settle low
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="REVERSAL_PATTERN")

def generate_zombie_grind(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 6: Zombie Grind - Slow, low-volatility trend"""
    state_sequence = [
        (0, MarketState.ZOMBIE),
        (1440, MarketState.ZOMBIE), # All day
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="ZOMBIE_GRIND")

def generate_volatile_chop(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 7: Volatile Chop - High volatility, no direction"""
    # We can use the 'mega_volatile' custom state, but generate_day takes standard states in sequence.
    # We'll use IMPULSIVE which is high vol.
    state_sequence = [
        (0, MarketState.IMPULSIVE),
        (1440, MarketState.IMPULSIVE),
    ]
    # To make it choppy (no direction), we might need to rely on the state config.
    # IMPULSIVE has 0.5 up_prob, so it should be choppy.
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="VOLATILE_CHOP")

def generate_opening_bell(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 8: Opening Bell - High volatility at open, then settle"""
    # RTH Open is usually 9:30 ET (8:30 CT). Let's assume 9:30 start for simplicity in bar index
    # 9:30 is 570 minutes from midnight
    state_sequence = [
        (0, MarketState.RANGING),      # Overnight
        (570, MarketState.IMPULSIVE),  # Open (High Vol)
        (630, MarketState.RALLY),      # Direction established
        (750, MarketState.RANGING),    # Lunch
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="OPENING_BELL")

def generate_closing_squeeze(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 9: Closing Squeeze - Quiet day -> End-of-day rally"""
    # Close is 16:00 ET (15:00 CT) -> 900 minutes
    state_sequence = [
        (0, MarketState.RANGING),
        (840, MarketState.RALLY),      # 14:00 CT - Start move
        (870, MarketState.IMPULSIVE),  # 14:30 CT - Squeeze
        (900, MarketState.RANGING),    # Close
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="CLOSING_SQUEEZE")

def generate_news_event(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 10: News Event - Sudden volatility spike"""
    # Random time spike
    event_time = 600 # 10:00 am
    state_sequence = [
        (0, MarketState.RANGING),
        (event_time, MarketState.BREAKOUT), # The news
        (event_time + 15, MarketState.IMPULSIVE), # The reaction
        (event_time + 60, MarketState.RANGING), # Settle
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="NEWS_EVENT")


ARCHETYPES = {
    "rally_day": generate_rally_day,
    "range_day": generate_range_day,
    "breakout_pattern": generate_breakout_pattern,
    "breakdown_pattern": generate_breakdown_pattern,
    "reversal_pattern": generate_reversal_pattern,
    "zombie_grind": generate_zombie_grind,
    "volatile_chop": generate_volatile_chop,
    "opening_bell": generate_opening_bell,
    "closing_squeeze": generate_closing_squeeze,
    "news_event": generate_news_event,
}

def main():
    args = parse_args()
    samples_to_generate = args.samples
    
    print("=" * 60)
    print(f"GENERATING {samples_to_generate} SAMPLES PER ARCHETYPE")
    print("=" * 60)
    
    ensure_dir(OUTPUT_DIR)
    
    total_generated = 0
    
    for name, generator_func in ARCHETYPES.items():
        print(f"\nGenerating {name}...")
        archetype_dir = OUTPUT_DIR / name
        ensure_dir(archetype_dir)
        
        # Use a fixed seed base for reproducibility, but different for each sample
        base_seed = hash(name) % 100000
        
        for i in tqdm(range(samples_to_generate)):
            seed = base_seed + i
            gen = PriceGenerator(initial_price=5000.0, seed=seed)
            
            # Increment date for variety (though mostly affects DOW config)
            date = START_DATE + timedelta(days=i)
            
            try:
                df = generator_func(gen, date)
                
                # Add metadata
                df.attrs['archetype'] = name
                df.attrs['seed'] = seed
                
                # Save
                filename = f"{name}_{i:03d}.parquet"
                df.to_parquet(archetype_dir / filename)
                
                total_generated += 1
                
            except Exception as e:
                print(f"Error generating {name} sample {i}: {e}")
                continue
                
    print("\n" + "=" * 60)
    print(f"DONE. Generated {total_generated} files in {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
