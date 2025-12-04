
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.pipeline.scenario_runner import run_scenario, ScenarioSpec
from src.core.pipeline.visualizer import plot_scenario
from src.core.generator.engine import MarketState

def main():
    print("Generating 12 Test Charts...")
    
    out_dir = Path("out/test12")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Base date for all scenarios
    base_date = datetime(2024, 1, 1)
    
    scenarios = [
        # 1. Baseline RTH (Standard day)
        ("01_baseline_rth", ScenarioSpec(
            start_time=base_date.replace(hour=9, minute=30),
            duration_minutes=390,
            source="synthetic"
        )),
        
        # 2. Trend Up (Forced Rally)
        ("02_trend_up", ScenarioSpec(
            start_time=base_date.replace(hour=9, minute=30),
            duration_minutes=390,
            force_state="rally"
        )),
        
        # 3. Trend Down (Forced Breakdown)
        ("03_trend_down", ScenarioSpec(
            start_time=base_date.replace(hour=9, minute=30),
            duration_minutes=390,
            force_state="breakdown"
        )),
        
        # 4. Morning Drive (9:30-11:00, High Vol)
        ("04_morning_drive", ScenarioSpec(
            start_time=base_date.replace(hour=9, minute=30),
            duration_minutes=90,
            state_sequence=[(0, "IMPULSIVE"), (45, "RALLY")]
        )),
        
        # 5. Premarket Transition (08:00-10:00)
        ("05_premarket_transition", ScenarioSpec(
            start_time=base_date.replace(hour=8, minute=0),
            duration_minutes=120,
            state_sequence=[(0, "ZOMBIE"), (30, "RANGING"), (90, "IMPULSIVE")] # 8:30 econ, 9:30 open
        )),
        
        # 6. Lunch Lull (11:30-13:30, Flat)
        ("06_lunch_lull", ScenarioSpec(
            start_time=base_date.replace(hour=11, minute=30),
            duration_minutes=120,
            force_state="flat"
        )),
        
        # 7. Afternoon Reversal (Up then Down)
        ("07_afternoon_reversal", ScenarioSpec(
            start_time=base_date.replace(hour=13, minute=0),
            duration_minutes=180,
            state_sequence=[(0, "RALLY"), (90, "RANGING"), (120, "BREAKDOWN")]
        )),
        
        # 8. Market On Close (14:30-16:00, Volatile)
        ("08_moc_volatile", ScenarioSpec(
            start_time=base_date.replace(hour=14, minute=30),
            duration_minutes=90,
            state_sequence=[(0, "RANGING"), (60, "IMPULSIVE")]
        )),
        
        # 9. High Volatility Day
        ("09_high_vol_day", ScenarioSpec(
            start_time=base_date.replace(hour=9, minute=30),
            duration_minutes=390,
            physics_overrides={"base_volatility": 3.0}
        )),
        
        # 10. Low Volatility Day
        ("10_low_vol_day", ScenarioSpec(
            start_time=base_date.replace(hour=9, minute=30),
            duration_minutes=390,
            physics_overrides={"base_volatility": 0.5}
        )),
        
        # 11. Choppy Day (Forced Ranging/Zombie)
        ("11_choppy_day", ScenarioSpec(
            start_time=base_date.replace(hour=9, minute=30),
            duration_minutes=390,
            state_sequence=[(0, "RANGING"), (120, "ZOMBIE"), (240, "RANGING")]
        )),
        
        # 12. News Spike (Quiet then Massive Move)
        ("12_news_spike", ScenarioSpec(
            start_time=base_date.replace(hour=9, minute=30),
            duration_minutes=180,
            state_sequence=[(0, "FLAT"), (60, "IMPULSIVE"), (75, "RANGING")]
        ))
    ]
    
    for name, spec in scenarios:
        print(f"Running scenario: {name}")
        try:
            result = run_scenario(spec, seed=42)
            if result.df_5m.empty:
                print(f"  Warning: No data generated for {name}")
                continue
                
            plot_scenario(result, save_dir=str(out_dir), filename_prefix=name)
            print(f"  Saved chart to {out_dir}/{name}.png")
            
        except Exception as e:
            print(f"  Error running {name}: {e}")
            import traceback
            traceback.print_exc()

    print("Done.")

if __name__ == "__main__":
    main()
