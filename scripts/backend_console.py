"""
Backend Console CLI

Terminal-based control panel for the FracFire training engine.
Allows running scenarios, testing setups, and performing parameter sweeps.
"""

import argparse
import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.pipeline.scenario_runner import run_scenario, ScenarioSpec, SetupConfig
from src.core.pipeline.visualizer import plot_scenario
from src.core.detector.indicators import IndicatorConfig
from src.core.detector.sweep import sweep_setup_family

def check_venv():
    """Check if running in a virtual environment."""
    if sys.prefix == sys.base_prefix:
        print("WARNING: You are NOT running inside a virtual environment (venv).")
        print("Please activate your venv (e.g., source .venv312/bin/activate) and try again.")
        # We won't exit, just warn, in case they really want to run globally.

def parse_args():
    parser = argparse.ArgumentParser(description="FracFire Backend Console")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # RUN Command
    run_parser = subparsers.add_parser("run", help="Run a single scenario")
    run_parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic", help="Data source")
    run_parser.add_argument("--preset", type=str, help="Scenario preset (e.g., trend_up)")
    run_parser.add_argument("--setup", type=str, default="orb", help="Setup family to test (comma-separated)")
    run_parser.add_argument("--days", type=int, default=1, help="Number of days to generate")
    run_parser.add_argument("--knobs", type=str, help="JSON string for physics overrides")
    run_parser.add_argument("--force-state", type=str, help="Force a single market state (e.g. ranging)")
    run_parser.add_argument("--plot", action="store_true", help="Generate charts")
    run_parser.add_argument("--save-data", action="store_true", help="Save raw data (parquet)")
    run_parser.add_argument("--verbose", action="store_true", help="Print detailed output")

    # SWEEP Command
    sweep_parser = subparsers.add_parser("sweep", help="Run a parameter sweep")
    sweep_parser.add_argument("--setup", type=str, required=True, help="Setup family to sweep")
    sweep_parser.add_argument("--param", action='append', nargs='+', help="Parameter to sweep: name v1 v2 ...")
    sweep_parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic", help="Data source")
    sweep_parser.add_argument("--preset", type=str, help="Scenario preset")
    
    return parser.parse_args()

def main():
    check_venv()
    args = parse_args()
    
    if args.command == "run":
        handle_run(args)
    elif args.command == "sweep":
        handle_sweep(args)
    else:
        print("Please specify a command: run or sweep")

def handle_sweep(args):
    print(f"--- Starting Sweep for {args.setup} ---")
    
    # Parse Params
    param_grid = {}
    if args.param:
        for p in args.param:
            name = p[0]
            values = []
            for v in p[1:]:
                # Try to convert to int or float
                try:
                    if "." in v:
                        val = float(v)
                    else:
                        val = int(v)
                except ValueError:
                    val = v
                values.append(val)
            param_grid[name] = values
    
    if not param_grid:
        print("Error: No parameters specified to sweep. Use --param name v1 v2 ...")
        return

    # Base Spec
    spec = ScenarioSpec(
        source=args.source,
        start_time=datetime(2024, 1, 1, 9, 30)
    )
    if args.preset:
        if args.preset == "trend_up":
            spec.macro_regime = "UP_DAY"
            spec.state_sequence = [(0, "RANGING"), (60, "BREAKOUT"), (120, "RALLY")]
        elif args.preset == "range":
            spec.macro_regime = "CHOP_DAY"
            spec.state_sequence = [(0, "RANGING")]

    # Run Sweep
    results = sweep_setup_family(args.setup, spec, param_grid)
    
    # Print Results
    print("\n--- Sweep Results ---")
    # Sort by Total R descending
    results.sort(key=lambda x: x.total_r, reverse=True)
    
    print(f"{'Params':<40} | {'Trades':<6} | {'Win Rate':<8} | {'Total R':<8} | {'Avg R':<8}")
    print("-" * 85)
    
    for r in results:
        param_str = str(r.params)
        if len(param_str) > 38:
            param_str = param_str[:35] + "..."
        print(f"{param_str:<40} | {r.num_trades:<6} | {r.win_rate:<8.2%} | {r.total_r:<8.2f} | {r.avg_r:<8.2f}")


def handle_run(args):
    print(f"--- Starting Scenario Run ({args.source}) ---")
    
    # Parse Knobs
    physics_overrides = {}
    if args.knobs:
        try:
            physics_overrides = json.loads(args.knobs)
        except json.JSONDecodeError:
            print("Error: Invalid JSON for --knobs")
            return

    # Configure Setups
    active_families = args.setup.split(",")
    setup_cfg = SetupConfig(active_families=active_families)

    # Build Spec
    # For now, we'll just run one day per iteration if multiple days requested,
    # or we can modify ScenarioSpec to handle multiple days. 
    # The directive implies a "Scenario" is a unit of work. 
    # Let's run a loop here if days > 1.
    
    total_trades = 0
    total_r = 0.0
    wins = 0
    losses = 0
    
    for day in range(args.days):
        seed = 42 + day
        spec = ScenarioSpec(
            source=args.source,
            physics_overrides=physics_overrides,
            setup_cfg=setup_cfg,
            start_time=datetime(2024, 1, 1, 9, 30) + (day * timedelta(days=1)), # Just shifting start time
            force_state=args.force_state
        )
        
        if args.preset:
            # Apply preset logic here (e.g. map preset name to state sequence)
            if args.preset == "trend_up":
                spec.macro_regime = "UP_DAY"
                spec.state_sequence = [(0, "RANGING"), (60, "BREAKOUT"), (120, "RALLY")]
            elif args.preset == "range":
                spec.macro_regime = "CHOP_DAY"
                spec.state_sequence = [(0, "RANGING")]
        
        result = run_scenario(spec, seed=seed)
        
        # Aggregate Stats
        day_trades = len(result.outcomes)
        total_trades += day_trades
        
        for outcome in result.outcomes:
            total_r += outcome.r_multiple
            if outcome.hit_target:
                wins += 1
            elif outcome.hit_stop:
                losses += 1
        
        if args.plot:
            plot_scenario(result, save_dir="out/charts/backend_console", filename_prefix=f"run_{args.source}_{day}")

        if args.save_data:
            save_dir = Path("out/scenarios")
            save_dir.mkdir(parents=True, exist_ok=True)
            # Save 1m data
            filename = save_dir / f"run_{args.source}_{day}_1m.parquet"
            result.df_1m.to_parquet(filename)
            if args.verbose:
                print(f"Saved data to {filename}")

        if args.verbose:
            print(f"Day {day+1}: {day_trades} trades")

    # Print Summary
    print("\n--- Run Summary ---")
    print(f"Total Days: {args.days}")
    print(f"Total Trades: {total_trades}")
    if total_trades > 0:
        win_rate = wins / total_trades
        avg_r = total_r / total_trades
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total R: {total_r:.2f}")
        print(f"Avg R/Trade: {avg_r:.2f}")
    else:
        print("No trades found.")

if __name__ == "__main__":
    main()
