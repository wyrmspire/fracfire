"""
Sweep synthetic generator volatility scale to match real candle amplitude.
Computes mean 15m candle range RMSE across the 3-month window and recommends a scale.
"""

import os
import subprocess
import sys
import json

SCALES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]


def run_compare(scale: float):
    # Run compare_real_synth.py and capture stdout to parse metrics
    env = os.environ.copy()
    cmd = [sys.executable, 'scripts/compare_real_synth.py', '--vol-scale', str(scale)]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.abspath('.'))
    out = p.stdout
    # Look for the line "15m mean range: real=... synth=... ratio=..."
    metric_line = None
    for line in out.splitlines():
        if line.strip().startswith('15m mean range:'):
            metric_line = line.strip()
            break
    if not metric_line:
        return None, out
    # parse values via regex
    import re
    m = re.search(r"real=([0-9\.]+)\s+synth=([0-9\.]+)\s+ratio=([0-9\.]+)", metric_line)
    if not m:
        return None, out
    real = float(m.group(1))
    synth = float(m.group(2))
    ratio = float(m.group(3))
    return {'real': real, 'synth': synth, 'ratio': ratio, 'scale': scale}, out


def main():
    results = []
    for s in SCALES:
        m, out = run_compare(s)
        if m is None:
            print(f"Scale {s}: failed to parse metrics. Output:\n{out}")
            continue
        results.append(m)
        print(f"Scale {s:.2f}: ratio={m['ratio']:.3f} real={m['real']:.2f} synth={m['synth']:.2f}")
    if not results:
        print('No results collected.')
        return
    # pick the scale with ratio closest to 1.0
    best = min(results, key=lambda x: abs(x['ratio'] - 1.0))
    print('\nRecommended vol-scale:', f"{best['scale']:.2f}", 'ratio=', f"{best['ratio']:.3f}")
    # save results
    os.makedirs('charts', exist_ok=True)
    with open('charts/vol_scale_sweep.json', 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'best': best}, f, indent=2)
    print('Saved sweep results to charts/vol_scale_sweep.json')


if __name__ == '__main__':
    main()
