#!/bin/bash
# Helper script to test all shell scripts and Python scripts
# Usage: bash scripts_test.sh

echo "=========================================="
echo "FRACFIRE - COMPREHENSIVE SCRIPTS TEST"
echo "=========================================="

# Track results
PASSED=0
FAILED=0

echo ""
echo "Testing shell scripts..."
echo "------------------------"

# Test shell scripts can be parsed
for script in printcode.sh gitr.sh gitp.sh; do
    if [ -f "$script" ]; then
        if bash -n "$script" 2>/dev/null; then
            echo "✓ $script (syntax OK)"
            ((PASSED++))
        else
            echo "✗ $script (syntax error)"
            ((FAILED++))
        fi
    fi
done

# Test Python scripts
echo ""
echo "Testing Python scripts..."
echo "------------------------"

for script in scripts/run_orb_trade_runner.py \
              scripts/plot_trade_setups.py \
              scripts/test_installation.py \
              scripts/compare_real_vs_generator.py \
              scripts/analyze_real_vs_synth.py; do
    if [ -f "$script" ]; then
        if .venv312/Scripts/python.exe -m py_compile "$script" 2>/dev/null; then
            echo "✓ $(basename $script) (syntax OK)"
            ((PASSED++))
        else
            echo "✗ $(basename $script) (syntax error)"
            ((FAILED++))
        fi
    fi
done

# Test module imports
echo ""
echo "Testing module imports..."
echo "------------------------"

modules=(
    "src.core.generator"
    "src.core.detector"
    "src.data.loader"
    "src.ml.training.labeler"
)

for module in "${modules[@]}"; do
    if .venv312/Scripts/python.exe -c "import $module" 2>/dev/null; then
        echo "✓ $module"
        ((PASSED++))
    else
        echo "✗ $module"
        ((FAILED++))
    fi
done

# Summary
echo ""
echo "=========================================="
echo "TEST RESULTS: $PASSED passed, $FAILED failed"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Some tests failed!"
    exit 1
fi
