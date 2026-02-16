#!/bin/bash

PID=31221
LOG_FILE="/tmp/me_gate_train.log"

echo "Monitoring ME Gate training (PID: $PID)..."
echo "Log file: $LOG_FILE"
echo ""

while ps -p $PID > /dev/null 2>&1; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Training still running..."
    tail -5 "$LOG_FILE" | grep -E "(✅|▶️|📊|⚠️|❌)" | tail -3
    echo ""
    sleep 120  # Check every 2 minutes
done

echo ""
echo "=========================================="
echo "Training completed at $(date)"
echo "=========================================="
echo ""
echo "Final log tail:"
tail -50 "$LOG_FILE"
echo ""
echo "Checking results..."
if [ -f "results/strategies/me/results.json" ]; then
    echo "✅ results.json found"
    python3 -c "
import json
with open('results/strategies/me/results.json') as f:
    r = json.load(f)
    fa = r.get('failure_analysis', {})
    lift = fa.get('lift_rr_extreme', 0)
    print(f'Lift RR Extreme: {lift:.4f}')
    if lift > 1.0:
        print('✅ Lift > 1.0 - PASS')
    else:
        print('❌ Lift ≤ 1.0 - FAIL')
"
else
    echo "⚠️ results.json not found"
fi
