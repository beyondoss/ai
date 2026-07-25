set -x
cd /home/jared/ai
echo "=== decode (agent-core) ==="
cargo bench -p beyond-ai-agent-core --bench decode 2>&1 | tee perf-baselines/baseline-decode.txt
echo "=== serve_events (agent) ==="
cargo bench -p beyond-ai-agent --bench serve_events 2>&1 | tee perf-baselines/baseline-serve_events.txt
echo "=== persistence (agent) ==="
cargo bench -p beyond-ai-agent --bench persistence 2>&1 | tee perf-baselines/baseline-persistence.txt
echo "=== search (agent) ==="
cargo bench -p beyond-ai-agent --bench search 2>&1 | tee perf-baselines/baseline-search.txt
echo "=== ALL BASELINES DONE ==="
