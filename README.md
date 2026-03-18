# polymarket-backtest

Event-driven backtesting framework for Polymarket updown markets.

## Features
- Tick-by-tick replay using recorded order books
- **Market orders**: walk full book JSON, 2bps taker fee, partial fills
- **Limit orders**: 0 fees (maker), conservative/optimistic queue model
- UP and DOWN token support
- Configurable latency (0ms → 1000ms+)

## Usage
```bash
# Run on VM (data stays on server, no download needed)
ssh ubuntu@YOUR_VM_IP
cd ~/backtest
python3 run_backtest.py --dir ~/pm_watcher/historical_data/live --coin btc --interval 5
```

## Tests
```bash
python3 test_execution.py   # 14 unit tests
python3 validate_engine.py  # oracle + random strategy validation
```
