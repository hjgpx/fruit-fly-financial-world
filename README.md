# The Financial World of a Fruit Fly

Exploring whether a fruit fly connectome can learn, adapt, and survive in a financial market world.

- Goal: use the complete MaleCNS v1.0 connectome (162,517 neurons, 25.1M synapses) as a fixed "body", encode market signals into its sensory neurons, propagate through the fixed anatomy, read out BUY/SELL, and let reward-modulated Hebbian plasticity drive self-evolution.
- Current progress, problems encountered, and status: see [docs/PROGRESS.md](docs/PROGRESS.md) (中文).
- CPU benchmark results: see [docs/benchmark.md](docs/benchmark.md).

## Repository layout

```
benchmark_v0.py / benchmark_v1.py   # sparse propagation CPU benchmarks
docs/benchmark.md                   # benchmark report
docs/PROGRESS.md                    # work-in-progress report (Chinese)
src/market_encoder.py               # causal OHLCV -> sensory neuron encoder
src/brain_runner.py                 # fixed-topology propagation + BUY/SELL readout
src/learning_loop.py                # reward-modulated Hebbian learning loop v0
config/                             # input/output neuron mappings
tests/                              # unittest suite
```

Large data files (`data/raw/`, `data/processed/`, ~1.4 GB) are intentionally not committed; download MaleCNS v1.0 from the official source and regenerate with the pipeline.
