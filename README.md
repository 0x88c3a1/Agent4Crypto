# Agent4Crypto

Agent4Crypto is an LLM-driven multi-agent framework for cryptocurrency trading research. It models a small trading desk where specialized agents analyze market structure, news flow, on-chain context, and chart patterns before producing a coordinated trading action.

The repository is organized as a reusable Python package with thin CLI entrypoints at the root, so the project stays easy to navigate on GitHub while still supporting the original `python main.py` workflow.

## Highlights

- Multi-agent pipeline with analyst, risk, trader, and reflector roles
- Multimodal chart inspection using a visual model alongside numerical features
- Reinforcement-learning-based weighting for trader-style aggregation
- Local-data workflow for market, news, and on-chain context

## Repository Layout

```text
Agent4Crypto/
├── agent4crypto/
│   ├── core/                      # Agents, prompts, strategy, RL, experiment utils
│   ├── data/                      # Market/news/on-chain loading
│   ├── ui/                        # Console logging and terminal visuals
│   └── runners/                   # Main entry implementation
├── scripts/                       # Research utilities and one-off runners
├── docs/                          # Project notes and release docs
├── data/                          # Cached market/news/on-chain data
├── config.yaml                    # Main experiment configuration
├── pyproject.toml                 # Package metadata
└── main.py                        # Backward-compatible root entrypoint
```

## Installation

### Prerequisites

- Python 3.9+
- An OpenAI API key

### Setup

```bash
git clone <your-repository-url>
cd Agent4Crypto
pip install -e .
```

If you prefer not to install as a package, the root entry scripts still work directly as long as the dependencies are available in your environment.

## Configuration

Configure `config.yaml` before running experiments.

### API keys

```yaml
api_keys:
  openai: "YOUR_LLM_API_KEY"
  base_url: "https://api.openai.com/v1"
```

### Model selection

```yaml
agent_params:
  strategy_mode: "rl"
  model: ""
  visual_model: ""
```

Fill in `model` and `visual_model` with the OpenAI model names you want to use before running experiments. If `visual_model` is left blank in your local setup, the code will reuse `model`.

### Assets and windows

Define validation and evaluation windows under the `assets` section for each trading pair you want to test.
Repeated evaluation uses `backtest.run_seeds`, which is an explicit fixed seed list in `config.yaml`.

## Usage

Run the main Agent4Crypto pipeline:

```bash
python main.py
```

For repeated weighting experiments, use:

```bash
python scripts/run_repeated_weighting_experiments.py
```

The repeated-run script reads the fixed seed list from `backtest.run_seeds`. One-off runs still work through `python main.py`; when `backtest.run_seed` is unset, the single-run entrypoint uses the first seed from that list.

## Notes

- Root-level scripts are intentionally thin wrappers. Reusable implementation lives under `agent4crypto/`.
- Input data lives under `data/`, while output folders such as `results/`, `logs/`, and `debug_charts/` are treated as local artifacts rather than source files.
- Release and anonymization notes live under `docs/`.

## Disclaimer

This repository is for research and educational use only. It does not constitute financial advice, and live trading based on these outputs may lead to losses.
