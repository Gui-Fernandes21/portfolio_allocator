# Reward Shaping in Deep RL for Retail ETF Allocation

CM3070 Final Project — a decision-support tool for retail investors. It
recommends a portfolio across seven ETFs (SPY, QQQ, IWM, EFA, EEM, GLD, TLT)
and explains the recommendation in plain language. It does not execute
trades.

The core contribution is an ablation study comparing four PPO
reward formulations — raw return, Differential Sharpe Ratio, a Sortino-style
downside variant, and a transaction-cost penalty — under walk-forward
validation with five seeds and four test years.

## How the system fits together

```
 OFFLINE (research)                          LIVE (decision support)
 ─────────────────                           ───────────────────────
 data_cache.py   fetch + features, once
       │
 walk_forward.py 4 folds, 5-day embargo
       │
 run_finrl_ablation.py  80 PPO runs  ──►  results/   (evidence)
       │
       └── --final ──► models/ppo_*.zip ──►  FastAPI  ──►  5 agents  ──►  Nuxt UI
                       (the only thing that   /api/v1     pipeline        frontend
                        crosses the boundary)
```

Training happens only offline. The live system loads frozen weights and never
learns, so identical inputs always give identical weights.

### The five agents

| Agent | Uses an LLM | Role |
|---|---|---|
| Market Analyst | yes | Computes recent returns, volatility and drawdown; the model only narrates those numbers |
| Sentiment Analyser | yes | Classifies recent news headlines, keeping the source of every one |
| **Execution Agent** | **no** | Runs the frozen PPO policy to produce the allocation |
| **Risk Manager** | **no** | Applies a 40% per-ETF cap and records every change it makes |
| Coordinator | yes | Writes the explanation; any reply citing a figure not in the allocation is discarded for a verified template |

The allocation is produced and constrained entirely by the two agents with no
LLM. A test (`tests/test_agent_boundaries.py`) enforces this by walking the
import graph, and `GET /api/v1/pipeline` exposes it at runtime.

## Repository map

```
backend/
  offline/           training and evaluation pipeline
    rewards.py           the four reward formulations (the ablation's only variable)
    finrl_portfolio_env.py   FinRL environment; only the reward line is changed
    data_cache.py        fetch prices and engineer features once
    walk_forward.py      fold construction with embargo
    run_finrl_ablation.py    runs the study; --final exports the shipped policies
    aggregate_results.py     metrics, baselines (1/N, SPY) and ranking tables
    cost_sensitivity.py      re-evaluates every run across cost rates
    results/             run log, per-run returns and weights, findings tables
  models/            the four frozen policies + manifest.json
  orchestration/     small in-process agent framework written for this project
  agents/            the five agents
  app/               FastAPI service
  tests/             unit and contract tests
  verify_determinism.py  checks the shipped policies are bit-for-bit reproducible
frontend/            Nuxt web interface
```

## Running it

Requires Python 3.10+ and Node 20+.

**1. Backend**

```bash
cd backend
py -3.12 -m venv .venv  # install virtual environment
pip install -r requirements.txt
cp .env.example .env        # then add your OPENROUTER_API_KEY
uvicorn app.main:app --reload
```

API docs at http://127.0.0.1:8000/docs. Without an API key the system still
works: the allocation, risk checks and a template explanation are produced,
and the response is marked as `degraded`.

**2. Frontend** (in another terminal)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

## Reproducing the results

The study's results are already in `backend/offline/results/`. To verify them
without retraining (seconds, no network):

```bash
cd backend/offline
python aggregate_results.py     # headline tables, per-fold rankings, baselines
python cost_sensitivity.py      # transaction-cost rate sweep
cd ..
python verify_determinism.py    # shipped policies reproduce exactly
python -m unittest discover tests
```

To rerun the full study from scratch (~5 hours on CPU):

```bash
cd backend/offline
python data_cache.py                   # fetch data once (needs network)
python run_finrl_ablation.py --smoke   # quick end-to-end check
python run_finrl_ablation.py --save-weights   # 4 variants × 5 seeds × 4 folds
python aggregate_results.py
python run_finrl_ablation.py --final   # re-export the shipped policies
```

## Main findings

- Risk-shaped rewards (DSR, Sortino) rank above the raw-return reward in every
  test year; DSR wins 17 of 20 paired runs.
- That advantage comes mainly from lower turnover rather than better portfolio
  choice: gross of costs, DSR wins only 13 of 20.
- DSR beats the equal-weight (1/N) benchmark in all four years, at cost rates
  below about 30 basis points.
- The explicit transaction-cost penalty had no measurable effect on behaviour
  at a realistic 10 bps.
- SPY buy-and-hold beat every diversified strategy over 2021–2024 on both
  return and drawdown; the report discusses why.

Full analysis, limitations and references are in the project report.
