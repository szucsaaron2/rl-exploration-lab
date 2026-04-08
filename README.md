# RL Exploration Lab

[![CI](https://github.com/szucsaaron2/rl-exploration-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/szucsaaron2/rl-exploration-lab/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A comprehensive library of reinforcement learning exploration methods with unified evaluation on MiniGrid environments. **17 exploration methods** — from epsilon-greedy to Go-Explore to SHELM — implemented with a clean, modular architecture and evaluated under the same protocol for fair comparison.

## Methods

| # | Method | Type | Paper | Status |
|---|--------|------|-------|--------|
| 1 | Epsilon-Greedy | Baseline | Sutton & Barto, 2018 | ✅ |
| 2 | UCB | Bonus-based | Lattimore & Szepesvári, 2020 | ✅ |
| 3 | Count-Based | Bonus-based | Bellemare et al., 2016 | ✅ |
| 4 | **RND** | Prediction error | Burda et al., 2018 | ✅ |
| 5 | **ICM** | Curiosity-driven | Pathak et al., 2017 | ✅ |
| 6 | **RIDE** | Impact-driven | Raileanu et al., 2020 | ✅ |
| 7 | **AMIGo** | Goal-conditioned | Campero et al., 2020 | ✅ |
| 8 | **NGU** | Episodic + lifelong | Badia et al., 2020 | ✅ |
| 9 | **NovelD** | Novelty difference | Zhang et al., 2021 | ✅ |
| 10 | **Go-Explore** | Archive-based | Ecoffet et al., 2021 (Nature) | ✅ |
| 11 | CLIP-RND | CLIP + RND | — | ✅ |
| 12 | CLIP-NovelD | CLIP + NovelD | — | ✅ |
| 13 | **Semantic Exploration** | CLIP + NGU | Tam et al., 2022 | ✅ |
| 14 | **L-AMIGo** | Language goals | Li et al., 2022 | ✅ |
| 15 | **L-NovelD** | Language novelty | Li et al., 2022 | ✅ |
| 16 | **SHELM + RND** | Thesis method | Szűcs, 2025 | ✅ |
| 17 | **SHELM + Oracle** | Thesis extension | Szűcs, 2025 | ✅ |

## Quick Start

```bash
# Clone and install
git clone https://github.com/szucsaaron2/rl-exploration-lab.git
cd rl-exploration-lab
pip install -e ".[dev]"

# Run tests (104 tests)
pytest tests/ -v

# PPO baseline — quick sanity check
python scripts/train.py --method none --env Empty-8x8 --steps 50000 --seeds 0

# RND on KeyCorridorS3R2 — main thesis benchmark
python scripts/train.py --method rnd --env KeyCorridorS3R2 --steps 2000000 --seeds 0 1 2 3 4

# Go-Explore (no GPU needed — Phase 1 uses random exploration)
python scripts/train.py --method go_explore --env DoorKey-6x6 --steps 100000

# SHELM + Oracle (thesis proposed improvement)
python scripts/train.py --method shelm_oracle --env KeyCorridorS3R2 --steps 500000

# Full benchmark (all methods x all envs x 5 seeds)
python scripts/evaluate.py --suite full

# Quick benchmark (3 methods, 1 env, 2 seeds — minutes)
python scripts/evaluate.py --suite quick

# Generate comparison plots from results
python scripts/plot_results.py --results results/ --output plots/
```

## Architecture

```
rl_exploration_lab/
├── exploration/              # 17 exploration methods (all drop-in)
│   ├── base.py               #   BaseExploration ABC
│   ├── epsilon_greedy.py     #   No-op baseline
│   ├── ucb.py                #   Upper Confidence Bound
│   ├── count_based.py        #   Visit-count bonuses
│   ├── rnd.py                #   Random Network Distillation
│   ├── icm.py                #   Intrinsic Curiosity Module
│   ├── ride.py               #   Rewarding Impact-Driven Exploration
│   ├── amigo.py              #   Adversarially Motivated Intrinsic Goals
│   ├── ngu.py                #   Never Give Up
│   ├── noveld.py             #   NovelD + ERIR
│   ├── go_explore/           #   Go-Explore (Phase 1 + Phase 2)
│   ├── language/             #   CLIP-RND, CLIP-NovelD, Semantic, L-AMIGo, L-NovelD
│   └── shelm/                #   SHELM + RND, SHELM + Oracle (thesis)
├── envs/                     # MiniGrid wrappers + language oracle
├── networks/                 # Policy, predictors, dynamics, CLIP encoder
├── training/                 # PPO, rollout buffer, training orchestrator
├── evaluation/               # Metrics, cross-seed evaluation, plots
├── configs/                  # 16 experiment configs + defaults
└── tests/                    # 104 unit + integration tests
```

## Design Principles

Every exploration method inherits from `BaseExploration` and implements `compute_intrinsic_reward()` and `update()`. Swap methods with a single CLI flag. Every experiment is a YAML file, seeded and reproducible, using the same PPO implementation and evaluation protocol.

## Benchmark Environments

| Environment | Difficulty | Tests |
|------------|-----------|-------|
| MiniGrid-Empty-8x8 | Easy | Sanity check |
| MiniGrid-DoorKey-6x6 | Medium | Key + door interaction |
| MiniGrid-KeyCorridorS3R2 | Hard | Main thesis benchmark |
| MiniGrid-MultiRoomN6 | Hard | Long-horizon exploration |
| MiniGrid-ObstructedMaze-1Dl | Very Hard | Memory + exploration |

## Background

This project extends the bachelor's thesis *"Towards semantic exploration via implicit language abstraction"* (JKU Linz, Institute for Machine Learning, 2025). The thesis investigated combining Semantic HELM with RND for interpretable exploration in partially observable environments, finding that CLIP's language encoder produces generic tokens lacking task-specific semantics. The `shelm_oracle` method implements the thesis's proposed fix: ground-truth language descriptions from a MiniGrid-specific oracle.

## License

MIT
