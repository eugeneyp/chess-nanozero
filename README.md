# chess-nanozero

An AlphaZero-style chess engine trained on Lichess Elite games. Uses Monte Carlo
Tree Search (MCTS) guided by a dual-head residual network — no alpha-beta search,
no hand-crafted evaluation. Trained entirely via supervised learning on elite
human games. Reached ~2000 ELO after 21 hours of GPU training on ~200K games.

**[Play it live on Render →](https://chess-nanozero.onrender.com/)**

---

## Results

| Model | Training data | Est. ELO | Benchmark |
|-------|--------------|----------|-----------|
| medium (step 4) | 5M positions | ~1608 | 37% vs Stockfish 1700 (50 games) |
| medium (step 6) | 20.6M positions | **~2000** | 51% vs Stockfish 2000 (50 games) |

Target was 1500+ ELO. Achieved ~2000 ELO on a ~$15–27 GCP training budget.

All benchmarks use tc=60+1 (60s + 1s/move increment) with the UHO_4060_v4 opening book.

---

## Architecture

Dual-head ResNet following AlphaZero's design. Input is the current board
state encoded as 18 planes of 8×8.

```
Input: (batch, 18, 8, 8)
  │
  └─ Trunk: Conv(18→F, 3×3) → BN → ReLU
            × N residual blocks: Conv(F→F) → BN → ReLU → Conv → BN → skip → ReLU
  │
  ├─ Policy head: Conv(F→32, 1×1) → BN → ReLU → Conv(32→73, 1×1) → flatten → (4672,)
  └─ Value head:  Conv(F→1, 1×1)  → BN → ReLU → flatten → FC(64,256) → ReLU → FC(256,1) → tanh
```

**Configurations:**

| Config | Blocks | Filters | Parameters |
|--------|--------|---------|------------|
| tiny   | 4      | 64      | ~500K      |
| medium | 8      | 128     | ~3M        |

**Input encoding (18 planes):**

| Planes | Content |
|--------|---------|
| 0–5    | Current player pieces: P, N, B, R, Q, K |
| 6–11   | Opponent pieces: P, N, B, R, Q, K |
| 12–13  | Current player castling rights (K-side, Q-side) |
| 14–15  | Opponent castling rights |
| 16     | En passant square |
| 17     | Halfmove clock / 100 |

Board is always from the current player's perspective (rank-mirror for Black).

**Policy encoding (4672 moves = 73 planes × 64 squares):**
- Planes 0–55: Queen moves (8 directions × 7 distances); also covers pawn pushes, king moves, queen promotions
- Planes 56–63: Knight moves (8 L-shaped jumps)
- Planes 64–72: Underpromotions (N/B/R × 3 directions)

---

## Training

### Data

[Lichess Elite Database](https://database.nikonoel.fr/) — games where both
players are rated 2300+ (one must be 2500+), excluding bullet (rating thresholds
raised in December 2021). Each monthly file contains 100K–500K games.

Only positions after the first 6 half-moves are used (openings are too
theory-dependent). Game results (+1 / 0 / −1) are from the current player's
perspective.

### Training runs

Both runs on GCP NVIDIA L4 GPU (g2-standard-4, on-demand):

| Run | Positions | Est. games | Epochs | Duration | Cost | Best val_top1 |
|-----|-----------|------------|--------|----------|------|---------------|
| Step 4 | 5M | ~50K | 30 | 5.2 hrs | ~$7 | 61.0% (epoch 24) |
| Step 6 | +15.6M | ~156K | 25 | 15.7 hrs | ~$20 | 54.4% (epoch 22) |

**Total: ~21 GPU-hours, ~$27 (on-demand at ~$1.30/hr).**

Step 6 fine-tuned from the step 4 checkpoint (lr=5e-4 → 1.25e-4 cosine).
Near-zero train/val gap (1.89 vs 1.92) confirms no overfitting on 20.6M positions.

### Hyperparameters

```yaml
batch_size: 2048
learning_rate: 0.001      # cosine annealing
weight_decay: 0.0001
value_loss_weight: 1.0
skip_first_n_moves: 6
```

Loss: cross-entropy on policy + MSE on value.

---

## Setup

**Requirements:** Python 3.10+, PyTorch 2.0+

```bash
git clone https://github.com/eugeneyp/chess-nanozero
cd chess-nanozero
pip install -e ".[dev]"
```

Run tests to verify:

```bash
pytest tests/ -v
# 50 tests across encoding, model, MCTS, training pipeline, UCI, web
```

---

## Usage

### Play via web interface

```bash
uvicorn web.app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Or use the live demo: **https://chess-nanozero.onrender.com/**

### Play via UCI

```bash
python -u scripts/play_uci.py \
    --config configs/medium.yaml \
    --checkpoint models/medium2.pt
```

Compatible with any UCI-capable GUI (Arena, CuteChess, etc.).

### Prepare training data

```bash
# Download a monthly PGN from https://database.nikonoel.fr/
# Extract positions (auto-chunked into 500K-position .npz files):
python3 scripts/prepare_data.py \
    --pgn data/lichess_elite/lichess_elite_2025-11.pgn \
    --output data/lichess_elite/sample_5m.npz \
    --max-positions 5000000

# Extract next batch (non-overlapping):
python3 scripts/prepare_data.py \
    --pgn data/lichess_elite/lichess_elite_2025-11.pgn \
    --output data/lichess_elite/sample_rest.npz \
    --skip-positions 5000000
```

### Train

```bash
# Single file (<=500K positions):
python3 scripts/train_supervised.py \
    --config configs/medium.yaml \
    --data data/lichess_elite/sample_50k.npz \
    --checkpoint-dir checkpoints/

# Multiple chunks (>500K positions, one chunk loaded at a time to avoid OOM):
python3 scripts/train_supervised.py \
    --config configs/medium.yaml \
    --data data/lichess_elite/sample_5m_part*.npz \
    --checkpoint-dir checkpoints/

# Resume / fine-tune from an existing checkpoint:
python3 scripts/train_supervised.py \
    --config configs/medium.yaml \
    --data data/lichess_elite/sample_rest_part*.npz \
    --checkpoint-dir checkpoints/step6/ \
    --resume checkpoints/step4/epoch_0024.pt
```

### Benchmark vs Stockfish

Requires `stockfish` and `fastchess` binaries in `/usr/local/bin/`.

```bash
./tools/run_tournament.sh [rounds=10] [sf_elo=1320] [tc_base=60] [tc_inc=1] [checkpoint=...]

# Examples:
./tools/run_tournament.sh 25 1700                              # 50 games vs SF 1700
./tools/run_tournament.sh 25 2000 60 1 models/medium2.pt      # 50 games vs SF 2000
```

### Export to ONNX

```bash
python3 scripts/export_onnx.py \
    --checkpoint models/medium2.pt \
    --config configs/medium.yaml \
    --output models/medium2.onnx
```

---

## Project Structure

```
src/
  game/           chess_game.py, encoding.py   — board + move encoding
  neural_net/     model.py, losses.py           — ResNet, AlphaZero loss
  mcts/           mcts.py, node.py              — PUCT search
  agents/         alphazero_agent.py            — wraps MCTS + model
  training/
    supervised/   prepare_data.py, dataset.py, trainer.py
  uci/            uci_engine.py                 — UCI protocol handler
configs/          tiny.yaml, medium.yaml        — model + training configs
scripts/          train_supervised.py, prepare_data.py, play_uci.py, ...
tools/            run_tournament.sh             — fastchess ELO benchmarking
web/              app.py, static/               — FastAPI + chessboard.js
tests/            50 tests across all modules
```

---

## Acknowledgements

**Papers:**
- Silver et al. (2018) — [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- Silver et al. (2017) — [Mastering the Game of Go without Human Knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)

**Projects:**
- [Leela Chess Zero (Lc0)](https://github.com/LeelaChessZero/lc0) — open-source AlphaZero for chess; architecture reference
- [Maia Chess](https://github.com/CSSLab/maia-chess) — human-like chess engine trained on Lichess games
- [neural_network_chess](https://github.com/asdfjkl/neural_network_chess) — practical guide to NN chess engines
- [connect4-alphazero](https://github.com/eugeneyp/connect4-alphazero) — this project's direct predecessor (AlphaZero for Connect 4)
- [chess-ai](https://github.com/eugeneyp/chess-ai) — classical alpha-beta engine; UCI handler and web interface reused here

**Data:**
- [Lichess Elite Database](https://database.nikonoel.fr/) by Nikonoel — 2200+/2400+ rated games, free to download
- [UHO_4060_v4 opening book](https://github.com/official-stockfish/books) — balanced opening positions for engine testing
