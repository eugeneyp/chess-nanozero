# CLAUDE.md - AlphaZero Chess

## Project Overview

AlphaZero-style chess engine: MCTS + dual-head ResNet, trained via supervised
learning on elite human games (optional self-play RL fine-tuning). Direct port
of connect4-alphazero to chess.

**Goal:** Neural network evaluation + MCTS search (no alpha-beta), trained on
Lichess Elite Database, deployable as Lichess bot and web interface.

**Target:** 1500+ ELO | **Budget:** $300 GCP credits | **Inference:** 5 sec/move on CPU

---

## Related Repositories

- **chess-ai** (github.com/eugeneyp/chess-ai) - Classical engine: alpha-beta,
  PeSTO eval, UCI protocol, FastAPI web on Render.com, fastchess benchmarking
  vs Stockfish. ~1580-1640 ELO.
- **connect4-alphazero** (github.com/eugeneyp/connect4-alphazero) - AlphaZero
  for Connect 4: MCTS + ResNet, GCP L4, top 10 Kaggle ConnectX.

---

## What to Reuse

### From chess-ai

- **UCI handler** (interface/uci.py) - adapt for NN search backend. Protocol
  layer identical; only search backend changes (MCTS replaces alpha-beta).
- **FastAPI web** (web/app.py + web/static/) - stateless REST: POST /api/move
  with FEN returns bestmove. Swap engine backend. chessboard.js frontend as-is.
- **Render deployment** (Procfile) - same architecture.
- **fastchess benchmarking** (tools/run_tournament.sh) - wraps fastchess for
  ELO measurement vs Stockfish with UCI_LimitStrength. Takes any UCI engine.
- **ELO tracking** (tools/elo_tracker.py) - log and plot progression.
- **Tests** (tests/test_uci.py, tests/puzzles/mate_in_2.epd)
- **Design patterns:** python-chess for board logic, push/pop for make/unmake,
  stateless REST (FEN in bestmove out), centipawn integers for UCI info score,
  5-second move constraint.

### From connect4-alphazero

- **MCTS** (src/mcts/) - game-agnostic. PUCT, backprop, virtual loss, Dirichlet
  noise, temperature. Adapt game interface only.
- **Training loop** - Coach, Trainer, ReplayBuffer, Arena.
- **ResNet dual-head** - same pattern, different dimensions.
- **Config system** - YAML with tiny/medium/large.
- **ONNX export, GCP infra, test structure.**

---

## Input Encoding: 18 planes of 8x8

Current player perspective. Rank-mirror the board when Black's turn (`sq ^ 56`,
vertical flip only — files unchanged), swap piece colors. Same approach as
Connect 4. Note: this is NOT a 180° rotation; files stay in place so
kingside/queenside orientation is preserved.

| Planes | Content |
|--------|---------|
| 0-5    | Current player: Pawn, Knight, Bishop, Rook, Queen, King |
| 6-11   | Opponent: Pawn, Knight, Bishop, Rook, Queen, King |
| 12     | Current player kingside castling (all 1s or 0s) |
| 13     | Current player queenside castling |
| 14     | Opponent kingside castling |
| 15     | Opponent queenside castling |
| 16     | En passant square (1 on target square) |
| 17     | Halfmove clock / 100 (constant plane) |

Simplified from AlphaZero 119 planes (no 8-step history). Add later if needed.

---

## Policy Encoding: 73 planes of 8x8 (4672 moves)

AlphaZero action representation:

| Planes | Content |
|--------|---------|
| 0-55   | Queen moves: 8 directions x 7 distances. Also covers pawn pushes, king moves, queen promotions. |
| 56-63  | Knight moves: 8 L-shaped jumps |
| 64-72  | Underpromotions: 3 pieces (N,B,R) x 3 dirs (left-capture, straight, right-capture) |

Index = plane * 64 + from_rank * 8 + from_file.
Illegal moves masked to -inf before softmax, renormalize.

---

## Network Architecture

| Config | Blocks | Filters | Params | Use |
|--------|--------|---------|--------|-----|
| tiny   | 4      | 64      | ~500K  | Pipeline validation, debugging |
| medium | 8      | 128     | ~3M    | Primary training target |
| large  | 12     | 256     | ~15M   | Maximum strength |

```
Input: (batch, 18, 8, 8)
Trunk:  Conv(18->F, 3x3, pad=1) -> BN -> ReLU
        N x [Conv(F->F, 3x3, pad=1) -> BN -> ReLU -> Conv -> BN -> skip -> ReLU]
Policy: Conv(F->32, 1x1) -> BN -> ReLU -> Conv(32->73, 1x1) -> Reshape(4672)
Value:  Conv(F->1, 1x1) -> BN -> ReLU -> Flat -> FC(64,256) -> ReLU -> FC(256,1) -> Tanh
```

---

## Progressive Training (CRITICAL - DO NOT SKIP)

| Step | Model  | Data   | Where     | Time     | Gate                                   |
|------|--------|--------|-----------|----------|----------------------------------------|
| 1    | tiny   | 1K pos | Local CPU | ~5 min   | Pipeline works. Loss decreases.        |
| 2    | tiny   | 50K    | Local     | ~30 min  | Policy acc >5%. MCTS non-random.       |
| 3    | medium | 50K    | GCP GPU   | ~1-2 hrs | Measure speed. Optimize bottlenecks.   |
| 4    | medium | 5M+    | GCP GPU   | ~8-24h   | Real run. Only after Steps 1-3 pass.   |

Step 3 discovers slow data loading, suboptimal batch sizes, encoding
bottlenecks. Fix before committing to 24-hour run.

---

## Supervised Learning (Phase 1 - PRIMARY)

**Data:** Lichess Elite Database (database.nikonoel.fr)
- 2400+ vs 2200+ rated, excluding bullet
- Monthly PGN files, 100K-500K games each, free

**Pipeline:** Download PGN -> parse with python-chess -> extract (board, move,
result) triples -> skip first 6 moves -> encode -> store as .npz

**Loss:** CE(policy_logits, move_target) + MSE(value_pred, game_result)

**Optimizer:** AdamW, lr=1e-3, cosine annealing, weight_decay=1e-4, batch=2048

**Metrics:**
- Policy top-1 accuracy: expect 30-40%
- Policy top-5 accuracy: expect 60-70%
- Value MSE
- Training speed (samples/sec, time/epoch)

---

## Self-Play RL (Phase 2 - OPTIONAL)

Use supervised model as starting checkpoint. Same pipeline as Connect 4.
Budget 100-500 games/iteration, 200-400 MCTS sims/move.

---

## Hyperparameter Tuning Strategy

### When to tune

Do NOT tune during pipeline development (Steps 1-3). Use defaults. Tune only
after baseline model is trained.

### MCTS parameters (tune at inference - no retraining needed)

| Param             | Default | Range    | How to tune |
|-------------------|---------|----------|-------------|
| c_puct            | 2.0     | 1.0-4.0  | fastchess: 50+ games per value vs Stockfish |
| num_simulations   | 400     | 100-1600 | Find sweet spot within 5-sec budget |
| dirichlet_alpha   | 0.3     | 0.1-0.5  | Only if self-play shows exploration collapse |
| dirichlet_epsilon | 0.25    | 0.1-0.4  | Only relevant during self-play |
| temperature       | 1.0->0  | -        | temp=1.0 first 30 moves, then greedy |

**Key:** c_puct and num_simulations interact. Few sims needs higher c_puct
(more exploration per sim). Many sims prefers lower c_puct (exploit). Tune
together via fastchess from chess-ai tools/.

### Training parameters (require retraining)

| Param              | Default | When to change |
|--------------------|---------|----------------|
| learning_rate      | 1e-3    | Reduce if oscillating, increase if plateau |
| batch_size         | 2048    | Larger = stable. GPU memory limited. |
| weight_decay       | 1e-4    | Increase if overfitting |
| value_loss_weight  | 1.0     | Adjust if policy/value imbalanced |
| skip_first_n_moves | 6       | Opening positions less informative |

**Priority:** Tune c_puct via fastchess after training (free, fast). LR
schedule and data quality matter most for training.

---

## Implementation Plan with Required Tests

### Rules for Claude Code

1. **Every phase MUST have passing tests before next phase.** Write tests
   alongside implementation (TDD preferred).
2. **All tests runnable with `pytest tests/ -v`.**
3. **Follow progressive training - do NOT skip to full training.**

### Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| 0     | Partial | pyproject.toml done; configs/tools/CLOUD_TRAINING.md deferred to later phases |
| 1     | ✅ COMPLETE (2026-03-03) | 25/25 tests passing |
| 2     | ✅ COMPLETE (2026-03-03) | 8/8 tests passing (33 total) |
| 3     | ✅ COMPLETE (2026-03-03) | 7/7 tests passing (40 total) |
| 4     | Pending | |
| 5     | Pending | |
| 6     | Pending | |
| 7     | Pending | |
| 8     | Pending | |

### Phase 0: Project Setup (2-3 hrs)

- Create repo chess-alphazero with pyproject.toml
- Deps: torch, python-chess, numpy, pyyaml, tqdm, onnx, onnxruntime, tensorboard
- Copy from Connect 4: config system, checkpoint management, logging
- Copy from chess-ai: tools/run_tournament.sh, tools/elo_tracker.py
- Create configs: tiny.yaml, medium.yaml, large.yaml, supervised.yaml
- Set up pytest

No tests required (scaffolding only).

### Phase 1: Game Engine + Encoding (4-6 hrs)

Implement: chess_game.py (python-chess wrapper for MCTS), encoding.py

**Required tests (test_encoding.py) - ALL MUST PASS before Phase 2:**
- test_encode_starting_position - correct piece planes for initial board
- test_encode_perspective_flip - same pos as white/black, verify flip
- test_encode_castling_rights - various states, planes 12-15
- test_encode_en_passant - plane 16
- test_move_encoding_round_trip - every legal move from 10+ diverse
  positions (opening, middlegame, endgame, castling, en passant, all
  promotion types): verify decode(encode(move)) == move
- test_all_legal_moves_encodable - 20+ random positions, no errors
- test_no_duplicate_indices - no two legal moves share same policy index
- test_queen_promotion_uses_queen_planes - planes 0-55
- test_underpromotion_encoding - planes 64-72

**Required tests (test_chess_game.py):**
- test_legal_moves - matches python-chess
- test_game_result - checkmate, stalemate, draw by repetition, 50-move
- test_make_undo_move - push/pop preserves state

### Phase 2: Neural Network (2-3 hrs) ✅ COMPLETE

Implemented: `src/neural_net/model.py` (ChessResNet, ResBlock, masked_policy_probs),
`src/neural_net/losses.py` (AlphaZeroLoss)

**Implementation notes:**
- Policy head: F→32 (Conv 1x1, BN, ReLU) → 73 (Conv 1x1, bias=True) → flatten to 4672
- Value head: F→1 (Conv 1x1, BN, ReLU) → flatten to 64 → FC(64,256) → ReLU → FC(256,1) → Tanh
- `masked_policy_probs(logits, mask)`: sets illegal indices to `-inf`, then softmax
- `AlphaZeroLoss`: KL-style CE for policy (`-(target * log_softmax).sum`), MSE for value
- `ChessResNet.from_config(cfg)` reads the `model:` section of a YAML config dict
- NUM_PLANES=18 and POLICY_SIZE=4672 imported from `src/game/encoding.py` (single source of truth)
- PyTorch compat: convert numpy bool masks via `.astype(np.uint8)` + `dtype=torch.bool`

**Tests (test_model.py) - ALL PASS:**
- test_forward_pass_shape - (B,18,8,8) -> policy (B,4672), value (B,1)
- test_value_range - output in [-1, +1]
- test_policy_masking - mask illegal, softmax over legal = 1.0
- test_different_configs - tiny/medium/large correct shapes
- test_gradient_flow - backward pass, non-zero grads everywhere
- test_loss_computation - known inputs -> expected loss
- test_from_config - from_config(dict) produces correct architecture
- test_batched_masked_probs - masked_policy_probs works on batch of 4

### Phase 3: Data Pipeline + Supervised Training (6-10 hrs) ✅ COMPLETE

Implemented: `src/training/supervised/prepare_data.py`, `src/training/supervised/dataset.py`,
`src/training/supervised/trainer.py`, `configs/tiny.yaml`, `configs/medium.yaml`,
`scripts/train_supervised.py`

**Implementation notes:**
- `parse_pgn_to_positions(pgn_path, ...)` accepts both file paths and `io.StringIO` (for tests)
- Games with result `*` (unfinished) are skipped entirely
- Value from current player's perspective: `white_result if board.turn == WHITE else -white_result`
- `ChessDataset`: shuffles with fixed seed, val takes first `val_split` fraction of shuffled indices
- `SupervisedTrainer`: AdamW + CosineAnnealingLR; converts hard move labels to one-hot for `AlphaZeroLoss`
- Top-1 and top-5 accuracy logged each epoch; checkpoint saves model + optimizer + scheduler state
- `scripts/train_supervised.py`: CLI with `--config`, `--data` (multi-file), `--checkpoint-dir`, `--resume`
- Configs (`tiny.yaml`, `medium.yaml`) include `model`, `supervised_training`, `mcts`, and `inference` sections

**Tests (test_data_pipeline.py) - ALL PASS:**
- test_pgn_parsing - Scholar's Mate (7 half-moves) yields 7 positions with skip=0
- test_encoding_consistency - first parsed position matches manual `encode_board(chess.Board())`
- test_dataset_loading - correct tensor shapes and dtypes from saved .npz
- test_train_val_split - lengths sum to total; both splits non-empty
- test_skip_first_n_moves - skip=4 gives exactly 4 fewer positions
- test_result_encoding - White-to-move +1.0, Black-to-move -1.0, draws all 0.0
- test_unfinished_games_skipped - result `*` yields 0 positions

**Then execute progressive training Steps 1-4.**

### Phase 4: MCTS Integration (3-4 hrs)

Port MCTS from Connect 4. Implement alphazero_agent.py.

**Required tests (test_mcts.py) - ALL MUST PASS before Phase 5:**
- test_mcts_legal_moves_only
- test_mcts_checkmate_in_one - finds mate from mate-in-1
- test_mcts_avoids_blunder - doesn't hang piece in obvious position
- test_mcts_visit_counts - non-zero, sum correctly
- test_mcts_deterministic - same seed, same result
- test_inference_speed - >=200 sims in 5 seconds on CPU

### Phase 5: Evaluation + Benchmarking (3-4 hrs)

Implement: uci_agent.py, evaluate.py. Use fastchess from chess-ai.

**Required tests:**
- test_uci_agent_interface - start/stop Stockfish, get bestmove
- test_match_runner - 2-game match without crash

**Benchmarking:** 100+ games vs Stockfish Skill Level 0-5

### Phase 6: UCI Protocol + Lichess Bot (3-4 hrs)

Adapt interface/uci.py from chess-ai. Set up lichess-bot bridge.

**Required tests (adapt from chess-ai):**
- test_uci_protocol - isready->readyok, position->ok, go->bestmove
- test_uci_time_management

### Phase 7: Web Interface + Deployment (3-4 hrs)

Copy web/ from chess-ai. ONNX export. Deploy to Render.com.

### Phase 8: Self-Play RL (optional, ongoing)

Adapt from Connect 4. Compare vs supervised via fastchess.

---

## Project Structure

```
chess-alphazero/
  src/
    game/              chess_game.py, encoding.py, utils.py
    neural_net/        model.py, losses.py
    mcts/              mcts.py, batched_mcts.py, node.py
    training/
      supervised/      prepare_data.py, dataset.py, trainer.py
      selfplay/        self_play.py, coach.py, arena.py, replay_buffer.py
      common/          checkpoint.py, logger.py
    agents/            random_agent.py, alphazero_agent.py, uci_agent.py
    uci/               uci_engine.py (adapted from chess-ai)
    export/            export_onnx.py
  configs/             tiny.yaml, medium.yaml, large.yaml, supervised.yaml
  scripts/             prepare_data.py, train_supervised.py, train_selfplay.py,
                       evaluate.py, export_onnx.py, play_terminal.py, lichess_bot.py
  tools/               run_tournament.sh, elo_tracker.py (from chess-ai)
  tests/               test_encoding.py, test_chess_game.py, test_model.py,
                       test_data_pipeline.py, test_mcts.py, test_evaluation.py,
                       test_uci.py, puzzles/mate_in_2.epd
  web/                 app.py, static/ (from chess-ai)
  CLAUDE.md
  CLOUD_TRAINING.md
  Procfile
  pyproject.toml
  README.md
```

---

## Key Differences from Connect 4

| Aspect          | Connect 4      | Chess              |
|-----------------|----------------|--------------------|
| Board           | 6x7            | 8x8                |
| Input planes    | 3              | 18                 |
| Action space    | 7              | 4672               |
| Game length     | ~36 moves      | ~80 moves          |
| Training        | Self-play only | Supervised primary |
| MCTS sims/move  | 50-400         | 200-800            |
| Dirichlet alpha | 0.6            | 0.3                |
| c_puct          | 1.5-2.0        | 2.0 (tune 1.0-4.0) |

---

## Config Example (medium.yaml)

```yaml
model:
  num_res_blocks: 8
  num_filters: 128
  input_planes: 18
  policy_output_size: 4672

supervised_training:
  data_dir: data/lichess_elite/
  batch_size: 2048
  learning_rate: 0.001
  lr_schedule: cosine
  weight_decay: 0.0001
  num_epochs: 30
  value_loss_weight: 1.0
  val_split: 0.05
  skip_first_n_moves: 6

mcts:
  num_simulations: 400
  c_puct: 2.0
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25
  temperature: 1.0
  temperature_threshold_move: 30

selfplay:
  games_per_iteration: 200
  num_iterations: 50
  arena_games: 40
  arena_threshold: 0.55
  replay_buffer_size: 500000

inference:
  max_time_seconds: 5.0
  default_simulations: 400
  onnx_export: true
```

---

## Dependencies

```
python >= 3.11
torch >= 2.0
python-chess >= 1.10
numpy, pyyaml, tqdm, onnx, onnxruntime, tensorboard
```

For Lichess bot: lichess-bot
For evaluation: Stockfish binary + fastchess

---

## References

- AlphaZero: Silver et al. (2018) "Mastering Chess and Shogi by Self-Play"
- AlphaGo Zero: Silver et al. (2017)
- Neural Networks for Chess: github.com/asdfjkl/neural_network_chess
- Connect 4 AlphaZero: github.com/eugeneyp/connect4-alphazero
- Classical Chess AI: github.com/eugeneyp/chess-ai
- Lc0: github.com/LeelaChessZero/lc0
- Maia Chess: github.com/CSSLab/maia-chess
- Lichess Elite Database: database.nikonoel.fr
- Lichess Open Database: database.lichess.org
