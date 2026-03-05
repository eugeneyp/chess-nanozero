#!/usr/bin/env bash
# run_tournament.sh — chess-nanozero vs Stockfish benchmarking
#
# Usage:
#   ./tools/run_tournament.sh [rounds=10] [sf_elo=1320] [tc_base=60] [tc_inc=1] [checkpoint=...]
#
# Examples:
#   ./tools/run_tournament.sh                              # 20 games smoke test
#   ./tools/run_tournament.sh 25                           # 50 games ELO estimate
#   ./tools/run_tournament.sh 25 1700                      # vs Stockfish 1700
#   ./tools/run_tournament.sh 10 1320 120 2                # 2-min game + 2s increment
#   ./tools/run_tournament.sh 5 1500 60 1 checkpoints/step6/epoch_0010.pt  # custom checkpoint
#
# ELO formula (printed after run):
#   Elo ≈ opponent_elo − 400 × log10((1 − score%) / score%)

set -euo pipefail

ROUNDS=${1:-10}
SF_ELO=${2:-1320}
TC_BASE=${3:-60}    # seconds per player per game
TC_INC=${4:-1}      # seconds increment per move

FASTCHESS=/usr/local/bin/fastchess
STOCKFISH=/usr/local/bin/stockfish
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 5th arg: checkpoint (relative to PROJECT_DIR); empty string = use default
if [ -n "${5:-}" ]; then
    CHECKPOINT="${PROJECT_DIR}/${5}"
else
    CHECKPOINT="${PROJECT_DIR}/checkpoints/step4/epoch_0024.pt"
fi

CONFIG="${PROJECT_DIR}/configs/medium.yaml"
OPENING_BOOK="${PROJECT_DIR}/data/openings/UHO_4060_v4.epd"
RESULTS_DIR="${PROJECT_DIR}/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PGN_OUT="${RESULTS_DIR}/vs_sf${SF_ELO}_${TIMESTAMP}.pgn"

mkdir -p "$RESULTS_DIR"

TOTAL_GAMES=$(( ROUNDS * 2 ))
echo "======================================================"
echo " chess-nanozero vs Stockfish ${SF_ELO}"
echo " Rounds: ${ROUNDS} (${TOTAL_GAMES} games, -repeat)"
echo " Time control: ${TC_BASE}s + ${TC_INC}s/move"
echo " Checkpoint:   ${CHECKPOINT}"
echo " Opening book: ${OPENING_BOOK}"
echo " PGN output:   ${PGN_OUT}"
echo "======================================================"

# Note: fastchess treats cmd= as a binary path only; use args= for arguments.
# Also, fastchess has a bug with RLIM_INFINITY - set a concrete fd limit.
ulimit -n 4096 2>/dev/null || true

ENGINE_ARGS="-u ${PROJECT_DIR}/scripts/play_uci.py --config ${CONFIG} --checkpoint ${CHECKPOINT}"

# Use python -u for unbuffered stdout — required so bestmove reaches fastchess
# without block-buffer delay.  timemargin=500 gives Stockfish 500ms grace.
"$FASTCHESS" \
    -engine \
        cmd="python" \
        "args=${ENGINE_ARGS}" \
        name="nanozero" \
    -engine \
        cmd="$STOCKFISH" \
        name="stockfish-${SF_ELO}" \
        option.UCI_LimitStrength=true \
        option.UCI_Elo=${SF_ELO} \
    -each tc=${TC_BASE}+${TC_INC} timemargin=500 \
    -openings file="${OPENING_BOOK}" format=epd order=random \
    -rounds ${ROUNDS} \
    -repeat \
    -recover \
    -pgnout "file=${PGN_OUT}"

echo ""
echo "Results saved to: $PGN_OUT"
echo ""
echo "ELO formula:"
echo "  Elo ≈ ${SF_ELO} − 400 × log10((1 − score%) / score%)"
echo "  score%=50% → ${SF_ELO}, score%=20% → $((SF_ELO - 280)), score%=10% → $((SF_ELO - 381))"
