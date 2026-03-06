# ELO Benchmarking Results

All tournaments use:
- Time control: 60s + 1s/move per player
- Opening book: `data/openings/UHO_4060_v4.epd` (random order)
- fastchess with `timemargin=500`
- PyTorch inference (no ONNX)

---

## Step 4: medium model, 5M positions

**Checkpoint:** `checkpoints/step4/epoch_0024.pt`
**Training:** 30 epochs, best val_top1=61.0% at epoch 24

| Date | Opponent | Games | W | L | D | Score | Est. ELO |
|------|----------|-------|---|---|---|-------|----------|
| 2026-03-05 | Stockfish 1700 | 50 | 18 | 31 | 1 | 37% | ~1608 |

**ELO estimate: ~1608**

---

## Step 6: medium model, +15.6M positions (continued training)

**Checkpoint:** `models/medium2.pt` (step6/epoch_0024.pt)
**Training:** 25 epochs on remaining ~15.6M positions, best val_top1=54.4% at epoch 22, best val_loss at epoch 24
**Starting point:** step 4 checkpoint (fine-tuned, not trained from scratch)

| Date | Opponent | Games | W | L | D | Score | Est. ELO |
|------|----------|-------|---|---|---|-------|----------|
| 2026-03-06 | Stockfish 1700 | 10 | 8 | 2 | 0 | 80% | ~1902 |
| 2026-03-06 | Stockfish 2000 | 50 | 24 | 23 | 3 | 51% | ~2007 |

**ELO estimate: ~2000 ± 104**

---

## Progress Summary

| Model | Data | Est. ELO | vs Target (1500+) |
|-------|------|----------|-------------------|
| step4/epoch_0024.pt | 5M positions | ~1608 | ✅ +108 |
| models/medium2.pt   | 20.6M positions | ~2000 | ✅ +500 |

---

## Notes

- Step 6 val_top1 (54.4%) is not directly comparable to step 4 (61.0%) — different validation sets.
- Step 6 shows near-zero train/val loss gap (1.89 vs 1.92), indicating the medium architecture
  is not overfitting on 15.6M positions and has capacity for more data or a larger architecture.
- Next steps: train large model (12 blocks, 256 filters, ~15M params) or add more training data.
