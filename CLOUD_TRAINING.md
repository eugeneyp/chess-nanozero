# CLOUD_TRAINING.md — GCP Training Guide

## Hardware Recommendation

**Use the existing L4 VM for all training steps.**

| GPU   | VRAM  | Cost/hr | Notes |
|-------|-------|---------|-------|
| L4    | 24 GB | ~$0.70  | Primary choice. Fits batch=2048 easily. Good for Steps 3-4. |
| A100  | 40 GB | ~$2.50  | Only worth it if Step 4 needs to finish faster. ~3.5x cost. |

The L4 has plenty of VRAM for the medium model at batch_size=2048 (~2-3 GB used).
Only upgrade to A100 if the Step 4 runtime is unacceptable after measuring in Step 3.

---

## One-Time VM Setup

SSH into the VM, then run once:

```bash
# 1. Clone the repo
git clone https://github.com/eugeneyp/chess-nanozero.git
cd chess-nanozero

# 2. Install PyTorch with CUDA (L4 uses CUDA 12.x)
pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install project dependencies
pip3 install --upgrade pip setuptools
pip3 install -e ".[dev]"

# 4. Verify GPU is visible
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected output: NVIDIA L4
```

---

## Progressive Training Steps

### Step 3: Medium model, 50K positions (gate: measure speed)

**Purpose:** Verify the full pipeline runs on GPU and measure training speed.
Identify any bottlenecks (data loading, batch size) before committing to Step 4.

**Get data onto VM** — the 50K dataset is already prepared locally (2.2 MB):

```bash
# On the VM — create the data directory first:
mkdir -p ~/chess-nanozero/data/lichess_elite

# From your LOCAL machine:
gcloud compute scp data/lichess_elite/sample_50k.npz \
    <vm-name>:~/chess-nanozero/data/lichess_elite/ \
    --zone=<zone>
```

**Run training:**

```bash
# On the VM:
python3 scripts/train_supervised.py \
    --config configs/medium.yaml \
    --data data/lichess_elite/sample_50k.npz \
    --checkpoint-dir checkpoints/step3/ \
    --log-file logs/step3_medium_50k.csv
```

**Expected speed:** 5,000–15,000 samp/s (vs ~270 samp/s on CPU).

**Gate:** Training completes without errors. Check `logs/step3_medium_50k.csv`:
- `samples_per_sec` tells you how long Step 4 will take
- `val_top1` should be in the same range as Step 2 (~20-25%)

**If GPU utilization is low (<80%)**, increase `num_workers` in `scripts/train_supervised.py`:
```python
num_workers = 0 if device == "cpu" else min(8, torch.get_num_threads())
```

**If VRAM allows**, increase `batch_size` in `configs/medium.yaml` (try 4096):
```bash
nvidia-smi  # check VRAM usage during training
```

---

### Step 4: Medium model, 5M+ positions (real training run, 8-24 hrs)

Only proceed after Step 3 passes and you are satisfied with the speed.

#### Get the PGN file on the VM

Download directly from the Lichess Elite Database (database.nikonoel.fr):

```bash
# On the VM:
wget -P data/lichess_elite/ \
    https://database.nikonoel.fr/lichess_elite_2025-11.zip

# Install unzip if needed
sudo apt install -y unzip

# Decompress
unzip data/lichess_elite/lichess_elite_2025-11.zip -d data/lichess_elite/
# Produces: lichess_elite_2025-11.pgn
```

Alternatively, if you ran `prepare_data.py` locally, upload the chunk files
(10 × ~22 MB = ~220 MB total — faster than uploading the full PGN):

```bash
# On the VM — create the data directory first:
mkdir -p ~/chess-nanozero/data/lichess_elite

# From your LOCAL machine:
gcloud compute scp data/lichess_elite/sample_5m_part*.npz \
    <vm-name>:~/chess-nanozero/data/lichess_elite/ \
    --zone=<zone>
```

#### Extract 5M positions (if using PGN)

5M × 18×8×8 × 4 bytes = 21.5 GB — too large to hold in RAM at once.
The script automatically splits into 500K-position chunks (~2.2 GB each):

```bash
# On the VM (~15-20 min). Produces 10 chunk files:
#   sample_5m_part001.npz ... sample_5m_part010.npz
python3 scripts/prepare_data.py \
    --pgn data/lichess_elite/lichess_elite_2025-11.pgn \
    --output data/lichess_elite/sample_5m.npz \
    --max-positions 5000000
```

Progress is printed every 500K positions so you can confirm it's running.

#### Run Step 4

```bash
# Run in a tmux session so it survives disconnects
tmux new -s training

# Pass all chunk files via shell glob — ChessDataset concatenates them
python3 scripts/train_supervised.py \
    --config configs/medium.yaml \
    --data data/lichess_elite/sample_5m_part*.npz \
    --checkpoint-dir checkpoints/step4/ \
    --log-file logs/step4_medium_5m.csv

# Detach with Ctrl+B, D
# Reattach later with: tmux attach -t training
```

**Monitor progress from local machine:**

```bash
# Tail the CSV log remotely
gcloud compute ssh <vm-name> --zone=<zone> -- \
    tail -f chess-nanozero/logs/step4_medium_5m.csv
```

**Expected runtime:** 8–24 hrs on L4 (depends on speed measured in Step 3).

**Actual Step 4 results (2026-03-04, L4 GPU):**
- Speed: ~8,765 samp/s, ~10.7 min/epoch, ~5 hrs 12 min total (30 epochs)
- Best val_top1: **61.0%** at epoch 24 (`checkpoints/step4/epoch_0024.pt`)
- Final val_top1: 60.4% at epoch 30 (slight overfit after epoch 24)
- Best checkpoint saved as `models/medium1.pt`

---

### Step 5: Fine-tune on same 5M dataset with ReduceLROnPlateau

**Purpose:** Test whether the model has exhausted learning on the 5M dataset.
Loads weights only from step 4's best checkpoint; fresh optimizer at lower LR.
Uses `ReduceLROnPlateau` (adaptive) instead of cosine annealing (fixed schedule).

**Prerequisites:** Same PGN file on VM from Step 4. Pull latest code first:

```bash
git pull  # get scripts/train_step5.py
```

**Run Step 5:**

```bash
tmux new -s step5

python3 scripts/train_step5.py \
    --checkpoint checkpoints/step4/epoch_0024.pt \
    --data data/lichess_elite/sample_5m_part*.npz \
    --checkpoint-dir checkpoints/step5/

# Detach: Ctrl+B, D
```

**Key hyperparameters:**
- LR: 5e-4 (lower than step 4's 1e-3; model already near convergence)
- Patience: 2 (halve LR after 2 epochs with no val_top1 improvement)
- Factor: 0.5 (LR schedule: 5e-4 → 2.5e-4 → 1.25e-4 ...)
- Epochs: 10

**What to watch:**
- `★` in output marks best val_top1 epoch
- If val_top1 never exceeds 61.0% (step 4 best) → dataset is exhausted, get more data
- If val_top1 improves → keep training or expand to more data at higher LR

**Retrieve results:**

```bash
# From LOCAL machine:
gcloud compute scp --recurse \
    <vm-name>:~/chess-nanozero/checkpoints/step5/ \
    checkpoints/step5/ \
    --zone=<zone>

# Copy training CSV log (epoch-by-epoch metrics)
gcloud compute scp \
    "<vm-name>:~/chess-nanozero/logs/step5_medium_5m_*.csv" \
    logs/ \
    --zone=<zone>

# Copy best checkpoint as medium2.pt if it beats 61%
gcloud compute scp \
    <vm-name>:~/chess-nanozero/checkpoints/step5/epoch_XXXX.pt \
    models/medium2.pt \
    --zone=<zone>
```

**Expected runtime:** ~1.7 hrs (same dataset/speed as Step 4, 10 epochs instead of 30).
**Expected cost:** ~$1.20 (L4 at $0.70/hr × 1.7 hrs).

**Result (2026-03-05):** Val_top1 improved only marginally (57-59% vs 61% in step 4).
Dataset is effectively exhausted — step 6 trains on fresh positions from the same PGN.

---

### Step 6: Train on all remaining positions in the PGN (~15.6M)

**Purpose:** The same PGN file has ~15.6M positions total; step 4 used only the first 5M
(~68K games). Training on all remaining ~212K games gives the model 3x more fresh data
it has never seen, which should yield substantial ELO gains toward 2000.

**Step 1: Extract all remaining positions** (on the VM, ~30 min):

```bash
python3 scripts/prepare_data.py \
    --pgn data/lichess_elite/lichess_elite_2025-11.pgn \
    --output data/lichess_elite/sample_rest.npz \
    --skip-positions 5000000
# No --max-positions = extract everything remaining (~15.6M positions)
# Produces: sample_rest_part001.npz ... sample_rest_part031.npz (~31 chunks)
```

`--skip-positions 5000000` fast-forwards past exactly the positions used in step 4,
guaranteeing zero overlap. Progress is printed every 500K positions.

**Step 2: Train** (starting from step 4's best checkpoint):

```bash
tmux new -s step6

python3 scripts/train_step5.py \
    --checkpoint checkpoints/step4/epoch_0024.pt \
    --data data/lichess_elite/sample_rest_part*.npz \
    --checkpoint-dir checkpoints/step6/ \
    --num-epochs 20 \
    --log-file logs/step6_medium_rest.csv

# Detach: Ctrl+B, D
```

Uses `train_step5.py` (ReduceLROnPlateau, lr=5e-4, patience=2).
Starting from step 4 epoch 24 (61.0% val_top1) — not step 5, which barely improved.

**Retrieve results:**

```bash
# From LOCAL machine:
gcloud compute scp --recurse \
    <vm-name>:~/chess-nanozero/checkpoints/step6/ \
    checkpoints/step6/ \
    --zone=<zone>

gcloud compute scp \
    <vm-name>:~/chess-nanozero/logs/step6_medium_rest.csv \
    logs/ \
    --zone=<zone>

# Copy best checkpoint as medium2.pt if it beats 61%
gcloud compute scp \
    <vm-name>:~/chess-nanozero/checkpoints/step6/epoch_XXXX.pt \
    models/medium2.pt \
    --zone=<zone>
```

**Expected runtime:** ~30 min extraction + ~10 hrs training (20 epochs × 15.6M positions).
**Expected cost:** ~$7.50 (L4 at $0.70/hr × ~10.5 hrs).

---

## Retrieving Results

After training completes, copy checkpoints and logs back to local:

```bash
# From your LOCAL machine:
gcloud compute scp --recurse \
    <vm-name>:~/chess-nanozero/checkpoints/step4/ \
    checkpoints/step4/ \
    --zone=<zone>

gcloud compute scp \
    <vm-name>:~/chess-nanozero/logs/step4_medium_5m.csv \
    logs/ \
    --zone=<zone>
```

---

## Cost Estimates

| Step | Model  | Data   | Epochs | Time est.           | L4 cost |
|------|--------|--------|--------|---------------------|---------|
| 3    | medium | 50K    | varies | ~15 min             | <$0.20  |
| 4    | medium | 5M     | 30     | ~5 hrs @ 8758 s/s   | ~$3.50  |
| 5    | medium | 5M     | 10     | ~1.7 hrs @ 8758 s/s | ~$1.20  |
| 6    | medium | 15.6M  | 20     | ~10.5 hrs @ 8880 s/s | ~$7.50  |

Calculation for Step 4: 5M × 30 epochs / 8,758 samp/s ≈ 4 hrs 45 min + overhead.
Calculation for Step 5: 5M × 10 epochs / 8,758 samp/s ≈ 1 hr 35 min + overhead.
Calculation for Step 6: 15.6M × 20 epochs / 8,880 samp/s ≈ 9.7 hrs + ~30 min extraction.

Stop the VM (not delete) between sessions to avoid idle charges:

```bash
gcloud compute instances stop <vm-name> --zone=<zone>
gcloud compute instances start <vm-name> --zone=<zone>
```
