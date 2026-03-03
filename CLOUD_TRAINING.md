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

**Gate (from CLAUDE.md):**
- Policy top-1 accuracy: 30–40% on val
- Policy top-5 accuracy: 60–70% on val

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

| Step | Model  | Data  | Time est.        | L4 cost |
|------|--------|-------|------------------|---------|
| 3    | medium | 50K   | ~15 min          | <$0.20  |
| 4    | medium | 5M    | ~5 hrs @ 8758 s/s | ~$3.50  |

Calculation for Step 4: 5M × 30 epochs / 8,758 samp/s ≈ 4 hrs 45 min + ~10 min chunk-load overhead.

Stop the VM (not delete) between sessions to avoid idle charges:

```bash
gcloud compute instances stop <vm-name> --zone=<zone>
gcloud compute instances start <vm-name> --zone=<zone>
```
