#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lab/jmx/SPIDER"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT"

"$PYTHON_BIN" "$ROOT/lcovt_dds_level_specific_pipeline.py" \
  --root "$ROOT" \
  --train-trajectory-dir "$ROOT/results_lcovt_G1_G5_21frames_train_newvae_batch21" \
  --val-trajectory-dir "$ROOT/results_lcovt_G1_G5_21frames_val_newvae_batch21" \
  --metadata-csv "$ROOT/data/prepared_metadata.csv" \
  --stage1-script "$ROOT/train_ordinal_latent_diffusion_v2.py" \
  --diffusion-checkpoint "$ROOT/runs_ordinal_ldm_v2_224x160_newvae/checkpoints/best.pt" \
  --vae-checkpoint "$ROOT/runs_vae_v2_224x160/checkpoints/vae_best.pt" \
  --cache-root "$ROOT/results_lcovt_dds_final" \
  --output-dir "$ROOT/results_lcovt_dds_level_specific" \
  --level-names "L1-L2,L2-L3,L3-L4,L4-L5,L5-S1" \
  --top-k-grid "64,128,256" \
  --min-relevance-grid "0.03,0.05" \
  --feature-chunk-cases 128 \
  --test-batch-size 64 \
  --amp-dtype none \
  --device cuda \
  --bootstrap 1000 \
  --seed 2026
