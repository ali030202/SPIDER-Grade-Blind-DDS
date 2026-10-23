# SPIDER — Grade-Blind Residual MLP++ DDS

This repository isolates the latest compact SPIDER severity-scoring component found in the project workspace: a grade-blind Residual MLP++ that maps frozen VAE latent features to a continuous degeneration severity score (DDS). It is a downstream scoring model; it does not train the MRI VAE or generate the 21-frame LCOVT trajectories.

## Method

- Converts 4,480-dimensional frozen VAE latents into a selected 384-dimensional feature representation.
- Scores each MRI with a residual MLP: 384 → 128, two residual blocks, then 64 → 1.
- Learns relative ordering from adjacent and sampled long-range pairs in generated grade-conditioned trajectories.
- Selects checkpoints using validation trajectory ordering criteria.
- Calibrates the continuous score to a 0–100 DDS range using training-set latent scores.
- Excludes real Pfirrmann grades from model fitting, feature selection, validation checkpoint selection, and score calibration. Optional grade association is a post-freeze analysis.

The model has 123,905 parameters for a 384-dimensional input. The provided checkpoint bundle contains the selected model, feature normalizer, and calibration values required by the prediction command.

## Repository layout

    ResidualMLP_PlusPlus_384/
    ├── residual_mlp_plusplus_dds.py
    ├── run_all.sh
    ├── requirements.txt
    └── checkpoints/
        ├── model_residual_mlp_plusplus.pt
        ├── feature_normalizer.npz
        └── calibration.json

## Setup

    python -m venv .venv
    # Activate the environment, then:
    pip install -r ResidualMLP_PlusPlus_384/requirements.txt

## Training

Training expects separately prepared metadata, VAE latent caches, and generated 21-frame LCOVT trajectories. Those files are not included in this repository. Provide the path to your authorized local SPIDER data workspace:

    SPIDER_ROOT=/path/to/SPIDER-data bash ResidualMLP_PlusPlus_384/run_all.sh

The training script supports explicit paths for metadata, caches, and train/validation trajectory directories. Run:

    python ResidualMLP_PlusPlus_384/residual_mlp_plusplus_dds.py train --help

The training workflow writes generated checkpoints, predictions, and analysis artifacts under its output directory. These are ignored by Git.

## Predict from pre-encoded latent features

The prediction command expects an N × 4,480 latent matrix produced by the same frozen VAE preprocessing and an N-row manifest with an `image_path` column:

    python ResidualMLP_PlusPlus_384/residual_mlp_plusplus_dds.py predict \
      --model-dir ResidualMLP_PlusPlus_384/checkpoints \
      --latents new_latents.npy \
      --manifest new_manifest.csv \
      --output predictions.csv \
      --device cuda

The script outputs `raw_score` and calibrated `DDS`. It does not convert DICOM/JPEG images into VAE latents.

## Scope and interpretation

This release contains the final scoring code and its compact inference assets, not the upstream generator, source images, masks, latent caches, patient metadata, trajectory images, or experiment reports. DDS is a research score anchored to the training latent distribution; trajectory-order consistency alone does not establish clinical validity. It is not a clinical diagnostic tool.
