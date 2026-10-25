# Reproducing the study

Inference and evaluation use the top-level scripts. The `reproduction/` directory separately preserves the frozen EDVR/Restormer and NAFNet training implementations. Their original source inventories bind code, draws, statistics, and initializers to exact hashes.

## Restore training assets

Run from the repository root after downloading both Kaggle packages:

```bash
python scripts/prepare_training.py \
  --weights-root /path/to/kaggle-model-weights \
  --data-root /path/to/kaggle-dataset
```

The command copies the deferred training initializers, physical inputs, and natural-image originals into their original relative locations. It checks their hashes and both complete frozen source inventories. It does not start training. The two Kaggle packages remain the source of the model-ready dataset arrays.

## Primary restoration fits

Choose a new results directory. The archived trainers require one compatible CUDA device. For the primary EDVR/Restormer protocol:

```bash
python reproduction/legacy/code/campaign_train.py \
  --run restormer_RGB_mean_mixed \
  --protocol reproduction/legacy/config/CAMPAIGN_v1.json \
  --data-root /path/to/kaggle-dataset \
  --results-root /path/to/new-results
```

The seven run IDs are listed in `reproduction/legacy/config/CAMPAIGN_v1.json`. For NAFNet, use `reproduction/nafnet/code/campaign_train.py` and its matching protocol. Its run IDs are `nafnet_RGB_single_mixed`, `nafnet_RGB_mean_mixed`, and `nafnet_RGB_burst_mixed`.

The frozen legacy source includes classifier training and evaluation. It uses the same five-category label mapping and ImageNet-initialized ResNet50 backbone for seeds 1337, 2027, and 4099. Read the archived protocol before running these phases; selection must precede evaluation. The curated weights package contains restoration checkpoints rather than the recognition classifiers.

`reproduction/seed4099/CAMPAIGN_v1.json` records the completed burst replication recipe. Its source training paths refer to the historical campaign; use it as provenance alongside the two selected weights. The original amendment and incomplete seed-2027 disclosure are in `results/seed_evidence/`.

The release verifies source integrity, strict checkpoint loading, representative forward inference, the data contract, and representative visual output. It does not rerun the full training campaign. Hardware and library differences can affect exact floating-point results. Historical machine paths inside frozen source records are provenance; the top-level inference and evaluation interfaces use explicit local roots.
