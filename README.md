# Riemannian Geometry of Multimodal Biometric Embedding Spaces

Code for the paper accepted for oral presentation at [MathAI 2026](https://mathai.club) (International Conference on Mathematics of Artificial Intelligence).

**[Paper on OpenReview](https://openreview.net/forum?id=SPIdRsn5GD)**

## Overview

We characterize the intrinsic Riemannian geometry of pretrained face and voice encoder embedding spaces and show that geometric properties — particularly Centered Kernel Alignment (CKA) — predict cross-modal person-matching difficulty without any cross-modal training.

Key finding: CKA correlates with cross-modal EER at Spearman &rho; = &minus;0.87 (p < 0.001), with leave-one-out cross-validated R&sup2; = 0.77.

## Encoders

| Modality | Encoder | Architecture | Embedding dim |
|----------|---------|-------------|---------------|
| Face | ArcFace | ResNet-100 | 512 |
| Face | SigLIP | ViT-B/16 | 768 |
| Face | DINOv2 | ViT-B/14 | 768 |
| Face | CLIP | ViT-B/16 | 512 |
| Voice | WavLM | Transformer-Large | 1024 |
| Voice | HuBERT | Transformer-Large | 1024 |
| Voice | wav2vec 2.0 | Transformer-Large | 1024 |

## Setup

```bash
# Create conda environment
conda create -n fm-mmpr python=3.10 -y
conda activate fm-mmpr

# Install dependencies
pip install -e .

# Symlink your datasets
ln -s /path/to/VoxCeleb1 data/VoxCeleb1
ln -s /path/to/MAV-Celeb data/MAV-Celeb
```

## Pipeline

The analysis runs as an 8-step pipeline. Each script reads from `results/` and writes back to `results/`, so steps can be run independently after their dependencies complete.

```bash
# Set PYTHONPATH for all scripts
export PYTHONPATH=.

# Step 1: Extract embeddings for all encoders
python scripts/01_extract_embeddings.py

# Step 2: Estimate intrinsic dimensionality (MLE, TwoNN)
python scripts/02_intrinsic_dimension.py

# Step 3: Estimate local curvature via second fundamental form
python scripts/03_curvature.py

# Step 4: Compute cluster topology (intra/inter-class, compactness gap)
python scripts/04_cluster_topology.py

# Step 5: Compute cross-modal geometry (GW distance, spectral gap, CKA, ID mismatch)
python scripts/05_cross_modal_geometry.py

# Step 6: Compute CCA-aligned cross-modal EER
python scripts/06_cross_modal_eer.py

# Step 7: Run correlation analysis (Spearman, bootstrap CIs, regression)
python scripts/07_correlation_analysis.py

# Step 8: Generate figures and tables
python scripts/08_generate_figures.py
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

76 unit tests covering all geometry and evaluation modules.

## Project Structure

```
src/
  encoders/         # Face and voice encoder wrappers
  geometry/         # Intrinsic dim, curvature, cluster topology, cross-modal metrics
  evaluation/       # CCA-aligned cross-modal EER
scripts/            # 8-step pipeline (01-08)
tests/              # Unit tests
configs/            # geometry.yaml configuration
results/            # Computed metrics (JSON) and figures (PDF)
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{upadhyay2026riemannian,
  title={Riemannian Geometry of Multimodal Biometric Embedding Spaces},
  author={Upadhyay, Alok},
  booktitle={Proceedings of the International Conference on Mathematics of Artificial Intelligence (MathAI)},
  year={2026},
  url={https://openreview.net/forum?id=SPIdRsn5GD}
}
```

## License

MIT
