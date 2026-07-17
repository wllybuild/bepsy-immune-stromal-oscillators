# BEPSY

BEPSY models coupled immune and stromal compartments as adaptive Stuart–Landau oscillators. A hierarchical multi-omics encoder supplies phase-amplitude states to a pseudo-Hamiltonian generator with state-dependent dissipation, a dual-rate state-space forecaster, a four-class bifurcation head, and a sparse equation readout. The package supports longitudinal forecasting, terminal-state regime classification, and critical-slowing-down analysis.

## Scope

The primary longitudinal analysis uses HMP2 IBDMDB. TCGA-COAD, TCGA-READ, GSE116222, GSE150115, and GSE181919 supply cross-sectional terminal states. The analysis is retrospective and uses public, de-identified archives; it is not a clinical validation system.

## Installation

Python 3.10 or newer and PyTorch 2.2 or newer are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Conda users can create the supplied environment.

```bash
conda env create -f environment.yml
conda activate bepsy
pip install -e .
```

The container uses CUDA 12.1.

```bash
docker build -t bepsy .
```

## Data

Canonical archive locations, release identifiers, and access terms are collected in `dataset_sources.txt`. HMP2 is represented as nine irregular longitudinal channels. TCGA profiles use variance-stabilized RNA measurements, transformed methylation, and log-ratio copy-number values. Single-cell archives require doublet removal, quality filtering, library-size normalization, log transformation, cell-type assignment, and compartment-frequency construction.

The preparation command accepts a long-format CSV containing `subject`, `time`, and `modality_0` through `modality_8` columns.

```bash
bepsy-prepare data/observations.csv data/processed/subjects.pt --modalities 9
```

Access-controlled files must remain under the applicable archive terms. The project does not redistribute source observations.

## Model

The latent network contains 12 compartments, each represented by radius, phase, and momentum. The generator combines a symplectic energy field, normalized dissipation gradient, adaptive phase coupling, and the Stuart–Landau radial field. Slow radial states evolve on a month scale while phase states evolve on an hour-to-day scale. The classifier distinguishes supercritical Hopf, subcritical Hopf, fold, and no-bifurcation regimes.

The training objective is

`L = Ldata + 0.3 Lham + 0.5 Lradial + 0.001 Lsparse + 0.1 Lterminal + Lbifurcation`.

## Training

The primary settings use AdamW, peak learning rate `3e-4`, weight decay `1e-4`, cosine decay, 10% warmup, batch size 16, 200 epochs, ten Hamiltonian warm-start epochs, mixed precision, and five seeds. The sparse readout is updated every 20 steps.

```bash
bepsy-train --config configs/main.yaml --data data/processed/subjects.pt
```

The paper reports NVIDIA A100 mixed-precision training but does not report GPU count, VRAM allocation, storage, or wall-clock duration. These quantities should be recorded for each run rather than inferred. The base configuration has 18.4 million reported parameters; the equal-budget control has 12.1 million.

## Evaluation

T1 uses leave-one-subject-out validation over 132 HMP2 subjects, excluding the 67 flare-positive subjects from training. T2 uses five folds stratified by source institution and inverse archive-size sampling. T3 calculates lag-one autocorrelation in a pre-specified 60-day window. Primary confidence intervals use 10,000 bias-corrected and accelerated resamples, and subgroup comparisons use Benjamini–Hochberg control at 0.05.

```bash
bepsy-evaluate runs/main.pt data/processed/subjects.pt
```

Expected HMP2 NRMSE values are `0.103 ± 0.014` at 30 days, `0.131 ± 0.022` at 90 days, and `0.158 ± 0.046` at 12 months over five seeds. Expected phase-state AUROC is 0.854 on HMP2, 0.812 on TCGA-COAD, 0.798 on TCGA-READ, 0.847 on GSE116222, 0.789 on GSE150115, and 0.836 on GSE181919. The expected critical-slowing exponent is 0.504 with a 95% interval of 0.39–0.62.

## Ablations

`configs/ablations.yaml` enumerates the equal-budget model, removals of the Hamiltonian generator, Stuart–Landau prior, dissipation potential, multiplex coupling, dual-rate forecaster, classifier, sparse readout, the joint physics removal, and a random-coupling null. Each entry is intended to change only the named component.

## Repository map

`code/bepsy/model` contains the encoder, generator, forecaster, heads, and assembled world model. `code/bepsy/data` handles irregular subject records and masked transformations. `code/bepsy/cohorts.py` supplies archive-specific channel transformations and continuous-time alignment. `code/bepsy/baselines.py` contains the forecasting control families. `code/bepsy/analysis.py` implements the reported trajectory, phase, calibration, coupling, and stability measures. `code/bepsy/synthetic.py` defines the Stuart–Landau calibration regimes. `code/bepsy/training` contains optimization and atomic state persistence. `code/bepsy/commands` provides preparation, training, and evaluation entry points. Experiment values live under `configs`, and verified archive URLs live only in `dataset_sources.txt`.

## License

The source code is distributed under the BSD 3-Clause License. Dataset licenses and access agreements remain independent.
