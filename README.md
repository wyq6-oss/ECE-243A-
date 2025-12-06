Brain-to-Text Benchmark – GRU Decoder with Augmentation Experiments

This repository contains experimental code built on top of the official Brain-to-Text Benchmark ’24 baseline (neural_seq_decoder) for intracortical speech decoding.
The project explores a wide range of modifications to improve phoneme-level decoding from microelectrode array (MEA) recordings, including:

Time-masking regularization

Feature-masking (channel dropout)

White noise & baseline-shift augmentation

Layer normalization

Adam vs AdamW optimizers

Smoothed CTC loss

Diphone output representations

Parameter sweeps over masking width, number of masks, and noise strength

A summary of all attempted modifications can be found in:

neural_seq_decoder-master/modification.txt


The best-performing method achieved 22.5% PER (±0.19) over 5 seeds, improving upon the course baseline 23.5% PER (±0.21) by adjusting augmentation parameters—specifically increasing white-noise augmentation and applying moderate time-masking.

Repository Structure
neural_seq_decoder-master/
│
├── scripts/
│   ├── trainModel.py          # Main training script for GRU models
│   ├── evaluation.py          # Computes validation/test PER
│   ├── run_experiments.py     # Parameter sweep for augmentations (your custom script)
│   └── ...
│
├── neural_decoder/
│   ├── model.py               # Baseline GRU model definition
│   ├── neural_decoder_trainer_tm.py 
│   │                          # Modified trainer supporting time/feature masking
│   └── ...
│
├── competitionData/           # Preprocessed Brain-to-Text dataset (formatCompetitionData.ipynb)
│
├── modification.txt           # List of *all* attempted architectural/augmentation changes
└── README.md                  # This file

Installation
1. Clone the repository
git clone https://github.com/<your-name>/<your-repo>.git
cd neural_seq_decoder-master

2. Create and activate virtual environment
uv venv -p 3.9
source .venv/bin/activate

3. Install dependencies
uv pip install -e .


(Optional) For running notebooks:

uv pip install ipykernel

Training Models

All training scripts are inside:

neural_seq_decoder-master/scripts/


Example: running the baseline GRU

python scripts/trainModel.py \
    --datasetPath ./competitionData/ptDecoder_ctc \
    --outputDir ./logs/baseline_run/


To run your augmentation sweeps:

python scripts/train_model.py


This script loops over combinations of:

whiteNoiseSD

timeMaskWidth, nTimeMasks

featMaskWidth, nFeatMasks

learning rate schedule

GRU seeds

and logs everything to /baseline_logs/

Evaluating Models

Use:

python scripts/evaluation.py --modelDir <path-to-trained-model>


This computes:

PER (Phoneme Error Rate)

CTC loss

best checkpoint statistics

By default, evaluation is run on the benchmark validation set.

Checking the List of All Modifications

All conceptual and experimental tweaks you attempted are documented in:

neural_seq_decoder-master/modification.txt


This file includes:

architectural experiments (layer norm, additional linear layers)

optimizer variants (AdamW, LR decay schedules)

objective-related attempts (smoothed CTC, diphone representations)

augmentation sweeps (time masks, feature masks, noise levels)

negative results and discarded ideas

Dataset

The data used in this project comes from the Brain-to-Text Benchmark ’24, containing intracortical MEA recordings from a participant with ALS attempting speech.

Dataset link:
https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq

Processed data used for training must be generated using:

notebooks/formatCompetitionData.ipynb

Citing Related Work

If you use time-masking augmentation, please cite:

Feghhi et al., 
“Time-masked transformers with lightweight test-time adaptation for neural speech decoding,” 
arXiv, 2025.

Contact

Author: Yuanqin Wang
Bioengineering Department, UCLA
Email: wyq6@g.ucla.edu
