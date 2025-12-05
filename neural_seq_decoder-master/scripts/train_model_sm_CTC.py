# Fine-tune label-smoothed CTC for speechBaseline4

import sys
sys.path.append('/home/jupyter/neural_seq_decoder-master/src')

from neural_decoder.neural_decoder_trainer_sm_CTC import trainModel

BASE_OUTPUT_ROOT = "/home/jupyter/baseline_logs/"
DATASET_PATH = "/home/jupyter/neural_seq_decoder-master/competitionData/ptDecoder_ctc"

# --------------------------------------------------------
# Experiment grid:
#   - different CTC label-smoothing strengths
#   ctc_smoothing = 0.0  -> standard CTC (baseline)
#                   0.05 -> mild smoothing
#                   0.10 -> stronger smoothing
# --------------------------------------------------------

experiments = [
    {"label": "sm0.02", "ctc_smoothing": 0.02},
    {"label": "sm0.04", "ctc_smoothing": 0.04},
    {"label": "sm0.06", "ctc_smoothing": 0.06},
    {"label": "sm0.08", "ctc_smoothing": 0.08},
    {"label": "sm0.1", "ctc_smoothing": 0.1},
    {"label": "sm0.12", "ctc_smoothing": 0.12},
    {"label": "sm0.14", "ctc_smoothing": 0.14},
    {"label": "sm0.16", "ctc_smoothing": 0.16},
    {"label": "sm0.18", "ctc_smoothing": 0.18},
    {"label": "sm0.2", "ctc_smoothing": 0.2},
]

N_BATCH = 10000  # can bump to 10000 once you see which smoothing works best

for exp in experiments:
    label = exp["label"]
    smoothing = exp["ctc_smoothing"]

    modelName = f"speechBaseline4_{label}"
    outputDir = BASE_OUTPUT_ROOT + modelName

    args = {}

    # paths
    args["earlyStopPatienceEvals"] = 20   # 25 evals * 100 batches = 2500 batches
    args["earlyStopMinDelta"] = 5e-4
    args["earlyStopWarmupEvals"] = 10
    args["earlyStopStartBatch"] = 5000   # don’t start plateau check too early
    args["evalEvery"] = 100

    args["outputDir"] = outputDir
    args["datasetPath"] = DATASET_PATH

    # core model / training hyperparams (same as your base script)
    args["seqLen"] = 150
    args["maxTimeSeriesLen"] = 1200
    args["batchSize"] = 128

    # optimizer LR (still using Adam as in your trainer)
    args["lrStart"] = 0.05
    args["lrEnd"]   = 0.005
    # if you later switch trainer to AdamW, you can change to:
    # args["lrStart"] = 0.001
    # args["lrEnd"]   = 0.0001

    args["nUnits"] = 256
    args["nBatch"] = N_BATCH
    args["nLayers"] = 5
    args["seed"] = 0
    args["nClasses"] = 40
    args["nInputFeatures"] = 256
    args["dropout"] = 0.2

    # augmentations (same as baseline)
    args["whiteNoiseSD"] = 0.8
    args["constantOffsetSD"] = 0.2
    args["gaussianSmoothWidth"] = 2.0

    # temporal params
    args["strideLen"] = 4
    args["kernelLen"] = 32
    args["bidirectional"] = False

    # regularization
    args["l2_decay"] = 1e-5

    # NEW: label-smoothing strength for CTC
    args["ctc_smoothing"] = smoothing

    print("\n========================================")
    print(f"Training model: {modelName}")
    print(f" ctc_smoothing  = {smoothing}")
    print(f" nBatch         = {N_BATCH}")
    print("========================================\n")

    trainModel(args)
