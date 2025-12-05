# Detailed CR-CTC tuning sweep (STABLE FIRST: alpha + masking + noise) for GRUDecoder
# This version avoids the blank-collapse you saw by using gentler CR/masking/noise.

import sys
sys.path.append('/home/jupyter/neural_seq_decoder-master/src')

from neural_decoder.neural_decoder_trainer_crctc import trainModel

BASE_OUTPUT_ROOT = "/home/jupyter/baseline_logs/"
DATASET_PATH = "/home/jupyter/neural_seq_decoder-master/competitionData/ptDecoder_ctc"

# --------------------------------------------------------
# Experiment grid (stability-first):
#   - smaller alpha early (0.0~0.4)
#   - gentler masking (time <= 0.15, feat <= 0.08)
#   - lower noise
# --------------------------------------------------------

experiments = [
    # -------- Near-CTC baseline (helps verify no collapse) --------
    {
        "label": "A_alpha0.0_mask0_feat0_noise0.10",
        "cr_alpha": 0.0,
        "cr_time_masks": 0,
        "cr_time_max_ratio": 0.0,
        "cr_feat_masks": 0,
        "cr_feat_max_ratio": 0.0,
        "whiteNoiseSD": 0.10,
        "constantOffsetSD": 0.02,
        "gaussianSmoothWidth": 1.0,
    },

    # -------- Mild CR (recommended starting point) --------
    {
        "label": "B_alpha0.2_tm1x10_fm1x05_noise0.15",
        "cr_alpha": 0.2,
        "cr_time_masks": 1,
        "cr_time_max_ratio": 0.10,
        "cr_feat_masks": 1,
        "cr_feat_max_ratio": 0.05,
        "whiteNoiseSD": 0.15,
        "constantOffsetSD": 0.03,
        "gaussianSmoothWidth": 1.0,
    },

    # -------- Moderate CR (still safe) --------
    {
        "label": "C_alpha0.3_tm2x12_fm1x08_noise0.15",
        "cr_alpha": 0.3,
        "cr_time_masks": 2,
        "cr_time_max_ratio": 0.12,
        "cr_feat_masks": 1,
        "cr_feat_max_ratio": 0.08,
        "whiteNoiseSD": 0.15,
        "constantOffsetSD": 0.03,
        "gaussianSmoothWidth": 0.8,
    },

    # -------- Slightly stronger alpha but conservative masking --------
    {
        "label": "D_alpha0.4_tm2x15_fm1x08_noise0.10",
        "cr_alpha": 0.4,
        "cr_time_masks": 2,
        "cr_time_max_ratio": 0.15,
        "cr_feat_masks": 1,
        "cr_feat_max_ratio": 0.08,
        "whiteNoiseSD": 0.10,
        "constantOffsetSD": 0.02,
        "gaussianSmoothWidth": 0.8,
    },

    # -------- Mask-heavy but low alpha (tests “paper-ish” masking safely) --------
    {
        "label": "E_alpha0.2_tm3x15_fm1x08_noise0.10",
        "cr_alpha": 0.2,
        "cr_time_masks": 3,
        "cr_time_max_ratio": 0.15,
        "cr_feat_masks": 1,
        "cr_feat_max_ratio": 0.08,
        "whiteNoiseSD": 0.10,
        "constantOffsetSD": 0.02,
        "gaussianSmoothWidth": 0.5,
    },
]

# You can shorten for quick screening (e.g., 2000),
# then rerun best configs with 10000.
N_BATCH = 10000

BASE_ARGS = {
    "datasetPath": DATASET_PATH,

    "seqLen": 150,
    "maxTimeSeriesLen": 1200,
    "batchSize": 96,

    # Stable Adam learning rates (avoid 0.05)
    "lrStart": 1e-3,
    "lrEnd": 2e-4,

    "nUnits": 256,
    "nBatch": N_BATCH,
    "nLayers": 4,
    "seed": 0,
    "nClasses": 40,
    "nInputFeatures": 256,
    "dropout": 0.20,

    "strideLen": 4,
    "kernelLen": 24,
    "bidirectional": False,

    "l2_decay": 1e-4,
}

for exp in experiments:
    modelName = f"speechBaseline4_crctc_{exp['label']}"
    outputDir = BASE_OUTPUT_ROOT + modelName

    args = dict(BASE_ARGS)
    args["outputDir"] = outputDir

    # Aug/noise
    args["whiteNoiseSD"] = exp["whiteNoiseSD"]
    args["constantOffsetSD"] = exp["constantOffsetSD"]
    args["gaussianSmoothWidth"] = exp["gaussianSmoothWidth"]

    # CR-CTC knobs
    args["cr_alpha"] = exp["cr_alpha"]
    args["cr_time_masks"] = exp["cr_time_masks"]
    args["cr_time_max_ratio"] = exp["cr_time_max_ratio"]
    args["cr_feat_masks"] = exp["cr_feat_masks"]
    args["cr_feat_max_ratio"] = exp["cr_feat_max_ratio"]

    print("\n========================================")
    print(f"Training model: {modelName}")
    print(f" outputDir          = {outputDir}")
    print(f" lrStart -> lrEnd   = {args['lrStart']} -> {args['lrEnd']}")
    print(f" batchSize          = {args['batchSize']}")
    print(f" nUnits / nLayers   = {args['nUnits']} / {args['nLayers']}")
    print(f" kernel/stride      = {args['kernelLen']} / {args['strideLen']}")
    print(f" dropout / l2_decay = {args['dropout']} / {args['l2_decay']}")
    print(f" CR alpha           = {args['cr_alpha']}")
    print(f" time masks         = {args['cr_time_masks']} @ max_ratio {args['cr_time_max_ratio']}")
    print(f" feat masks         = {args['cr_feat_masks']} @ max_ratio {args['cr_feat_max_ratio']}")
    print(f" whiteNoiseSD       = {args['whiteNoiseSD']}")
    print(f" constantOffsetSD   = {args['constantOffsetSD']}")
    print(f" gaussianSmoothWidth= {args['gaussianSmoothWidth']}")
    print(f" nBatch             = {args['nBatch']}")
    print("========================================\n")

    trainModel(args)
