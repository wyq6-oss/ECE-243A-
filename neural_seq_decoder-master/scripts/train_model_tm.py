# Detailed time-mask + feature-mask + noise augmentation tuning for speechBaseline4

import sys
sys.path.append('/home/jupyter/neural_seq_decoder-master/src')

from neural_decoder.neural_decoder_trainer_tm import trainModel

BASE_OUTPUT_ROOT = "/home/jupyter/baseline_logs/"
DATASET_PATH = "/home/jupyter/neural_seq_decoder-master/competitionData/ptDecoder_ctc"

# --------------------------------------------------------
# Experiment grid:
#   - different white noise levels
#   - different timeMaskWidth / nTimeMasks
#   - different featMaskWidth / nFeatMasks
#   - includes: baseline, time-only, feature-only, time+feature
# --------------------------------------------------------

experiments = [
    # -------- Baselines (no time mask, no feature mask) --------
    # {
    #     "label": "base_wn0.8_tm0x0_fm0x0",
    #     "whiteNoiseSD": 0.8,
    #     "constantOffsetSD": 0.2,
    #     "timeMaskWidth": 0,
    #     "nTimeMasks": 0,
    #     "featMaskWidth": 0,
    #     "nFeatMasks": 0,
    # },
    # {
    #     "label": "base_wn1.0_tm0x0_fm0x0",
    #     "whiteNoiseSD": 1.0,
    #     "constantOffsetSD": 0.2,
    #     "timeMaskWidth": 0,
    #     "nTimeMasks": 0,
    #     "featMaskWidth": 0,
    #     "nFeatMasks": 0,
    # },

    # -------- Time-only masks (no feature mask) --------
    # {
    #     "label": "wn0.8_tm10x1_fm0x0",
    #     "whiteNoiseSD": 0.8,
    #     "constantOffsetSD": 0.2,
    #     "timeMaskWidth": 10,
    #     "nTimeMasks": 1,
    #     "featMaskWidth": 0,
    #     "nFeatMasks": 0,
    # },
    {
        "label": "wn1.0_tm10x1_fm0x0",
        "whiteNoiseSD": 1.0,
        "constantOffsetSD": 0.2,
        "timeMaskWidth": 10,
        "nTimeMasks": 1,
        "featMaskWidth": 0,
        "nFeatMasks": 0,
    },
#     {
#         "label": "wn0.8_tm20x2_fm0x0",
#         "whiteNoiseSD": 0.8,
#         "constantOffsetSD": 0.2,
#         "timeMaskWidth": 20,
#         "nTimeMasks": 2,
#         "featMaskWidth": 0,
#         "nFeatMasks": 0,
#     },
#     {
#         "label": "wn1.0_tm20x2_fm0x0",
#         "whiteNoiseSD": 1.0,
#         "constantOffsetSD": 0.2,
#         "timeMaskWidth": 20,
#         "nTimeMasks": 2,
#         "featMaskWidth": 0,
#         "nFeatMasks": 0,
#     },

#     # -------- Feature-only masks (no time mask) --------
#     {
#         "label": "wn0.8_tm0x0_fm8x1",
#         "whiteNoiseSD": 0.8,
#         "constantOffsetSD": 0.2,
#         "timeMaskWidth": 0,
#         "nTimeMasks": 0,
#         "featMaskWidth": 8,   # mask up to 8 contiguous features
#         "nFeatMasks": 1,
#     },
#     {
#         "label": "wn0.8_tm0x0_fm16x1",
#         "whiteNoiseSD": 0.8,
#         "constantOffsetSD": 0.2,
#         "timeMaskWidth": 0,
#         "nTimeMasks": 0,
#         "featMaskWidth": 16,
#         "nFeatMasks": 1,
#     },

#     # -------- Time + Feature masks together --------
#     {
#         "label": "wn0.8_tm10x1_fm8x1",
#         "whiteNoiseSD": 0.8,
#         "constantOffsetSD": 0.2,
#         "timeMaskWidth": 10,
#         "nTimeMasks": 1,
#         "featMaskWidth": 8,
#         "nFeatMasks": 1,
#     },
#     {
#         "label": "wn0.8_tm20x2_fm8x1",
#         "whiteNoiseSD": 0.8,
#         "constantOffsetSD": 0.2,
#         "timeMaskWidth": 20,
#         "nTimeMasks": 2,
#         "featMaskWidth": 8,
#         "nFeatMasks": 1,
#     },
#     {
#         "label": "wn1.0_tm20x2_fm8x1",
#         "whiteNoiseSD": 1.0,
#         "constantOffsetSD": 0.2,
#         "timeMaskWidth": 20,
#         "nTimeMasks": 2,
#         "featMaskWidth": 8,
#         "nFeatMasks": 1,
#     },
#     {
#         "label": "wn1.0_tm40x2_fm8x2",
#         "whiteNoiseSD": 1.0,
#         "constantOffsetSD": 0.2,
#         "timeMaskWidth": 40,
#         "nTimeMasks": 2,
#         "featMaskWidth": 8,
#         "nFeatMasks": 2,
#     },
]

# You can shorten this for quick tests (e.g., nBatch=1000),
# then increase to 10000 once you see which configs look promising.
N_BATCH = 10000 # change to 10000 for full runs

for exp in experiments:
    label = exp["label"]
    wn   = exp["whiteNoiseSD"]
    co   = exp["constantOffsetSD"]
    tmw  = exp["timeMaskWidth"]
    nmt  = exp["nTimeMasks"]
    fmw  = exp["featMaskWidth"]
    nfm  = exp["nFeatMasks"]

    modelName = f"speechBaseline4_{label}"
    outputDir = BASE_OUTPUT_ROOT + modelName

    args = {}

    # paths
    args["outputDir"] = outputDir
    args["datasetPath"] = DATASET_PATH

    # core model / training hyperparams (same as your base script)
    args["seqLen"] = 150
    args["maxTimeSeriesLen"] = 1200
    args["batchSize"] = 128

    # using original Adam settings (as in your trainer)
    args["lrStart"] = 0.05
    args["lrEnd"]   = 0.02
    # If you switch trainer to AdamW later, change to

    args["nUnits"] = 256
    args["nBatch"] = N_BATCH
    args["nLayers"] = 5
    args["seed"] = 1
    args["nClasses"] = 40
    args["nInputFeatures"] = 256
    args["dropout"] = 0.2

    # augmentations (tuned)
    args["whiteNoiseSD"] = wn
    args["constantOffsetSD"] = co
    args["gaussianSmoothWidth"] = 2

    # time mask params (used by apply_time_mask in trainer)
    args["timeMaskWidth"] = tmw
    args["nTimeMasks"] = nmt

    # NEW: feature mask params (used by apply_feature_mask in trainer)
    args["featMaskWidth"] = fmw
    args["nFeatMasks"] = nfm

    # temporal params
    args["strideLen"] = 4
    args["kernelLen"] = 32
    args["bidirectional"] = False

    # regularization
    args["l2_decay"] = 1e-5

    print("\n========================================")
    print(f"Training model: {modelName}")
    print(f" whiteNoiseSD      = {wn}")
    print(f" constantOffsetSD  = {co}")
    print(f" timeMaskWidth     = {tmw}")
    print(f" nTimeMasks        = {nmt}")
    print(f" featMaskWidth     = {fmw}")
    print(f" nFeatMasks        = {nfm}")
    print(f" nBatch            = {N_BATCH}")
    print("========================================\n")

    trainModel(args)
