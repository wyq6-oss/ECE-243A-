import sys, os, copy
sys.path.append('/home/jupyter/neural_seq_decoder-master/src')

from neural_decoder.neural_decoder_trainer_trans import trainModel

BASE_OUTPUT_ROOT = "/home/jupyter/baseline_logs/"
DATASET_PATH = "/home/jupyter/neural_seq_decoder-master/competitionData/ptDecoder_ctc"

N_BATCH = 10000   # use 3000 for quick screen, 10000 for final
SEED = 0

base = dict(
    outputDir=None,
    datasetPath=DATASET_PATH,
    seqLen=150,
    maxTimeSeriesLen=1200,
    batchSize=128,

    nUnits=256,
    nBatch=N_BATCH,
    nLayers=5,
    seed=SEED,
    nClasses=40,
    nInputFeatures=256,
    dropout=0.2,

    gaussianSmoothWidth=2.0,
    strideLen=4,
    kernelLen=32,
    bidirectional=False,

    # ---- NEW: transform ----
    logTransform="signed_log1p",  # "none" | "signed_log1p" | "log1p_nonneg"

    # ---- optimizer defaults (AdamW target) ----
    useAdamW=True,         # your trainer should branch on this (or just swap manually)
    lrStart=0.002,
    lrEnd=0.0002,
    l2_decay=1e-3,

    # ---- augmentations (smaller after zscore) ----
    whiteNoiseSD=0.3,
    constantOffsetSD=0.08,

    # ---- architecture knobs (must be supported by your model.py) ----
    useLayerNorm=False,
    usePreLayerNorm=False,
    usePostNet=False,
    postNetLayers=3,
    postNetDim=256,
    postNetDropout=0.2,
    useDayLayer=True,
    usePerDayInputLayer=False,
)

experiments = [
    dict(label="T_base", useLayerNorm=False, usePostNet=False),

    dict(label="T_ln",   useLayerNorm=True,  usePostNet=False),

    dict(label="T_post", useLayerNorm=True,  usePostNet=True,
         postNetLayers=3, postNetDropout=0.2, postNetDim=256),

    dict(label="T_post4", useLayerNorm=True, usePostNet=True,
         postNetLayers=4, postNetDropout=0.25, postNetDim=256),

    dict(label="T_ln_lownoise", useLayerNorm=True, usePostNet=False,
         whiteNoiseSD=0.2, constantOffsetSD=0.05),

    dict(label="T_ln_highnoise", useLayerNorm=True, usePostNet=False,
         whiteNoiseSD=0.5, constantOffsetSD=0.12),

    # experimental: remove day layer with pre-LN
    dict(label="T_preLN_noDay", usePreLayerNorm=True, useLayerNorm=True,
         useDayLayer=False, usePostNet=True, postNetLayers=3, postNetDropout=0.2),
]

for exp in experiments:
    args = copy.deepcopy(base)
    args.update(exp)

    modelName = f"speechBaseline4_{exp['label']}_log{args['logTransform']}"
    args["outputDir"] = os.path.join(BASE_OUTPUT_ROOT, modelName)

    print("\n========================================")
    print("Training:", modelName)
    print(" logTransform    =", args["logTransform"])
    print(" LN(pre/post)    =", args["usePreLayerNorm"], args["useLayerNorm"])
    print(" PostNet         =", args["usePostNet"], "layers", args["postNetLayers"], "drop", args["postNetDropout"])
    print(" DayLayer        =", args["useDayLayer"])
    print(" noise/offset    =", args["whiteNoiseSD"], args["constantOffsetSD"])
    print(" lrStart/lrEnd   =", args["lrStart"], args["lrEnd"], "wd", args["l2_decay"])
    print(" nBatch          =", args["nBatch"])
    print("========================================\n")

    trainModel(args)
