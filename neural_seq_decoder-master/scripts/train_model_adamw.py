import sys
sys.path.append('/home/jupyter/neural_seq_decoder-master/src')

from neural_decoder.neural_decoder_trainer_adamw import trainModel

BASE_OUTPUT_ROOT = "/home/jupyter/baseline_logs/"
DATASET_PATH = "/home/jupyter/neural_seq_decoder-master/competitionData/ptDecoder_ctc"

N_BATCH = 10000

# A small, sensible sweep for AdamW on this task
experiments = [
    # conservative / usually stable
    {"label": "adamw_lr1e-3_wd1e-4_eps1e-8",  "lrStart": 1e-3,  "lrEnd": 1e-4,  "l2_decay": 1e-4, "adam_eps": 1e-8},
    {"label": "adamw_lr8e-4_wd1e-4_eps1e-8",  "lrStart": 8e-4,  "lrEnd": 8e-5,  "l2_decay": 1e-4, "adam_eps": 1e-8},

    # slightly stronger weight decay (can help generalization)
    {"label": "adamw_lr1e-3_wd5e-4_eps1e-8",  "lrStart": 1e-3,  "lrEnd": 1e-4,  "l2_decay": 5e-4, "adam_eps": 1e-8},

    # try your repo’s large eps style (sometimes helps stability with noisy grads)
    {"label": "adamw_lr1e-3_wd1e-4_eps1e-3",  "lrStart": 1e-3,  "lrEnd": 1e-4,  "l2_decay": 1e-4, "adam_eps": 1e-3},
]

for exp in experiments:
    modelName = f"speechBaseline4_{exp['label']}"
    args = {}

    # paths
    args["outputDir"] = BASE_OUTPUT_ROOT + modelName
    args["datasetPath"] = DATASET_PATH

    # core
    args["seqLen"] = 150
    args["maxTimeSeriesLen"] = 1200
    args["batchSize"] = 128

    # AdamW tuned
    args["lrStart"] = exp["lrStart"]
    args["lrEnd"] = exp["lrEnd"]
    args["l2_decay"] = exp["l2_decay"]      # for AdamW: true decoupled weight decay
    args["adam_eps"] = exp["adam_eps"]      # you must read this in trainer

    # model
    args["nUnits"] = 256
    args["nBatch"] = N_BATCH
    args["nLayers"] = 5
    args["seed"] = 0
    args["nClasses"] = 40
    args["nInputFeatures"] = 256
    args["dropout"] = 0.2

    # augmentations (keep fixed while comparing optimizers)
    args["whiteNoiseSD"] = 0.8
    args["constantOffsetSD"] = 0.2
    args["gaussianSmoothWidth"] = 2.0

    # streaming params
    args["strideLen"] = 4
    args["kernelLen"] = 32
    args["bidirectional"] = False

    # NEW: tells trainer to use AdamW (you must implement this switch)
    args["optimizer"] = "adamw"

    print("\n========================================")
    print("Training:", modelName)
    print(" lrStart:", args["lrStart"], "lrEnd:", args["lrEnd"])
    print(" weight_decay:", args["l2_decay"], "eps:", args["adam_eps"])
    print("========================================\n")

    trainModel(args)
