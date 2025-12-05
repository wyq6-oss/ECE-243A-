import os, copy, pickle
from datetime import datetime
import gc, torch

import sys
sys.path.append('/home/jupyter/neural_seq_decoder-master/src')

from neural_decoder.neural_decoder_trainer_experiments import trainModel


def run_experiments(base_args: dict, exp_list: list, root=None):
    root = root or os.path.dirname(base_args["outputDir"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for exp in exp_list:
        a = copy.deepcopy(base_args)
        run_name = exp["name"]
        a["outputDir"] = os.path.join(root, f"{run_name}_{stamp}")
        os.makedirs(a["outputDir"], exist_ok=True)

        for k, v in exp.get("overrides", {}).items():
            a[k] = v

        print("\n" + "=" * 60)
        print("RUN:", run_name)
        print("outputDir:", a["outputDir"])
        print("overrides:", exp.get("overrides", {}))
        print("=" * 60)

        gc.collect()
        torch.cuda.empty_cache()

        trainModel(a)
        results.append((run_name, a["outputDir"]))

    return results


def read_best_cer(run_dir: str):
    stats_path = os.path.join(run_dir, "trainingStats")
    if not os.path.exists(stats_path):
        return None
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    cer = stats.get("testCER", None)
    if cer is None or len(cer) == 0:
        return None
    cer = list(cer)
    best = float(min(cer))
    best_step = int(cer.index(min(cer)) * 100)  # because you eval every 100 batches
    return best, best_step


def leaderboard(results):
    rows = []
    for name, outdir in results:
        info = read_best_cer(outdir)
        if info is None:
            rows.append((name, None, None, outdir))
        else:
            best, step = info
            rows.append((name, best, step, outdir))

    rows.sort(key=lambda x: (1e9 if x[1] is None else x[1]))
    print("\n" + "#" * 70)
    print("Leaderboard (lower CER is better)")
    print("#" * 70)
    for name, best, step, outdir in rows:
        if best is None:
            print(f"{name:28s}  bestCER: N/A    best@batch: N/A     dir: {outdir}")
        else:
            print(f"{name:28s}  bestCER: {best:.4f}  best@batch: {step:5d}  dir: {outdir}")
    return rows


# -------------------------
# Your base args
# -------------------------
modelName = 'speechBaseline4_sample'
args = {}
args['outputDir'] = '/home/jupyter/baseline_logs/' + modelName
args['datasetPath'] = '/home/jupyter/neural_seq_decoder-master/competitionData/ptDecoder_ctc'
args['batchSize'] = 128
args['lrStart'] = 0.05
args['lrEnd'] = 0.02
args['nUnits'] = 256
args['nBatch'] = 10000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.2
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5

# enable improvements
args["speckle_p"] = 0.3
args["blank_reg_weight"] = 0.01
args["postnet_layers"] = 2
args["postnet_dropout"] = 0.2

args["lrStart"] = 0.02
args["lrEnd"] = 0.001

args["whiteNoiseSD"] = 0.3
args["constantOffsetSD"] = 0.1

args["speckle_p"] = 0.2         # start smaller to avoid OOM + instability
args["blank_reg_weight"] = 0.02 # helps escape all-blank collapse

# -------------------------
# Multi-model experiment list
# (small, high-ROI set)
# -------------------------
exp_list = [
    # baseline-ish reference
    {"name": "M0_base_like", "overrides": {"speckle_p": 0.0, "blank_reg_weight": 0.0, "postnet_layers": 0}},

    # your current
    {"name": "M1_yours", "overrides": {}},

    # LR schedule improvements (stronger decay)
    {"name": "M2_LR_0p02_to_0p001", "overrides": {"lrStart": 0.02, "lrEnd": 0.001}},
    {"name": "M3_LR_0p01_to_0p0005", "overrides": {"lrStart": 0.01, "lrEnd": 0.0005}},

    # blank/speckle sweeps
    {"name": "M4_blank_0p005", "overrides": {"blank_reg_weight": 0.005}},
    {"name": "M5_blank_0p02",  "overrides": {"blank_reg_weight": 0.02}},
    {"name": "M6_speckle_0p2", "overrides": {"speckle_p": 0.2}},
    {"name": "M7_speckle_0p4", "overrides": {"speckle_p": 0.4}},

    # window / stride
    {"name": "M8_win24", "overrides": {"kernelLen": 24}},
    {"name": "M9_stride2", "overrides": {"strideLen": 2}},  # more compute, often helps CTC

    # capacity tweak
    {"name": "M10_units384_layers3", "overrides": {"nUnits": 384, "nLayers": 3}},
]

# Run them + show leaderboard
results = run_experiments(args, exp_list, root="/home/jupyter/baseline_logs/")
rows = leaderboard(results)
