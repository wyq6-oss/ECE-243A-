# scripts/run_arch_sweep.py
import os, sys, json, csv, time, pickle
from datetime import datetime
import numpy as np

# ---- adjust these two lines if needed ----
sys.path.append("/home/jupyter/neural_seq_decoder-master/src")
from neural_decoder.neural_decoder_trainer import trainModel
# -----------------------------------------

BASE_OUTPUT_ROOT = "/home/jupyter/baseline_logs"
DATASET_PATH = "/home/jupyter/neural_seq_decoder-master/competitionData/ptDecoder_ctc"

def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def read_training_stats(outdir):
    """Return dict with arrays or None if missing."""
    path = os.path.join(outdir, "trainingStats")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        stats = pickle.load(f)
    # expected keys: testLoss, testCER
    return stats

def best_metrics_from_stats(stats, eval_every=100, last_k=10):
    cer = np.asarray(stats["testCER"], dtype=float)
    loss = np.asarray(stats["testLoss"], dtype=float)
    # eval steps correspond to batches: 0, 100, 200, ...
    batches = np.arange(len(cer)) * eval_every

    best_idx = int(np.nanargmin(cer))
    best = {
        "best_cer": float(cer[best_idx]),
        "best_batch": int(batches[best_idx]),
        "best_loss": float(loss[best_idx]),
        "final_cer": float(cer[-1]),
        "final_loss": float(loss[-1]),
    }

    k = min(last_k, len(cer))
    tail = cer[-k:]
    best["last_k"] = int(k)
    best["last_k_cer_mean"] = float(np.nanmean(tail))
    best["last_k_cer_std"] = float(np.nanstd(tail, ddof=1)) if k >= 2 else 0.0
    return best

def make_base_args():
    # your baseline defaults
    args = {}
    args["datasetPath"] = DATASET_PATH

    args["seqLen"] = 150
    args["maxTimeSeriesLen"] = 1200
    args["batchSize"] = 128

    args["lrStart"] = 0.05
    args["lrEnd"] = 0.02

    args["nUnits"] = 256
    args["nBatch"] = 3000
    args["nLayers"] = 5
    args["seed"] = 0
    args["nClasses"] = 40
    args["nInputFeatures"] = 256

    args["dropout"] = 0.2
    args["whiteNoiseSD"] = 0.8
    args["constantOffsetSD"] = 0.2
    args["gaussianSmoothWidth"] = 2.0

    args["strideLen"] = 4
    args["kernelLen"] = 32
    args["bidirectional"] = False

    args["l2_decay"] = 1e-5

    # NEW architecture flags (only work if your model.py supports them)
    args["useLayerNorm"] = False
    args["usePreLayerNorm"] = False
    args["usePostNet"] = False
    args["postNetLayers"] = 2
    args["postNetDim"] = None
    args["postNetDropout"] = None
    args["useDayLayer"] = True
    args["usePerDayInputLayer"] = False

    return args

def with_overrides(base, overrides):
    out = dict(base)
    out.update(overrides)
    return out

def main():
    run_id = now_tag()
    sweep_root = os.path.join(BASE_OUTPUT_ROOT, f"sweep_{run_id}")
    os.makedirs(sweep_root, exist_ok=True)

    summary_csv = os.path.join(sweep_root, "summary.csv")
    summary_json = os.path.join(sweep_root, "summary.json")

    base = make_base_args()

    # ---- Experiment definitions ----
    # Keep it “small but useful”. Add/remove as you like.
    experiments = [
        # Controls

        # LayerNorm after GRU output (most stable)
        ("ln_post", {"useLayerNorm": True}),

        # PreLN + PostLN (if implemented in model.py)
        ("ln_pre_post", {"useLayerNorm": True, "usePreLayerNorm": True}),

        # PostNet variants (Linderman-style)
        ("ln_postnet_2x256_d0.3",
         {"useLayerNorm": True, "usePostNet": True,
          "postNetLayers": 2, "postNetDim": 256, "postNetDropout": 0.3, "dropout": 0.3}),

        ("ln_postnet_3x256_d0.2",
         {"useLayerNorm": True, "usePostNet": True,
          "postNetLayers": 3, "postNetDim": 256, "postNetDropout": 0.2, "dropout": 0.3}),

        ("ln_postnet_2x384_d0.3",
         {"useLayerNorm": True, "usePostNet": True,
          "postNetLayers": 2, "postNetDim": 384, "postNetDropout": 0.3, "dropout": 0.3}),

        # Ablation: remove day layer (only meaningful if LN is on)
        ("ln_postnet_2x256_d0.3_dayOFF",
         {"useLayerNorm": True, "usePreLayerNorm": True, "usePostNet": True,
          "postNetLayers": 2, "postNetDim": 256, "postNetDropout": 0.3,
          "useDayLayer": False, "dropout": 0.3}),
    ]

    results = []
    header = [
        "label", "outputDir",
        "best_cer", "best_batch", "best_loss",
        "final_cer", "final_loss",
        "last_k", "last_k_cer_mean", "last_k_cer_std"
    ]

    # write CSV header early
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

    for label, overrides in experiments:
        model_name = f"{label}"
        outdir = os.path.join(sweep_root, model_name)

        args = with_overrides(base, overrides)
        args["outputDir"] = outdir

        print("\n" + "=" * 70)
        print(f"RUN: {label}")
        print(f"outdir: {outdir}")
        print("overrides:", overrides)
        print("=" * 70)

        # if already ran (has trainingStats), skip
        if os.path.exists(os.path.join(outdir, "trainingStats")):
            print("-> Found existing trainingStats. Skipping training.")
        else:
            os.makedirs(outdir, exist_ok=True)
            trainModel(args)

        stats = read_training_stats(outdir)
        if stats is None:
            print("!! No trainingStats found. Recording as failed.")
            row = {k: None for k in header}
            row["label"] = label
            row["outputDir"] = outdir
        else:
            m = best_metrics_from_stats(stats, eval_every=100, last_k=10)
            row = {"label": label, "outputDir": outdir, **m}

        results.append(row)

        # append row to CSV after each run
        with open(summary_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow(row)

        # also write JSON after each run (so you never lose progress)
        with open(summary_json, "w") as f:
            json.dump(results, f, indent=2)

    print("\nDONE.")
    print("Summary CSV :", summary_csv)
    print("Summary JSON:", summary_json)

if __name__ == "__main__":
    main()
