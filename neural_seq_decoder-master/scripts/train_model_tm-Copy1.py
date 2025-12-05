# train.py
import os
import csv
import time
import pickle
from copy import deepcopy
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------
# Make sure your repo src is importable
import sys
sys.path.append("/home/jupyter/neural_seq_decoder-master/src")

# Pick the trainer you want to use (time/feature mask version)
from neural_decoder.neural_decoder_trainer_tm import trainModel
# ---------------------------------------------------------------------


BASE_OUTPUT_ROOT = "/home/jupyter/baseline_logs/"
DATASET_PATH = "/home/jupyter/neural_seq_decoder-master/competitionData/ptDecoder_ctc"

RESULTS_CSV = os.path.join(BASE_OUTPUT_ROOT, "results_adamw_sweep.csv")


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def read_best_cer(output_dir: str, eval_every: int = 100):
    """
    Reads output_dir/trainingStats created by your trainer.
    Returns (best_cer, best_batch, last_cer, n_evals) or (None, None, None, 0) if missing.
    Assumes trainer evaluates every `eval_every` batches (your code uses batch % 100 == 0).
    """
    stats_path = os.path.join(output_dir, "trainingStats")
    if not os.path.exists(stats_path):
        return None, None, None, 0

    try:
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        cer_list = np.array(stats.get("testCER", []), dtype=float)
        if cer_list.size == 0:
            return None, None, None, 0
        best_idx = int(np.argmin(cer_list))
        best_cer = float(cer_list[best_idx])
        best_batch = best_idx * eval_every  # IMPORTANT: assumes eval schedule
        last_cer = float(cer_list[-1])
        return best_cer, best_batch, last_cer, int(cer_list.size)
    except Exception:
        return None, None, None, 0


def append_row(csv_path: str, fieldnames, row: dict):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def main():
    safe_makedirs(BASE_OUTPUT_ROOT)

    # -----------------------------
    # 1) Your augmentation grid
    # -----------------------------
    experiments = [
        # Example time-only mask as you showed
        {
            "label": "wn1.0_tm10x1_fm0x0",
            "whiteNoiseSD": 1.0,
            "constantOffsetSD": 0.2,
            "timeMaskWidth": 10,
            "nTimeMasks": 1,
            "featMaskWidth": 0,
            "nFeatMasks": 0,
        },
        # Add more experiments here (time-only, feat-only, time+feat)
    ]

    # -----------------------------
    # 2) AdamW tuning grid
    # -----------------------------
    adamw_sweeps = [
        {"optLabel": "aw_lr1e-3_wd1e-3_eps1e-8_b099",  "lrStart": 1e-3, "lrEnd": 1e-4, "l2_decay": 1e-3, "adamw_eps": 1e-8, "adamw_betas": (0.9, 0.999)},
        {"optLabel": "aw_lr8e-4_wd1e-3_eps1e-8_b099",  "lrStart": 8e-4, "lrEnd": 8e-5, "l2_decay": 1e-3, "adamw_eps": 1e-8, "adamw_betas": (0.9, 0.999)},
        {"optLabel": "aw_lr6e-4_wd3e-3_eps1e-8_b099",  "lrStart": 6e-4, "lrEnd": 6e-5, "l2_decay": 3e-3, "adamw_eps": 1e-8, "adamw_betas": (0.9, 0.999)},
        {"optLabel": "aw_lr1e-3_wd1e-2_eps1e-8_b099",  "lrStart": 1e-3, "lrEnd": 1e-4, "l2_decay": 1e-2, "adamw_eps": 1e-8, "adamw_betas": (0.9, 0.999)},
        {"optLabel": "aw_lr7e-4_wd1e-2_eps1e-8_b099",  "lrStart": 7e-4, "lrEnd": 7e-5, "l2_decay": 1e-2, "adamw_eps": 1e-8, "adamw_betas": (0.9, 0.999)},
        {"optLabel": "aw_lr8e-4_wd1e-3_eps1e-6_b099",  "lrStart": 8e-4, "lrEnd": 8e-5, "l2_decay": 1e-3, "adamw_eps": 1e-6, "adamw_betas": (0.9, 0.999)},
        {"optLabel": "aw_lr8e-4_wd1e-3_eps1e-8_b098",  "lrStart": 8e-4, "lrEnd": 8e-5, "l2_decay": 1e-3, "adamw_eps": 1e-8, "adamw_betas": (0.9, 0.98)},
        {"optLabel": "aw_lr6e-4_wd3e-3_eps1e-8_b0995", "lrStart": 6e-4, "lrEnd": 6e-5, "l2_decay": 3e-3, "adamw_eps": 1e-8, "adamw_betas": (0.9, 0.995)},
    ]

    # -----------------------------
    # 3) Shared base args
    # -----------------------------
    N_BATCH = 10000
    base_args = {
        "datasetPath": DATASET_PATH,
        "seqLen": 150,
        "maxTimeSeriesLen": 1200,
        "batchSize": 128,

        "nUnits": 256,
        "nLayers": 5,
        "seed": 0,
        "nClasses": 40,
        "nInputFeatures": 256,
        "dropout": 0.2,

        "gaussianSmoothWidth": 2.0,

        "strideLen": 4,
        "kernelLen": 32,
        "bidirectional": False,

        "nBatch": N_BATCH,

        # trainer will read these for noise
        "whiteNoiseSD": 0.0,
        "constantOffsetSD": 0.0,

        # trainer_tm will read these for masks
        "timeMaskWidth": 0,
        "nTimeMasks": 0,
        "featMaskWidth": 0,
        "nFeatMasks": 0,

        # optional: if your trainer supports these
        # "clipGradNorm": 1.0,
        # "ctc_smoothing": 0.0,
    }

    # -----------------------------
    # 4) CSV logging
    # -----------------------------
    fieldnames = [
        "timestamp",
        "modelName",
        "outputDir",

        "whiteNoiseSD",
        "constantOffsetSD",
        "timeMaskWidth",
        "nTimeMasks",
        "featMaskWidth",
        "nFeatMasks",

        "lrStart",
        "lrEnd",
        "weight_decay",
        "adamw_eps",
        "adamw_beta1",
        "adamw_beta2",

        "nBatch",
        "bestCER",
        "bestBatchApprox",
        "lastCER",
        "nEvals",
        "status",
        "notes",
    ]

    total_runs = len(experiments) * len(adamw_sweeps)
    run_idx = 0

    for exp in experiments:
        for opt in adamw_sweeps:
            run_idx += 1

            args = deepcopy(base_args)

            # apply augmentation settings
            args["whiteNoiseSD"] = exp["whiteNoiseSD"]
            args["constantOffsetSD"] = exp["constantOffsetSD"]
            args["timeMaskWidth"] = exp["timeMaskWidth"]
            args["nTimeMasks"] = exp["nTimeMasks"]
            args["featMaskWidth"] = exp["featMaskWidth"]
            args["nFeatMasks"] = exp["nFeatMasks"]

            # apply AdamW settings
            args["lrStart"] = opt["lrStart"]
            args["lrEnd"] = opt["lrEnd"]
            args["l2_decay"] = opt["l2_decay"]
            args["adamw_eps"] = opt["adamw_eps"]
            args["adamw_betas"] = opt["adamw_betas"]

            modelName = f"speechBaseline4_{exp['label']}_{opt['optLabel']}"
            outputDir = os.path.join(BASE_OUTPUT_ROOT, modelName)
            args["outputDir"] = outputDir

            print("\n========================================")
            print(f"[{run_idx}/{total_runs}] Training model: {modelName}")
            print(f"  wn/co            = {args['whiteNoiseSD']} / {args['constantOffsetSD']}")
            print(f"  time mask        = {args['timeMaskWidth']} x {args['nTimeMasks']}")
            print(f"  feat mask        = {args['featMaskWidth']} x {args['nFeatMasks']}")
            print(f"  AdamW lr         = {args['lrStart']} -> {args['lrEnd']}")
            print(f"  AdamW wd         = {args['l2_decay']}")
            print(f"  AdamW eps        = {args['adamw_eps']}")
            print(f"  AdamW betas      = {args['adamw_betas']}")
            print(f"  nBatch           = {args['nBatch']}")
            print("========================================\n")

            status = "ok"
            notes = ""

            t0 = time.time()
            try:
                trainModel(args)
            except Exception as e:
                status = "error"
                notes = repr(e)
                print(f"[ERROR] {modelName}: {notes}")

            # summarize
            bestCER, bestBatch, lastCER, nEvals = read_best_cer(outputDir, eval_every=100)
            dt = time.time() - t0

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "modelName": modelName,
                "outputDir": outputDir,

                "whiteNoiseSD": args["whiteNoiseSD"],
                "constantOffsetSD": args["constantOffsetSD"],
                "timeMaskWidth": args["timeMaskWidth"],
                "nTimeMasks": args["nTimeMasks"],
                "featMaskWidth": args["featMaskWidth"],
                "nFeatMasks": args["nFeatMasks"],

                "lrStart": args["lrStart"],
                "lrEnd": args["lrEnd"],
                "weight_decay": args["l2_decay"],
                "adamw_eps": args["adamw_eps"],
                "adamw_beta1": args["adamw_betas"][0],
                "adamw_beta2": args["adamw_betas"][1],

                "nBatch": args["nBatch"],
                "bestCER": bestCER,
                "bestBatchApprox": bestBatch,
                "lastCER": lastCER,
                "nEvals": nEvals,
                "status": status,
                "notes": (notes + f" | walltime_sec={dt:.1f}").strip(),
            }

            append_row(RESULTS_CSV, fieldnames, row)

    print("\nDONE.")
    print(f"Results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
