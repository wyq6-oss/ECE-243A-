import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .dataset import SpeechDataset


import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .dataset import SpeechDataset

# -------- NEW: import your transforms --------
from .transforms import SignedLog1p, Log1pNonneg, ZScore, Compose
# -------------------------------------------

def _apply_log_transform(x: torch.Tensor, log_transform: str):
    """
    x: [T, F] float tensor
    log_transform:
      - "none"
      - "signed_log1p"
      - "log1p_nonneg"
    """
    if log_transform is None or log_transform == "none":
        return x
    if log_transform == "signed_log1p":
        return torch.sign(x) * torch.log1p(torch.abs(x))
    if log_transform == "log1p_nonneg":
        return torch.log1p(torch.clamp(x, min=0.0))
    raise ValueError(f"Unknown log_transform: {log_transform}")

def _compute_feature_stats(train_days, log_transform="none", eps=1e-5):
    """
    train_days: loadedData["train"]  (likely list over days/sessions)
      each element d has d["sentenceDat"] which is a list of [T, F] arrays (ragged T)
    Returns:
      mean: [F] torch.float32
      std:  [F] torch.float32
    """
    feat_sum = None
    feat_sumsq = None
    feat_count = 0  # total number of time steps accumulated

    for d in train_days:
        trials = d["sentenceDat"]

        # trials can be list of np arrays, or sometimes already an ndarray
        if isinstance(trials, np.ndarray):
            # if it's a single [T,F] array, wrap it; if it's [N,T,F], iterate N
            if trials.ndim == 2:
                trials_iter = [trials]
            elif trials.ndim == 3:
                trials_iter = [trials[i] for i in range(trials.shape[0])]
            else:
                raise ValueError(f"Unexpected sentenceDat ndim: {trials.ndim}")
        else:
            trials_iter = trials  # assume list-like

        for arr in trials_iter:
            x = torch.as_tensor(arr, dtype=torch.float32)

            # ensure shape is [T, F] (some datasets store [F, T])
            if x.ndim != 2:
                raise ValueError(f"Trial has ndim={x.ndim}, expected 2D [T,F]")

            # If you know nInputFeatures=256, you can enforce:
            # if x.shape[1] != 256 and x.shape[0] == 256: x = x.T
            if x.shape[0] < x.shape[1] and x.shape[1] > 256 and x.shape[0] == 256:
                # optional heuristic; remove if not needed
                x = x.T

            x = _apply_log_transform(x, log_transform=log_transform)

            # init accumulators once we know F
            if feat_sum is None:
                F = x.shape[1]
                feat_sum = torch.zeros(F, dtype=torch.float64)
                feat_sumsq = torch.zeros(F, dtype=torch.float64)

            feat_sum += x.double().sum(dim=0)
            feat_sumsq += (x.double() ** 2).sum(dim=0)
            feat_count += x.shape[0]

    if feat_count == 0:
        raise ValueError("No frames found while computing feature stats.")

    mean = (feat_sum / feat_count).float()
    var = (feat_sumsq / feat_count - mean.double() ** 2).clamp(min=0.0).float()
    std = torch.sqrt(var + eps)

    return mean, std


def getDatasetLoaders(datasetName, batchSize, args=None):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    # -------- NEW: build transform pipeline --------
    args = args or {}
    log_t = args.get("logTransform", "none")  # "none" | "signed_log1p" | "log1p_nonneg"

    mean, std = _compute_feature_stats(loadedData["train"], log_transform=log_t)
    zscore = ZScore(mean, std)

    if log_t == "signed_log1p":
        transform = Compose([SignedLog1p(), zscore])
    elif log_t == "log1p_nonneg":
        transform = Compose([Log1pNonneg(), zscore])
    else:
        transform = zscore
    # ------------------------------------------------

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)
        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    # IMPORTANT: apply SAME transform to train & test; stats from train only
    train_ds = SpeechDataset(loadedData["train"], transform=transform)
    test_ds  = SpeechDataset(loadedData["test"],  transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData, mean, std, log_t

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData, feat_mean, feat_std, log_t = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
        args=args,
    )

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
        useLayerNorm=args.get("useLayerNorm", False),
        usePreLayerNorm=args.get("usePreLayerNorm", False),
        usePostNet=args.get("usePostNet", False),
        postNetLayers=args.get("postNetLayers", 2),
        postNetDim=args.get("postNetDim", None),
        postNetDropout=args.get("postNetDropout", None),
        useDayLayer=args.get("useDayLayer", True),
        usePerDayInputLayer=args.get("usePerDayInputLayer", False),
    ).to(device)


    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=args["lrStart"],
    #     betas=(0.9, 0.999),
    #     eps=0.1,
    #     weight_decay=args["l2_decay"],
    # )
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args["lrStart"],
    betas=(0.9, 0.999),
    weight_decay=args['l2_decay'],  # now this is proper decoupled weight decay
    eps=1e-8,  # closer to PyTorch default
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # Compute prediction error
        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if batch >= 3000 and cer > 0.32:
                print(f"[EARLY STOP] batch={batch} cer={cer:.4f} > 0.32")
                testLoss.append(avgDayLoss)
                testCER.append(cer)
                tStats = {"testLoss": np.array(testLoss), "testCER": np.array(testCER)}
                with open(args["outputDir"] + "/trainingStats", "wb") as file:
                    pickle.dump(tStats, file)
                break

            # --- save best weights ---
            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")

            testLoss.append(avgDayLoss)
            testCER.append(cer)
            tStats = {"testLoss": np.array(testLoss), "testCER": np.array(testCER)}
            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)

            # --- EARLY STOP (CER plateau / converge stop) --


            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)
            with open(os.path.join(args["outputDir"], "normalization_stats.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "logTransform": log_t,
                        "mean": feat_mean.cpu().numpy(),
                        "std": feat_std.cpu().numpy(),
                    },
                    f,
                )



def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()
