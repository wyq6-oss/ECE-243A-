import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .dataset import SpeechDataset


import torch
import torch.nn as nn

class LabelSmoothingCTCLoss(nn.Module):
    """
    CTC Loss + entropy regularization (often called "label smoothing" in practice for CTC).

    Key property:
      smoothing == 0  => EXACTLY matches nn.CTCLoss(..., reduction=reduction)
    """
    def __init__(self, blank=0, smoothing=0.1, reduction="mean", zero_infinity=True):
        super().__init__()
        self.blank = blank
        self.smoothing = float(smoothing)
        self.reduction = reduction
        self.zero_infinity = zero_infinity

        # Use SAME reduction as baseline to match exactly when smoothing==0
        self.ctc_loss = nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        log_probs: (T, N, C)  (log-softmaxed)
        targets: (N, S)
        input_lengths: (N,)
        target_lengths: (N,)
        """
        base = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        # Exactly the baseline CTC objective
        if self.smoothing <= 0:
            return base

        # Entropy regularizer term (scalar)
        probs = log_probs.exp()                               # [T, N, C]
        entropy = -(probs * log_probs).sum(dim=-1).mean()     # scalar

        # Combine: keep most of CTC, add entropy encouragement
        return (1.0 - self.smoothing) * base - self.smoothing * entropy



def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

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

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

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

    return train_loader, test_loader, loadedData


def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
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
    ).to(device)

    # ---- use label-smoothed CTC instead of plain CTCLoss ----
    smoothing = args.get("ctc_smoothing", 0.1)
    loss_ctc = LabelSmoothingCTCLoss(
        blank=0,
        smoothing=smoothing,
        reduction="mean",
        zero_infinity=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[3000, 5000, 7000],
        gamma=0.3
    )

    # --- EARLY STOP (CER plateau) state + hyperparams ---
    eval_every = args.get("evalEvery", 100)  # your code effectively uses 100
    patience_evals = args.get("earlyStopPatienceEvals", 20)  # 20 evals * 100 batches = 2000 batches
    min_delta = args.get("earlyStopMinDelta", 1e-3)          # require at least this improvement
    warmup_evals = args.get("earlyStopWarmupEvals", 10)      # don’t stop too early
    plateau_start_batch = args.get("earlyStopStartBatch", 0) # optionally delay plateau logic

    best_cer = float("inf")
    best_eval_idx = -1
    eval_idx = 0
    # ---------------------------------------------------

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

        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if batch % eval_every == 0:
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
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0:adjustedLens[iterIdx], :]),
                            dim=-1,
                        )
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(y[iterIdx][0:y_len[iterIdx]].cpu().detach())

                        matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/eval_every:>7.3f}"
                )
                startTime = time.time()

            # --- your existing "bad model" early stop ---
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

            # --- EARLY STOP (CER plateau / converge stop) ---
            eval_idx += 1

            if batch >= plateau_start_batch:
                if cer < best_cer - min_delta:
                    best_cer = cer
                    best_eval_idx = eval_idx

                if eval_idx >= warmup_evals and (eval_idx - best_eval_idx) >= patience_evals:
                    print(
                        f"[EARLY STOP - PLATEAU] batch={batch} cer={cer:.6f} "
                        f"best_cer={best_cer:.6f} (no improvement >= {min_delta} "
                        f"for {patience_evals} evals)"
                    )
                    break
            # --------------------------------------------



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
