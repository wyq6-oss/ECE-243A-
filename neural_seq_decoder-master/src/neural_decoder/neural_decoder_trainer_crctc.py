# ===================== COPY/PASTE: MODIFIED VERSION WITH DEBUG PRINTS =====================
import os
import pickle
import time

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from edit_distance import SequenceMatcher
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .dataset import SpeechDataset
from .model import GRUDecoder


def getDatasetLoaders(datasetName, batchSize):
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


class CRCTCLoss(nn.Module):
    """
    L = 0.5*(CTC(a)+CTC(b)) + alpha * 0.5*( KL(sg(b)||a) + KL(sg(a)||b) )
    pred_a/pred_b are logits shaped (B, T', C)
    """
    def __init__(self, blank=0, alpha=0.5, reduction="mean", eps=1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.ctc = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=True)

    def _masked_frame_kl(self, logp_pred_BTC, p_tgt_det_BTC, input_lengths):
        logp_tgt = torch.log(p_tgt_det_BTC.clamp_min(self.eps))
        kl_BT = (p_tgt_det_BTC * (logp_tgt - logp_pred_BTC)).sum(dim=-1)  # (B,T)

        B, T = kl_BT.shape
        t = torch.arange(T, device=kl_BT.device).view(1, T)
        mask = (t < input_lengths.view(B, 1)).float()

        denom = mask.sum().clamp_min(1.0)
        return (kl_BT * mask).sum() / denom

    def forward(self, pred_a_BTC, pred_b_BTC, targets_1d, input_lengths, target_lengths):
        if pred_a_BTC.shape != pred_b_BTC.shape:
            raise ValueError(f"CR term requires same shape, got {pred_a_BTC.shape} vs {pred_b_BTC.shape}")

        logp_a_BTC = F.log_softmax(pred_a_BTC, dim=-1)
        logp_b_BTC = F.log_softmax(pred_b_BTC, dim=-1)

        # CTC expects (T,B,C)
        logp_a_TBC = logp_a_BTC.permute(1, 0, 2)
        logp_b_TBC = logp_b_BTC.permute(1, 0, 2)

        ctc_a = self.ctc(logp_a_TBC, targets_1d, input_lengths, target_lengths)
        ctc_b = self.ctc(logp_b_TBC, targets_1d, input_lengths, target_lengths)
        ctc = 0.5 * (ctc_a + ctc_b)

        # stop-gradient targets
        p_a_det = logp_a_BTC.detach().exp()
        p_b_det = logp_b_BTC.detach().exp()

        kl_b_to_a = self._masked_frame_kl(logp_pred_BTC=logp_a_BTC, p_tgt_det_BTC=p_b_det, input_lengths=input_lengths)
        kl_a_to_b = self._masked_frame_kl(logp_pred_BTC=logp_b_BTC, p_tgt_det_BTC=p_a_det, input_lengths=input_lengths)
        cr = 0.5 * (kl_b_to_a + kl_a_to_b)

        return ctc + self.alpha * cr


def two_view_augment(X, X_len, args, device):
    """
    X: (B, T, D). Two aligned views with independent time/freq masking + noise.
    """
    X_a = X.clone()
    X_b = X.clone()

    def apply_noise(Z):
        if args.get("whiteNoiseSD", 0) > 0:
            Z = Z + torch.randn_like(Z) * args["whiteNoiseSD"]
        if args.get("constantOffsetSD", 0) > 0:
            Z = Z + (torch.randn([Z.shape[0], 1, Z.shape[2]], device=device) * args["constantOffsetSD"])
        return Z

    def apply_masks(Z):
        B, T, D = Z.shape
        n_time = int(args.get("cr_time_masks", 2))
        max_time = float(args.get("cr_time_max_ratio", 0.20))
        n_feat = int(args.get("cr_feat_masks", 1))
        max_feat = float(args.get("cr_feat_max_ratio", 0.10))

        for b in range(B):
            L = int(X_len[b].item())
            if L <= 1:
                continue

            for _ in range(n_time):
                if max_time <= 0:
                    break
                maxw = max(2, int(L * max_time))
                w = int(torch.randint(1, maxw + 1, (1,), device=device).item())
                s = int(torch.randint(0, max(1, L - w + 1), (1,), device=device).item())
                Z[b, s:s+w, :] = 0

            for _ in range(n_feat):
                if max_feat <= 0:
                    break
                maxw = max(2, int(D * max_feat))
                w = int(torch.randint(1, maxw + 1, (1,), device=device).item())
                s = int(torch.randint(0, max(1, D - w + 1), (1,), device=device).item())
                Z[b, :L, s:s+w] = 0

        return Z

    X_a = apply_masks(X_a)
    X_b = apply_masks(X_b)
    X_a = apply_noise(X_a)
    X_b = apply_noise(X_b)

    return X_a, X_b


def _targets_to_1d(y_2d, y_len):
    # Convert padded (B,S) targets into 1D concatenated for CTCLoss (most robust across PyTorch versions)
    return torch.cat([y_2d[i, :y_len[i]] for i in range(y_2d.size(0))]).to(torch.long)


def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(args["datasetPath"], args["batchSize"])

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

    loss_fn = CRCTCLoss(blank=0, alpha=args.get("cr_alpha", 0.5), reduction="mean").to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )

    train_iter = iter(trainLoader)

    testLoss = []
    testCER = []
    startTime = time.time()

    for batch in range(args["nBatch"]):
        model.train()

        try:
            X, y, X_len, y_len, dayIdx = next(train_iter)
        except StopIteration:
            train_iter = iter(trainLoader)
            X, y, X_len, y_len, dayIdx = next(train_iter)

        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # enforce robust dtypes
        y = y.to(torch.long)
        X_len = X_len.to(torch.int32)
        y_len = y_len.to(torch.int32)

        # convert targets to 1D for CTCLoss
        y_1d = _targets_to_1d(y, y_len)

        # two augmented views
        X_a, X_b = two_view_augment(X, X_len, args, device)

        pred_a = model.forward(X_a, dayIdx)  # (B,T',C)
        pred_b = model.forward(X_b, dayIdx)  # (B,T',C)

        # correct output lengths: floor((L-K)/S)+1
        in_lens = ((X_len - model.kernelLen) // model.strideLen) + 1
        in_lens = torch.clamp(in_lens, min=1).to(torch.int32)

        loss = loss_fn(pred_a, pred_b, y_1d, in_lens, y_len)
        loss = torch.sum(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0

                loss_ctc_eval = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    y = y.to(torch.long)
                    X_len = X_len.to(torch.int32)
                    y_len = y_len.to(torch.int32)
                    y_1d = _targets_to_1d(y, y_len)

                    pred = model.forward(X, testDayIdx)

                    adjustedLens = ((X_len - model.kernelLen) // model.strideLen) + 1
                    adjustedLens = torch.clamp(adjustedLens, min=1).to(torch.int32)

                    loss_eval = loss_ctc_eval(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y_1d,
                        adjustedLens,
                        y_len,
                    )
                    loss_eval = torch.sum(loss_eval)
                    allLoss.append(loss_eval.cpu().detach().numpy())

                    # ---- DEBUG PRINT: blank collapse + label range ----
                    decoded = pred.argmax(dim=-1)  # (B,T')
                    blank_rate = (decoded == 0).float().mean().item()
                    avg_nonblank_frames = (decoded != 0).float().sum(dim=1).mean().item()
                    print("DEBUG blank_rate:", blank_rate,
                          "avg_nonblank_frames:", avg_nonblank_frames,
                          "y_min/max:", int(y.min().item()), int(y.max().item()))

                    for iterIdx in range(pred.shape[0]):
                        seq = pred[iterIdx, :adjustedLens[iterIdx], :].detach()
                        decodedSeq = torch.argmax(seq, dim=-1)
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(y[iterIdx, :y_len[iterIdx]].cpu().detach())

                        matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / max(1, total_seq_length)

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")

            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {"testLoss": np.array(testLoss), "testCER": np.array(testCER)}
            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


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
# ===================== END COPY/PASTE =====================
