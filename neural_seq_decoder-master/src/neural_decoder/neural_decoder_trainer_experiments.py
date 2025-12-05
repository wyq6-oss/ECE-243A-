# neural_decoder_trainer_experiments.py
import os
import time
import copy
import pickle
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from edit_distance import SequenceMatcher

from .dataset import SpeechDataset
from .augmentations import GaussianSmoothing


# -------------------------
# Utilities
# -------------------------
def ctc_input_lengths(X_len: torch.Tensor, kernelLen: int, strideLen: int) -> torch.Tensor:
    """
    Output length after Unfold(kernelLen, strideLen) with valid windowing:
      L_out = floor((T - kernelLen)/strideLen) + 1
    """
    out = torch.div((X_len - kernelLen), strideLen, rounding_mode="floor") + 1
    return torch.clamp(out, min=1).to(torch.int32)


def speckle_mask(X: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0 or (not X.is_cuda and not X.requires_grad):
        return X
    if not X.is_cuda:
        # still fine on CPU
        keep = 1.0 - p
        mask = (torch.rand(X.shape, device=X.device) < keep)
        return X * mask.to(X.dtype) / keep

    keep = 1.0 - p
    # boolean mask is much smaller than float rand_like + float mask
    mask = (torch.rand(X.shape, device=X.device, dtype=torch.float16) < keep)
    return X * mask.to(X.dtype) / keep



# -------------------------
# Model: Baseline GRU + Day Adapt + optional PostNet
# -------------------------
class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0.0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0.0,
        bidirectional=False,
        # New (optional, backward compatible)
        postnet_layers=0,
        postnet_dropout=0.2,
        postnet_activation="gelu",
    ):
        super().__init__()

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional

        self.inputLayerNonlinearity = torch.nn.Softsign()

        # (B,T,D) -> Unfold windows over time
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )

        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )

        # Day adaptation (init as identity)
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # GRU
        self.gru_decoder = nn.GRU(
            neural_dim * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        # init
        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        rnn_out_dim = hidden_dim * 2 if self.bidirectional else hidden_dim

        # Optional "post-RNN normalization stack" (LayerNorm -> Dropout -> Linear -> Act)
        if postnet_activation.lower() == "gelu":
            Act = nn.GELU
        elif postnet_activation.lower() in ("silu", "swish"):
            Act = nn.SiLU
        else:
            Act = nn.ReLU

        blocks = []
        for _ in range(int(postnet_layers)):
            blocks += [
                nn.LayerNorm(rnn_out_dim),
                nn.Dropout(float(postnet_dropout)),
                nn.Linear(rnn_out_dim, rnn_out_dim),
                Act(),
            ]
        self.postnet = nn.Sequential(*blocks) if blocks else nn.Identity()

        # Final logits to classes + CTC blank
        self.fc_decoder_out = nn.Linear(rnn_out_dim, n_classes + 1)

    def forward(self, neuralInput, dayIdx):
        # neuralInput: (B,T,D)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))  # (B,D,T)
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))  # (B,T,D)

        # day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)  # (B,D,D) if dayIdx is (B,)
        transformedNeural = torch.einsum("btd,bdk->btk", neuralInput, dayWeights) + torch.index_select(
            self.dayBias, 0, dayIdx
        )
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel via Unfold
        # transformedNeural (B,T,D) -> (B,D,T,1) -> unfold -> (B, D*kernelLen, T') -> permute -> (B,T',D*kernelLen)
        stridedInputs = torch.permute(
            self.unfolder(torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)),
            (0, 2, 1),
        )

        # GRU initial state
        num_dir = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.layer_dim * num_dir,
            transformedNeural.size(0),
            self.hidden_dim,
            device=self.device,
        )
        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # Postnet + logits
        hid = self.postnet(hid)
        seq_out = self.fc_decoder_out(hid)  # (B,T',C)
        return seq_out


# -------------------------
# Data loaders
# -------------------------
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


# -------------------------
# Training
# -------------------------
def trainModel(args: dict):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = args.get("device", "cuda")

    with open(os.path.join(args["outputDir"], "args"), "wb") as file:
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
        # optional postnet stack
        postnet_layers=args.get("postnet_layers", 0),
        postnet_dropout=args.get("postnet_dropout", 0.2),
        postnet_activation=args.get("postnet_activation", "gelu"),
    ).to(device)

    # NOTE: your decoding removes token 0 => keep blank=0 to match your pipeline
    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args["l2_decay"],
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )

    # Optional training tricks (all default to "off" if missing)
    speckle_p = float(args.get("speckle_p", 0.0))
    blank_reg_w = float(args.get("blank_reg_weight", 0.0))
    step_decay = bool(args.get("step_decay", False))
    step_at = int(args.get("step_at", 7500))
    step_factor = float(args.get("step_factor", 0.1))

    testLoss, testCER = [], []
    startTime = time.time()

    # FIX: persistent iterator
    train_iter = iter(trainLoader)

    for batch in range(int(args["nBatch"])):
        model.train()

        # Optional step LR drop
        if step_decay and batch == step_at:
            for pg in optimizer.param_groups:
                pg["lr"] *= step_factor

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

        # Speckled masking
        if speckle_p > 0:
            X = speckle_mask(X, speckle_p)

        # Noise augmentations (your originals)
        if args["whiteNoiseSD"] > 0:
            X = X + torch.randn_like(X) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X = X + (torch.randn([X.shape[0], 1, X.shape[2]], device=device) * args["constantOffsetSD"])

        pred = model(X, dayIdx)  # (B,T',C)
        log_probs = pred.log_softmax(dim=2)  # (B,T',C)
        in_lens = ctc_input_lengths(X_len, model.kernelLen, model.strideLen)

        ctc_loss = loss_ctc(
            log_probs.permute(1, 0, 2),
            y,
            in_lens,
            y_len.to(torch.int32),
        )

        # Encourage non-blank emissions by penalizing blank probability
        if blank_reg_w > 0:
            p_blank = log_probs.exp()[..., 0].mean()
            loss = ctc_loss + blank_reg_w * p_blank
        else:
            loss = ctc_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ---- Eval ----
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

                    pred = model(X, testDayIdx)
                    log_probs = pred.log_softmax(dim=2)
                    in_lens = ctc_input_lengths(X_len, model.kernelLen, model.strideLen)

                    loss = loss_ctc(
                        log_probs.permute(1, 0, 2),
                        y,
                        in_lens,
                        y_len.to(torch.int32),
                    )
                    allLoss.append(loss.item())

                    # Greedy CER
                    for i in range(pred.shape[0]):
                        T = int(in_lens[i].item())
                        decoded = pred[i, :T, :].argmax(dim=-1)
                        decoded = torch.unique_consecutive(decoded)
                        decoded = decoded[decoded != 0].detach().cpu().numpy()

                        trueSeq = y[i, : int(y_len[i].item())].detach().cpu().numpy()

                        matcher = SequenceMatcher(a=trueSeq.tolist(), b=decoded.tolist())
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgLoss = float(np.mean(allLoss))
                cer = total_edit_distance / max(total_seq_length, 1)

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            # save best
            if len(testCER) == 0 or cer < np.min(testCER):
                torch.save(model.state_dict(), os.path.join(args["outputDir"], "modelWeights"))

            testLoss.append(avgLoss)
            testCER.append(cer)

            with open(os.path.join(args["outputDir"], "trainingStats"), "wb") as file:
                pickle.dump({"testLoss": np.array(testLoss), "testCER": np.array(testCER)}, file)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = os.path.join(modelDir, "modelWeights")
    with open(os.path.join(modelDir, "args"), "rb") as handle:
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
        postnet_layers=args.get("postnet_layers", 0),
        postnet_dropout=args.get("postnet_dropout", 0.2),
        postnet_activation=args.get("postnet_activation", "gelu"),
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model

