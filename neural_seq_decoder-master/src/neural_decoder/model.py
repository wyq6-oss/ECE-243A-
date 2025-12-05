import torch
from torch import nn

from .augmentations import GaussianSmoothing


class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        usePreLayerNorm=False,   # NEW: normalize before GRU
        usePostLayerNorm=True,   # NEW: normalize GRU outputs
        ln_eps=1e-5,             # NEW: LN epsilon
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

        self.usePreLayerNorm = usePreLayerNorm
        self.usePostLayerNorm = usePostLayerNorm

        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )

        # Day-specific affine (kept as-is)
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # --- NEW: Pre-LN on per-timepoint features (after day layer) ---
        # transformedNeural has shape [B, T, neural_dim]
        if self.usePreLayerNorm:
            self.pre_ln = nn.LayerNorm(neural_dim, eps=ln_eps)

        # GRU
        self.gru_decoder = nn.GRU(
            neural_dim * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Output dim (bidirectional doubles hidden size)
        rnn_out_dim = hidden_dim * 2 if self.bidirectional else hidden_dim

        # --- NEW: Post-LN on GRU outputs ---
        # hid has shape [B, T', rnn_out_dim]
        if self.usePostLayerNorm:
            self.post_ln = nn.LayerNorm(rnn_out_dim, eps=ln_eps)

        # Classifier
        self.fc_decoder_out = nn.Linear(rnn_out_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        # Smooth in time (your original)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # Day-specific linear
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum("btd,bdk->btk", neuralInput, dayWeights) + \
                            torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # NEW: Pre-LN (optional)
        if self.usePreLayerNorm:
            transformedNeural = self.pre_ln(transformedNeural)

        # Stride/kernel unfold
        stridedInputs = torch.permute(
            self.unfolder(torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)),
            (0, 2, 1),
        )

        # Init hidden
        num_dirs = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.layer_dim * num_dirs,
            transformedNeural.size(0),
            self.hidden_dim,
            device=self.device,
        )

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # NEW: Post-LN (optional)
        if self.usePostLayerNorm:
            hid = self.post_ln(hid)

        return self.fc_decoder_out(hid)
