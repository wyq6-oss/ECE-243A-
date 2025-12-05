import torch
from torch import nn

class SpeckleMask(nn.Module):
    """
    'Speckled masking' / coordinated dropout style masking.
    Masks random elements of the neural input and rescales by 1/(1-p).
    """
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if (not self.training) or self.p <= 0:
            return x
        keep = 1.0 - self.p
        mask = (torch.rand_like(x) < keep).to(x.dtype)
        return x * mask / keep
    
class GRUDecoder(nn.Module):
    def __init__(self, neural_dim, n_classes, hidden_dim, layer_dim,
                 nDays=24, dropout=0.0, device="cuda",
                 strideLen=4, kernelLen=14, gaussianSmoothWidth=0,
                 bidirectional=False,
                 speckle_p=0.3,            # <-- ADD (paper used 0.3)
                 postnet_layers=2,         # <-- ADD (stack depth)
                 postnet_dropout=0.2):     # <-- ADD
        super().__init__()
        ...
        self.speckle = SpeckleMask(speckle_p)  # <-- ADD

        # GRU stays the same
        self.gru_decoder = nn.GRU(
            neural_dim * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        rnn_out_dim = hidden_dim * 2 if self.bidirectional else hidden_dim

        # ---- ADD: post-RNN normalization stack ----
        blocks = []
        dim = rnn_out_dim
        for _ in range(postnet_layers):
            blocks += [
                nn.LayerNorm(dim),
                nn.Dropout(postnet_dropout),
                nn.Linear(dim, dim),
                nn.GELU(),
            ]
        self.postnet = nn.Sequential(*blocks) if blocks else nn.Identity()

        # final classifier (CTC blank = last index)
        self.fc_decoder_out = nn.Linear(dim, n_classes + 1)

    def forward(self, neuralInput, dayIdx):
        # neuralInput: (B, T, D)
        neuralInput = self.speckle(neuralInput)  # <-- ADD (before smoothing/day layer)

        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # day layer (unchanged)
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum("btd,bdk->btk", neuralInput, dayWeights) \
                           + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # unfold/stride (unchanged)
        stridedInputs = torch.permute(
            self.unfolder(torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)),
            (0, 2, 1),
        )

        # h0 (unchanged)
        num_dir = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.layer_dim * num_dir,
            transformedNeural.size(0),
            self.hidden_dim,
            device=self.device,
        )

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # ---- ADD: postnet then classifier ----
        hid = self.postnet(hid)
        seq_out = self.fc_decoder_out(hid)
        return seq_out
