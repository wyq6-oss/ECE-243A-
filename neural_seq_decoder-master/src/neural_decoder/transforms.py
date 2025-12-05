import torch

class SignedLog1p:
    """sign(x)*log1p(|x|) works for any real-valued features."""
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sign(X) * torch.log1p(torch.abs(X))

class Log1pNonneg:
    """log1p(x) for nonnegative features only."""
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp(X, min=0.0))

class ZScore:
    """Z-score per feature using provided mean/std (torch tensors)."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / (self.std + self.eps)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
