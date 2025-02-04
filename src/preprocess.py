import torch

class PreprocessTransforms:
  def __init__(self, mask: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
    self.mask = mask
    self.mu = mu
    self.sigma = sigma
  
  def __call__(self, X: torch.Tensor) -> torch.Tensor:
    X_clamped = X.clamp(20, 16000)
    X_filtered = X_clamped[:, self.mask]
    X_log2 = X_filtered.log2()
    X_scaled = X_log2.sub(self.mu).div(self.sigma)
    return X_scaled