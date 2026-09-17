import torch
from torch.nn.modules.loss import _Loss


class KLDivergenceLoss(_Loss):
    def __init__(self, deterministic=False, feat_dim=-1, size_average=None, reduce=None, reduction: str = "mean"):
        """
        Initialize the KLDivergence loss.

        Args:
            feat_dim (int, optional): 
                Dimensions along the mean and logvar of the input parameters `feat_dim`,
                or a list of two tensors [mean, logvar].
            deterministic (bool, optional): If True, the distribution is deterministic (zero variance). 
                Default is False. feat_dim (int, optional): Dimension along which mean and logvar are 
                concatenated if parameters is a single tensor. Default is 1.
        """
        super().__init__(size_average, reduce, reduction)
        self.feat_dim = feat_dim

    def forward(self, parameters, other, dims=(1, 2)):
        """
        Compute the Kullback-Leibler (KL) divergence between this distribution and another.

        If `other` is None, compute KL divergence to a standard normal distribution N(0, I).

        Args:
            other (DiagonalGaussianDistribution, optional): Another diagonal Gaussian distribution.
            dims (tuple, optional): Dimensions along which to compute the mean KL divergence. 
                Default is (1, 2, 3).

        Returns:
            torch.Tensor: The mean KL divergence value.
        """
        if isinstance(parameters, list):
            mean = parameters[0]
            logvar = parameters[1]
        else:
            mean, logvar = torch.chunk(parameters, 2, dim=self.feat_dim)
        var = torch.exp(logvar)

        
        if isinstance(other, list):
            other_mean = other[0]
            other_logvar = other[1]
        else:
            other_mean, other_logvar = torch.chunk(other, 2, dim=self.feat_dim)
        # Clamp before exp() to prevent float32 underflow (< -87) → 0 → NaN in division
        logvar       = torch.clamp(logvar,       -30.0, 20.0)
        other_logvar = torch.clamp(other_logvar, -30.0, 20.0)
        var       = torch.exp(logvar)
        other_var = torch.exp(other_logvar)
        
        loss_kl = (0.5 * torch.mean(
            torch.pow(mean - other_mean, 2) / other_var
            + var / other_var - 1.0 - logvar + other_logvar,
            dim=dims))
        if self.reduction == "mean":
            return loss_kl.mean()
        else:
            return loss_kl
