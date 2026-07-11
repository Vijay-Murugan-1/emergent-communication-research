import torch
import torch.nn.functional as F

def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error."""
    return F.mse_loss(pred, target, reduction='none').view(pred.size(0), -1).mean(dim=1)

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio."""
    mse = compute_mse(pred, target)
    return 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))

def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Structural Similarity Index Measure (SSIM) approximation."""
    # This is a highly simplified SSIM approximation for native PyTorch without external deps
    mu1 = F.avg_pool2d(pred, 3, 1, 1)
    mu2 = F.avg_pool2d(target, 3, 1, 1)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(pred * pred, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, 3, 1, 1) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean(dim=[1, 2, 3])

def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """Computes percentage of pixels within a threshold."""
    diff = torch.abs(pred - target)
    correct = (diff < threshold).float()
    return correct.view(pred.size(0), -1).mean(dim=1)
