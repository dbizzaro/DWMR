import torch
import torch.nn.functional as F


def prediction_loss(pred: torch.Tensor, bits_tgt: torch.Tensor, loss_type="cross-entropy") -> torch.Tensor:
    if loss_type == "mse":
        pred = F.sigmoid(pred)
        return F.mse_loss(pred, bits_tgt)
    elif loss_type == "cross-entropy":
        return F.binary_cross_entropy_with_logits(pred, bits_tgt)
    elif loss_type == "l1":
        pred = F.sigmoid(pred)
        return F.l1_loss(pred, bits_tgt)
    else:
        raise ValueError(f"Unknown jepa_loss: {loss_type}")


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m, "off_diagonal expects a square matrix"
    return x.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()


def hamming_loss(z, z_pred, target_bits=(1, 6)):
    K = z.shape[-1]
    dist = torch.abs(z - z_pred) # Soft Hamming distance per sample
    mask = dist > 0.5 
    dist = (dist * mask).sum(dim=-1) / 0.75 # Sum of dist such that dist > 0.5, normalized
    frac = dist / K
    low, high = target_bits
    target = (low + high) / (2 * K)   # midpoint as normalized target
    width = (high - low) / (2 * K)
    loss = F.relu(torch.abs(frac - target) - width) ** 2  # Squared penalty if outside target range (parabolic around target)
    return loss.mean()
    

def kl_divergence(p):
    p = p.clamp(1e-6, 1 - 1e-6) # for numerical stability
    kl = p * torch.log(p / 0.5) + (1 - p) * torch.log((1 - p) / 0.5)
    return kl.sum(dim=-1).mean()


def var_cov_cos_regularizers(p, hinge_std, third_order=True, normalize=True):
    """
    p: [N, K] raw outputs from network
    """
    p = p.clamp(1e-6, 1 - 1e-6)  # for numerical stability
    
    # ---- Variance term
    batch_variance = p.var(dim=0) # [K]
    std = torch.sqrt(batch_variance + 1e-6)
    variance_loss = F.relu(hinge_std - std).mean()

    # normalize p
    p_centered = p - p.mean(dim=0, keepdim=True)
    if normalize: 
        p_centered = p_centered / torch.sqrt(batch_variance + 1e-6).unsqueeze(0)
    
    # ---- Covariance term
    cov = (p_centered.T @ p_centered) / (p.shape[0] - 1 + 1e-6)
    cov_off = _off_diagonal(cov)
    cov_loss = cov_off.abs().mean()

    # ---- Third-order correlation
    if third_order:
        N, B = p.shape
        # Full 3rd-order moment tensor: M[b,c,d] = E_n x[n,b]*x[n,c]*x[n,d]
        M = torch.einsum('nb,nc,nd->bcd', p_centered, p_centered, p_centered) / float(N)      # (B, B, B)
        # Mask out any indices with repeats: i==j or j==k or i==k
        eye = torch.eye(B, device=p.device, dtype=torch.bool)
        rep1 = eye.unsqueeze(2).expand(B, B, B)    # i==j
        rep2 = eye.unsqueeze(0).expand(B, B, B)    # j==k
        rep3 = eye.unsqueeze(1).expand(B, B, B)    # i==k
        mask_distinct = ~(rep1 | rep2 | rep3)      # True only for i,j,k all distinct
        third_order_loss = M[mask_distinct].abs().mean()
    else:
        third_order_loss = torch.tensor(0.0)

    return variance_loss, cov_loss, third_order_loss