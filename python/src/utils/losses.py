"""Domain-adaptation losses with numerical safeguards."""

import torch


def coral_loss(source, target):
    """Deep CORAL loss with a sqrt guard at zero."""
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    if ns <= 1 or nt <= 1:
        return torch.tensor(0.0, device=source.device, requires_grad=True)

    tmp_s = torch.ones((1, ns), device=source.device) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    tmp_t = torch.ones((1, nt), device=target.device) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    loss = (cs - ct).pow(2).sum()
    loss = torch.sqrt(loss + 1e-8)
    return loss / (4 * d * d)


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Maximum mean discrepancy with a multi-bandwidth Gaussian kernel."""
    if source.size(0) <= 1 or target.size(0) <= 1:
        return torch.tensor(0.0, device=source.device, requires_grad=True)

    batch_size = int(source.size(0))
    kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul,
                              kernel_num=kernel_num, fix_sigma=fix_sigma)
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return torch.mean(xx + yy - xy - yx)


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        denom = n_samples ** 2 - n_samples
        bandwidth = torch.sum(l2_distance.data) / denom if denom > 0 else 1.0
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-l2_distance / b) for b in bandwidth_list]
    return sum(kernel_val)


def dcca_loss(view1, view2, epsilon=1e-3):
    """Deep CCA loss with dimension truncation and numerical fallbacks.

    The covariance matrices become singular when the feature dimension exceeds
    the number of samples; the views are therefore truncated to N = batch-1
    dimensions, and a small diagonal epsilon is added before Cholesky.
    """
    view1 = view1 - view1.mean(dim=0)
    view2 = view2 - view2.mean(dim=0)

    n = view1.size(0) - 1
    if n <= 0:
        return torch.tensor(0.0, device=view1.device, requires_grad=True)

    if view1.size(1) > n:
        view1 = view1[:, :n]
    if view2.size(1) > n:
        view2 = view2[:, :n]

    sigma11 = view1.t() @ view1 / n + epsilon * torch.eye(view1.size(1), device=view1.device)
    sigma22 = view2.t() @ view2 / n + epsilon * torch.eye(view2.size(1), device=view2.device)
    sigma12 = view1.t() @ view2 / n

    try:
        l1 = torch.linalg.cholesky(sigma11)
        l2 = torch.linalg.cholesky(sigma22)
        tmp = torch.linalg.solve_triangular(l1, sigma12, upper=False)
        t_t = torch.linalg.solve_triangular(l2, tmp.t(), upper=False)
        t = t_t.t()
    except RuntimeError:
        return torch.tensor(0.0, device=view1.device, requires_grad=True)

    try:
        _, s, _ = torch.linalg.svd(t)
        return -torch.sum(s)
    except RuntimeError:
        return torch.tensor(0.0, device=view1.device, requires_grad=True)
