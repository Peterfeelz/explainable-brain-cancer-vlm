"""Reliability-weighted distillation loss — Step 9 (Equations 3.15-3.17)."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def distill_loss(z_local: torch.Tensor, z_cloud: torch.Tensor | None, y_star: torch.Tensor,
                 plan_nll: torch.Tensor | None, weight: torch.Tensor, teacher_wrong: torch.Tensor,
                 T_d: float = 2.0, lam: float = 0.7, mu: float = 0.5) -> torch.Tensor:
    """
    z_local      [B, C] student class-token logits
    z_cloud      [B, C] teacher class-token logits (None -> hard labels only)
    y_star       [B]    best available label (clinician, else argmax teacher)
    plan_nll     [B]    mean token NLL of verified plan (Eq. 3.16), or None
    weight       [B]    omega_i (Eq. 3.17)
    teacher_wrong[B]    bool; lambda forced to 0 where clinician overruled the teacher
    """
    z_local = z_local.float()
    hard = F.cross_entropy(z_local, y_star, reduction="none")
    if z_cloud is not None:
        z_cloud = z_cloud.float()
        soft = F.kl_div(F.log_softmax(z_local / T_d, -1), F.softmax(z_cloud / T_d, -1),
                        reduction="none").sum(-1) * (T_d ** 2)
        lam_i = torch.where(teacher_wrong, torch.zeros_like(hard), torch.full_like(hard, lam))
        per = lam_i * soft + (1 - lam_i) * hard
    else:
        per = hard
    if plan_nll is not None:
        per = per + mu * plan_nll.float()
    return (weight.float() * per).mean()          # Eq. 3.18: (1/B) * sum_i omega_i * L_i
