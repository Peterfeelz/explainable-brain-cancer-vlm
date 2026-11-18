import pytest
torch = pytest.importorskip("torch")
from bcd.losses import distill_loss

def test_loss_terms():
    zl = torch.tensor([[2.0, 0.1, 0.1, 0.1]], requires_grad=True)
    zc = torch.tensor([[2.0, 0.1, 0.1, 0.1]])
    y = torch.tensor([0]); w = torch.tensor([1.0]); tw = torch.tensor([False])
    l_same = distill_loss(zl, zc, y, None, w, tw, lam=1.0)
    assert l_same.item() < 1e-6                        # identical distributions -> KL 0
    l_hard = distill_loss(zl, None, y, None, w, tw)
    assert abs(l_hard.item() - torch.nn.functional.cross_entropy(zl, y).item()) < 1e-6
    # teacher_wrong forces lambda=0 (pure hard label)
    l_tw = distill_loss(zl, zc * -1, y, None, w, torch.tensor([True]), lam=0.7)
    assert abs(l_tw.item() - l_hard.item()) < 1e-6
    # weights scale the loss, plan term adds mu * nll
    l_w = distill_loss(zl, None, y, torch.tensor([2.0]), torch.tensor([0.5]), tw, mu=0.5)
    assert abs(l_w.item() - 0.5 * (l_hard.item() + 1.0)) < 1e-6
    l_w.backward(); assert zl.grad is not None
