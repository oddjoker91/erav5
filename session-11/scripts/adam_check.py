"""adam_check.py -- Part 1. The table was worked out by hand; PyTorch has to match it."""
import torch

from engine import ManualAdam

B1, B2, LR, EPS = 0.9, 0.999, 1e-3, 1e-8
W0 = 1.0
#    g     m               v               m_hat           v_hat           step            w after
HAND = [
    (0.5, 0.05, 0.00025, 0.5, 0.25, 0.00099999998, 0.99900000002),
    (0.4, 0.085, 0.00040975, 0.447368421053, 0.204977488744, 0.000988125782, 0.998011874238),
    (0.6, 0.1365, 0.00076934025, 0.5036900369, 0.2567033678, 0.000994140047, 0.997017734191),
    (0.45, 0.16785, 0.00097107091, 0.488077929631, 0.24313218264, 0.000989846693, 0.996027887498),
    (0.55, 0.206065, 0.001272599839, 0.503198945081, 0.255029516998, 0.000996424711, 0.995031462787),
]


def check_torch():
    w = torch.nn.Parameter(torch.tensor(W0, dtype=torch.float64))
    opt = torch.optim.Adam([w], lr=LR, betas=(B1, B2), eps=EPS)
    worst = 0.0
    print(f"{'t':>2} {'quantity':<7} {'hand':>16} {'pytorch':>16} {'difference':>12}")
    for t, (g, m, v, mh, vh, step, after) in enumerate(HAND, 1):
        before = w.item()
        w.grad = torch.tensor(g, dtype=torch.float64)
        opt.step()
        st = opt.state[w]
        tm, tv = st["exp_avg"].item(), st["exp_avg_sq"].item()
        got = dict(m=tm, v=tv, m_hat=tm / (1 - B1 ** t), v_hat=tv / (1 - B2 ** t),
                   step=before - w.item(), w=w.item())
        for name, hand in zip(got, (m, v, mh, vh, step, after)):
            diff = abs(hand - got[name])
            worst = max(worst, diff)
            print(f"{t:>2} {name:<7} {hand:>16.12f} {got[name]:>16.12f} {diff:>12.2e}")
    print(f"worst difference {worst:.2e}")
    assert worst < 1e-9


def check_manual():
    """ManualAdam must be AdamW when correction is on, or Part 2 tests the wrong thing."""
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(50, dtype=torch.float64))
    b = torch.nn.Parameter(a.detach().clone())
    oa = torch.optim.AdamW([a], lr=LR)
    ob = ManualAdam([b], lr=LR)
    for _ in range(200):
        g = torch.randn(50, dtype=torch.float64)
        a.grad, b.grad = g.clone(), g.clone()
        oa.step()
        ob.step()
    gap = (a - b).abs().max().item()
    print(f"ManualAdam vs torch AdamW after 200 steps: largest difference {gap:.2e}")
    assert gap < 1e-10


if __name__ == "__main__":
    check_torch()
    check_manual()
