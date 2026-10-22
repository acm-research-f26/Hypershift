"""Study FB Task 8 diagnostic (exploratory, synthetic only): do soft hyperedges learn planted groups?

Usage: python scripts/featbench_soft_diag.py SIGMA KIND   (KIND: soft | none; soft_t was a reverted variant)
Fresh synthetic batches every step: 4 planted groups x 16 stocks, white-noise group factor, own noise SIGMA.
"""
import sys
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, 'src')
from hypershift.screening.models import ScreenModel

kind = sys.argv[2] if len(sys.argv) > 2 else 'soft_t'
sigma = float(sys.argv[1])
torch.set_num_threads(4)
rng = np.random.default_rng(5)
n, g, b = 64, 4, 16
planted = np.repeat(np.arange(g), n // g)
noisy = planted.copy(); flip = rng.random(n) < .5; noisy[flip] = rng.integers(0, 16, flip.sum())
P = torch.as_tensor(planted)


def batch():
    factor = torch.randn(b, g, 64)
    x = factor[:, P][..., None].repeat(1, 1, 1, 5) + torch.randn(b, n, 64, 5) * sigma
    return x, factor[:, P, -1]


eligible = torch.ones(b, n, dtype=torch.bool)
torch.manual_seed(5)
model = ScreenModel(kind, nodes=n, hidden=16, groups=noisy)
opt = torch.optim.Adam(model.parameters(), lr=.01)
for step in range(1, 601):
    x, y = batch()
    opt.zero_grad(); loss = ((model(x, eligible) - y)**2).mean() + getattr(model, 'aux_loss', 0.); loss.backward(); opt.step()
    if step in (1, 100, 300, 600):
        with torch.no_grad():
            x, y = batch(); mse = float(((model(x, eligible) - y)**2).mean())
        a = getattr(model, 'last_assignment', None)
        ari = adjusted_rand_score(planted, a.mean((0, 1)).argmax(-1).numpy()) if a is not None else float('nan')
        print(kind, step, 'fresh mse', round(mse, 4), 'own-only bound', round(sigma**2/5/(1+sigma**2/5), 4),
              'ARI', round(ari, 3), 'entropy', round(getattr(model, 'assignment_entropy', float('nan')), 3), flush=True)
