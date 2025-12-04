import sys
sys.path.append('python')
sys.path.append('apps/sepulative_decoding')

import os
import numpy as np
import needle as ndl
import tempfile

from analysis import (
    plot_loss_curves,
    plot_speedup_vs_k,
    plot_timing_breakdown,
    plot_acceptance_vs_ratio,
    evaluate_speculative_timing,
    train_model,
)


class Draft:
    def __call__(self, x):
        T = x.numpy().shape[1]
        V = 8
        logits = np.zeros((1, T, V), dtype=np.float32)
        logits[0, -1, 1] = 1.0
        return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)


class Ver(Draft):
    pass


def test_plot_helpers_and_timing():
    # plotting
    with tempfile.TemporaryDirectory() as d:
        plot_loss_curves([1.0, 0.8], [1.2, 0.9], os.path.join(d, 'loss.png'))
        plot_speedup_vs_k([2,4,6], [1.0, 1.2, 1.1], os.path.join(d, 'speed.png'))
        plot_timing_breakdown([2,4], [0.1, 0.2], [0.2, 0.3], [0.05, 0.07], os.path.join(d, 'time.png'))
        plot_acceptance_vs_ratio([0.5, 0.6], [0.8, 0.9], [1.1, 1.2], os.path.join(d, 'ratio.png'))

    # quick timing eval with tiny settings
    tb = evaluate_speculative_timing(Draft(), Ver(), num_examples=1, max_seq_len=8, k=1, device=ndl.cpu())
    assert set(['total_time','t_draft','t_verify','t_overhead','tokens_per_sec','acceptance_rate']).issubset(tb.keys())


def test_train_model_runs_one_epoch():
    class Tiny:
        def train(self):
            pass
        def __call__(self, x):
            B, T = x.shape
            V = 8
            logits = np.zeros((B, T, V), dtype=np.float32)
            logits[:, :, 0] = 1.0
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    # single batch loader
    batch = np.stack([np.array([1,2,3,4], dtype=np.float32) for _ in range(2)], axis=0)
    loader = [batch]

    class NoOpt:
        def reset_grad(self):
            pass
        def step(self):
            pass

    curve = train_model(Tiny(), loader, NoOpt(), ndl.cpu(), epochs=1)
    assert isinstance(curve, list) and len(curve) == 1
