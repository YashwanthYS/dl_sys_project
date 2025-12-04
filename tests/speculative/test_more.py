import sys
sys.path.append('python')
sys.path.append('apps/sepulative_decoding')

import numpy as np
import needle as ndl
from needle import ops

from spec_decode_copy import greedy_next, draft_propose_k, speculative_generate, evaluate_speculative


def test_greedy_next_simple():
    class M:
        def __call__(self, x):
            T = x.numpy().shape[1]
            V = 10
            logits = np.zeros((1, T, V), dtype=np.float32)
            logits[0, -1, 7] = 1.0
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    out = greedy_next(M(), np.array([1, 2, 3], dtype=np.int64), ndl.cpu())
    assert out == 7


def test_draft_propose_k_constant():
    class Draft:
        def __call__(self, x):
            T = x.numpy().shape[1]
            V = 8
            logits = np.zeros((1, T, V), dtype=np.float32)
            logits[0, -1, 1] = 1.0
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    prefix = np.array([3, 4], dtype=np.int64)
    k = 4
    out = draft_propose_k(Draft(), prefix, k=k, device=ndl.cpu())
    assert out.shape == (k,)
    assert (out == 1).all()


def test_speculative_generate_progress():
    class Draft:
        def __call__(self, x):
            T = x.numpy().shape[1]
            V = 8
            logits = np.zeros((1, T, V), dtype=np.float32)
            logits[0, -1, 2] = 1.0
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    class Ver(Draft):
        pass

    prefix = np.array([1, 2], dtype=np.int64)
    gen, acc, drafted_total = speculative_generate(Draft(), Ver(), prefix, total_tokens=5, k=2, device=ndl.cpu())
    assert gen.shape[0] == prefix.shape[0] + 5
    assert acc == 5
    assert drafted_total >= 5


def test_evaluate_speculative_metrics_keys():
    class Draft:
        def __call__(self, x):
            T = x.numpy().shape[1]
            V = 8
            logits = np.zeros((1, T, V), dtype=np.float32)
            logits[0, -1, 1] = 1.0
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    class Ver(Draft):
        pass

    m = evaluate_speculative(Draft(), Ver(), num_examples=5, max_seq_len=10, k=1, device=ndl.cpu())
    for key in ["tokens_per_sec", "acceptance_rate", "latency_p50", "token_accuracy"]:
        assert key in m
        assert isinstance(m[key], float)


def test_batched_matmul_grad_shapes():
    # Small grad flow check: grad shapes should match inputs
    B, M, K, N = 2, 3, 4, 5
    a = ndl.init.randn(B, M, K, requires_grad=True)
    b = ndl.init.randn(B, K, N, requires_grad=True)
    y = ops.batched_matmul(a, b)
    loss = ops.summation(y)
    loss.backward()
    assert a.grad.shape == (B, M, K)
    assert b.grad.shape == (B, K, N)

