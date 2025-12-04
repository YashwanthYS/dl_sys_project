import sys
sys.path.append('python')
sys.path.append('apps/sepulative_decoding')

import numpy as np
import needle as ndl

from spec_decode_copy import verify_chunk, speculative_generate, draft_propose_k


def test_verify_chunk_empty_drafted():
    class Ver:
        def __call__(self, x):
            T = x.numpy().shape[1]
            V = 8
            logits = np.zeros((1, T, V), dtype=np.float32)
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    prefix = np.array([1, 2, 3], dtype=np.int64)
    drafted = np.array([], dtype=np.int64)
    acc, vnext = verify_chunk(Ver(), prefix, drafted, ndl.cpu())
    assert acc == 0
    assert vnext is None


def test_draft_propose_k_length():
    class Draft:
        def __call__(self, x):
            T = x.numpy().shape[1]
            V = 16
            logits = np.zeros((1, T, V), dtype=np.float32)
            logits[0, -1, 1] = 1.0  # always propose token 1
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    prefix = np.array([3, 4], dtype=np.int64)
    out = draft_propose_k(Draft(), prefix, k=5, device=ndl.cpu())
    assert out.shape == (5,)


def test_speculative_generate_no_tokens():
    class Draft:
        def __call__(self, x):
            T = x.numpy().shape[1]
            V = 8
            logits = np.zeros((1, T, V), dtype=np.float32)
            logits[0, -1, 0] = 1.0
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    class Ver(Draft):
        pass

    prefix = np.array([1, 2], dtype=np.int64)
    gen, acc, drafted_total = speculative_generate(Draft(), Ver(), prefix, total_tokens=0, k=3, device=ndl.cpu())
    np.testing.assert_array_equal(gen, prefix)
    assert acc == 0
    assert drafted_total == 0

