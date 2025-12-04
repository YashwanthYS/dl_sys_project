import sys
sys.path.append('python')

import numpy as np
import needle as ndl
from needle import ops


def test_batched_matmul_basic_cpu_np_parity():
    # Use numpy backend for reference
    # Shapes: (B,M,K) x (B,K,N) -> (B,M,N)
    B, M, K, N = 3, 4, 5, 2
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((B, M, K)).astype('float32')
    b_np = rng.standard_normal((B, K, N)).astype('float32')

    # Needle nd backend
    a = ndl.Tensor(a_np)
    b = ndl.Tensor(b_np)
    out = ops.batched_matmul(a, b)
    out_np = out.numpy()

    # Numpy reference
    ref = np.matmul(a_np, b_np)

    assert out_np.shape == ref.shape
    np.testing.assert_allclose(out_np, ref, atol=1e-5, rtol=1e-5)

