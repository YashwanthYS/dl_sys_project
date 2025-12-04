import sys
sys.path.append('python')
sys.path.append('apps/sepulative_decoding')

import numpy as np
import needle as ndl

import spec_decode_copy as sdc


def test_decode_ids_simple():
    ids = [1, 3, 4]
    out = sdc.decode_ids(ids)
    assert isinstance(out, str) and len(out) > 0


def test_greedy_generate_verifier_length():
    class Ver:
        def __call__(self, x):
            T = x.numpy().shape[1]
            V = 10
            logits = np.zeros((1, T, V), dtype=np.float32)
            logits[0, -1, 5] = 1.0
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    prefix = np.array([1, 2, 3], dtype=np.int64)
    out = sdc.greedy_generate_verifier(Ver(), prefix, total_tokens=4, device=ndl.cpu())
    assert out.shape[0] == prefix.shape[0] + 4


def test_evaluate_baseline_returns_keys():
    class Ver:
        def __call__(self, x):
            T = x.numpy().shape[1]
            V = 8
            logits = np.zeros((1, T, V), dtype=np.float32)
            logits[0, -1, 1] = 1.0
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    m = sdc.evaluate_baseline(Ver(), num_examples=3, max_seq_len=10, device=ndl.cpu())
    assert set(['tokens_per_sec','latency_p50','token_accuracy']).issubset(m.keys())


def test_speculative_generate_mismatch_first():
    class Draft:
        def __call__(self, x):
            T = x.numpy().shape[1]
            V = 8
            logits = np.zeros((1, T, V), dtype=np.float32)
            logits[0, -1, 1] = 1.0  # propose token 1
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    class Ver:
        def __call__(self, x):
            # Always predict token 2 (ensures mismatch at first check)
            T = x.numpy().shape[1]
            V = 8
            logits = np.zeros((1, T, V), dtype=np.float32)
            logits[0, :, 2] = 1.0
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    prefix = np.array([1, 2], dtype=np.int64)
    gen, acc, drafted_total = sdc.speculative_generate(Draft(), Ver(), prefix, total_tokens=3, k=2, device=ndl.cpu())
    # We should still make progress by 1 verified token
    assert gen.shape[0] >= prefix.shape[0] + 1
    assert acc == 0
    assert drafted_total >= 1


def test_speculative_generate_partial_accept():
    # Accept 1 token then mismatch
    class Draft:
        def __call__(self, x):
            T = x.numpy().shape[1]
            V = 8
            logits = np.zeros((1, T, V), dtype=np.float32)
            # propose token 3 always
            logits[0, -1, 3] = 1.0
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    class Ver:
        def __call__(self, x):
            T = x.numpy().shape[1]
            V = 8
            logits = np.zeros((1, T, V), dtype=np.float32)
            # At first check position (T0-1), predict 3 to accept; at T0, predict 4 to mismatch
            if T >= 2:
                logits[0, -2, 3] = 1.0  # accept first drafted token
                logits[0, -1, 4] = 1.0  # mismatch second drafted token
            else:
                logits[0, -1, 3] = 1.0
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    prefix = np.array([5, 6, 7], dtype=np.int64)
    gen, acc, drafted_total = sdc.speculative_generate(Draft(), Ver(), prefix, total_tokens=2, k=2, device=ndl.cpu())
    # One accepted, then one verified-next added → progress by 2 tokens
    assert gen.shape[0] == prefix.shape[0] + 2
    assert acc == 1
    assert drafted_total >= 2


def test_verify_chunk_out_of_bounds_prefix_empty():
    class Ver:
        def __call__(self, x):
            T = x.numpy().shape[1]
            V = 8
            logits = np.zeros((1, T, V), dtype=np.float32)
            logits[0, :, 1] = 1.0
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    prefix = np.array([], dtype=np.int64)
    drafted = np.array([3, 3], dtype=np.int64)
    acc, vnext = sdc.verify_chunk(Ver(), prefix, drafted, ndl.cpu())
    # pos starts at -1 and breaks immediately; accept stays 0
    assert acc == 0


def test_train_epoch_fallback_print_branch(monkeypatch):
    # Force the except branch in printing by returning a loss with numpy() that lacks .item
    class DummyLoss:
        def backward(self):
            pass
        class _NoItem:
            def __float__(self):
                return 0.0
        def numpy(self):
            return self._NoItem()

    def fake_mce(logits, tgt):
        return DummyLoss()

    monkeypatch.setattr(sdc, 'masked_cross_entropy', fake_mce)

    class M:
        def train(self):
            pass
        def __call__(self, x):
            B, T = x.shape
            V = 6
            logits = np.zeros((B, T, V), dtype=np.float32)
            return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)

    class NoOpt:
        def reset_grad(self):
            pass
        def step(self):
            pass

    batch = np.stack([np.array([1,2,3], dtype=np.float32) for _ in range(2)], axis=0)
    sdc.train_epoch(M(), [batch], NoOpt(), ndl.cpu(), print_every=1)
