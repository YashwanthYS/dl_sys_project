import sys
sys.path.append('python')
sys.path.append('apps/sepulative_decoding')

import numpy as np
import needle as ndl
from needle import ops

from spec_decode_copy import verify_chunk, speculative_generate, draft_propose_k


class DummyVerifier:
    def forward(self, x):
        # x: (1, T) float32 with integer values
        x_ids = x.numpy().astype(np.int64)[0]
        T = x_ids.shape[0]
        V = 16
        logits = np.zeros((T, V), dtype=np.float32)
        # Predict next token = input[t+1]
        for t in range(T - 1):
            nxt = int(x_ids[t + 1] % V)
            logits[t, nxt] = 1.0
        # last position unused
        return ndl.Tensor(logits[None, :, :], device=x.device, dtype='float32', requires_grad=False)

    __call__ = forward


class DummyVerifierMismatch(DummyVerifier):
    def __init__(self, mismatch_pos=0):
        self.mismatch_pos = mismatch_pos

    def forward(self, x):
        out = super().forward(x).numpy()
        # flip argmax at mismatch_pos if within range
        if self.mismatch_pos < out.shape[1]:
            row = out[0, self.mismatch_pos]
            j = int(np.argmax(row))
            row[j] = 0.0
            row[(j + 1) % row.shape[0]] = 1.0
        return ndl.Tensor(out, device=x.device, dtype='float32', requires_grad=False)


class DummyDraft:
    def __init__(self, truth):
        self.truth = np.array(truth, dtype=np.int64)

    def forward(self, x):
        # Return logits where argmax at pos t equals truth[t+1]
        ids = x.numpy().astype(np.int64)[0]
        T = ids.shape[0]
        V = 16
        logits = np.zeros((T, V), dtype=np.float32)
        for t in range(min(T - 1, self.truth.shape[0] - 1)):
            nxt = int(self.truth[t + 1] % V)
            logits[t, nxt] = 1.0
        return ndl.Tensor(logits[None, :, :], device=x.device, dtype='float32', requires_grad=False)

    __call__ = forward


def test_verify_chunk_accept_all():
    prefix = np.array([1, 2, 3], dtype=np.int64)
    drafted = np.array([4, 5], dtype=np.int64)
    acc, vnext = verify_chunk(DummyVerifier(), prefix, drafted, ndl.cpu())
    assert acc == len(drafted)
    assert vnext is None or isinstance(vnext, int)


def test_verify_chunk_mismatch():
    prefix = np.array([1, 2, 3], dtype=np.int64)
    drafted = np.array([4, 5, 6], dtype=np.int64)
    # force mismatch at first verification step (pos = len(prefix)-1)
    acc, vnext = verify_chunk(DummyVerifierMismatch(mismatch_pos=len(prefix) - 1), prefix, drafted, ndl.cpu())
    assert acc == 0
    assert isinstance(vnext, int)


def test_speculative_generate_full_accept():
    # Ground truth: <bos>=1 then 2,3,4,5
    truth = [1, 2, 3, 4, 5]
    prefix = np.array(truth[:3], dtype=np.int64)
    n = len(truth) - len(prefix)
    draft = DummyDraft(truth)
    verifier = DummyVerifier()
    gen, acc, drafted_total = speculative_generate(draft, verifier, prefix, n, k=4, device=ndl.cpu())
    assert acc == n
    assert drafted_total >= n
    np.testing.assert_array_equal(gen[-n:], np.array(truth[-n:], dtype=np.int64))

