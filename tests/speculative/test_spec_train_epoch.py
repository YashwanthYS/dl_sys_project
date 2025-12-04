import sys
sys.path.append('python')
sys.path.append('apps/sepulative_decoding')

import numpy as np
import needle as ndl

from spec_decode_copy import train_epoch


class NoOpOpt:
    def reset_grad(self):
        pass
    def step(self):
        pass


class TinyModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def train(self):
        pass

    def __call__(self, x):
        # x: Tensor (B,T)
        B, T = x.shape
        V = self.vocab_size
        logits = np.zeros((B, T, V), dtype=np.float32)
        # make it deterministic but cheap
        logits[:, :, 0] = 1.0
        return ndl.Tensor(logits, device=x.device, dtype='float32', requires_grad=False)


def test_train_epoch_microbatch():
    device = ndl.cpu()
    # batch of 5 sequences of length 6
    batch = np.stack([np.array([1,2,3,4,5,6], dtype=np.float32) for _ in range(5)], axis=0)
    loader = [batch]
    model = TinyModel(vocab_size=16)
    train_epoch(model, loader, NoOpOpt(), device, print_every=1, micro_batch_size=2)


def test_train_epoch_single_batch():
    device = ndl.cpu()
    batch = np.stack([np.array([1,2,3,4], dtype=np.float32) for _ in range(3)], axis=0)
    loader = [batch]
    model = TinyModel(vocab_size=8)
    train_epoch(model, loader, NoOpOpt(), device, print_every=1, micro_batch_size=0)
