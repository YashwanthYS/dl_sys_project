import sys
sys.path.append('python')
sys.path.append('apps/sepulative_decoding')

import numpy as np
import needle as ndl

from common import CopyDataset, Loader, GPTTinyNeedle, masked_cross_entropy, PAD_ID, VOCAB_SIZE


def test_copydataset_and_loader_shapes():
    ds = CopyDataset(num_examples=5, max_seq_len=12, seed=0)
    loader = Loader(ds, batch_size=3, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    b0 = batches[0]
    assert b0.ndim == 2
    assert b0.shape[0] == 3
    # padded with PAD_ID
    assert (b0[:, -1] == PAD_ID).any() or (b0[:, -1] != PAD_ID).any()


def test_gpttiny_forward_and_loss():
    device = ndl.cpu()
    model = GPTTinyNeedle(VOCAB_SIZE, d_model=16, n_head=2, n_layer=1, max_len=12, dropout=0.0, device=device)
    # create a small batch (B=2, T=6)
    x = np.array([[1,2,3,4,5,6],[1,2,3,4,0,0]], dtype=np.float32)
    logits = model(ndl.Tensor(x, device=device))
    assert logits.shape == (2, 6, VOCAB_SIZE)
    # targets offset by 1 (ignore PAD)
    y = x.copy()
    y[:,0] = PAD_ID
    loss = masked_cross_entropy(logits, ndl.Tensor(y, device=device))
    val = float(loss.numpy())
    assert np.isfinite(val)

