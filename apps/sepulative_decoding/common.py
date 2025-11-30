import numpy as np
import needle as ndl
import needle.nn as nn
from needle import ops

# Vocabulary
ALPHABET = list("abcdef")
SPECIAL = ["<pad>", "<bos>", "#"]
itos = SPECIAL + ALPHABET
stoi = {ch: i for i, ch in enumerate(itos)}
PAD_ID = stoi["<pad>"]
BOS_ID = stoi["<bos>"]
SEP_ID = stoi["#"]
VOCAB_SIZE = len(itos)


class CopyDataset:
    def __init__(self, num_examples, max_seq_len=32, seed=42):
        self.num_examples = num_examples
        self.max_seq_len = max_seq_len
        self.max_copy_len = (max_seq_len - 2) // 2
        self.rng = np.random.default_rng(seed)
        self.data = [self._make_example() for _ in range(num_examples)]

    def _make_example(self):
        n = int(self.rng.integers(1, self.max_copy_len + 1))
        base_ids = self.rng.integers(low=3, high=VOCAB_SIZE, size=n, dtype=np.int64)
        seq = np.empty(2 * n + 2, dtype=np.int64)
        seq[0] = BOS_ID
        seq[1 : 1 + n] = base_ids
        seq[1 + n] = SEP_ID
        seq[2 + n :] = base_ids
        return seq

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.data[idx]


def collate_pad(batch):
    lengths = [len(x) for x in batch]
    max_len = max(lengths)
    B = len(batch)
    out = np.full((B, max_len), PAD_ID, dtype=np.int64)
    for i, seq in enumerate(batch):
        out[i, : len(seq)] = seq
    return out


class Loader:
    def __init__(self, dataset, batch_size=128, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        idxs = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idxs)
        for s in range(0, len(idxs), self.batch_size):
            batch_idx = idxs[s : s + self.batch_size]
            yield collate_pad([self.dataset[i] for i in batch_idx])


class GPTTinyNeedle(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=4, n_layer=2, max_len=32, dropout=0.0, device=None, dtype="float32"):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer = nn.Transformer(
            embedding_size=d_model,
            hidden_size=4 * d_model,
            num_layers=n_layer,
            num_head=n_head,
            dim_head=d_model // n_head,
            dropout=dropout,
            causal=True,
            device=device,
            dtype=dtype,
            batch_first=False,
            sequence_len=max_len,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x_ids):
        x_tb = ops.transpose(x_ids, axes=(1, 0))
        h_tbd = self.tok_emb(x_tb)
        y_tbd, _ = self.transformer(h_tbd)
        y_btd = ops.transpose(y_tbd, axes=(1, 0))
        B, T, D = y_btd.shape
        y2d = ops.reshape(y_btd, (B * T, D))
        logits2d = self.lm_head(y2d)
        return ops.reshape(logits2d, (B, T, self.vocab_size))


def masked_cross_entropy(logits, targets):
    B, T, V = logits.shape
    logits2d = ops.reshape(logits, (B * T, V))
    targets2d = ops.reshape(targets, (B * T,))
    log_probs = ops.logsoftmax(logits2d)
    one_hot = ndl.init.one_hot(V, targets2d, device=logits.device, dtype=logits.dtype)
    nll = -ops.summation(log_probs * one_hot, axes=(1,))
    tgt_np = targets2d.numpy().astype(np.int32)
    mask_np = (tgt_np != PAD_ID).astype(np.float32)
    mask = ndl.Tensor(mask_np, device=logits.device, dtype=logits.dtype, requires_grad=False)
    masked_nll = nll * mask
    denom = float(mask.numpy().sum().item()) + 1e-8
    return ops.summation(masked_nll) / denom

