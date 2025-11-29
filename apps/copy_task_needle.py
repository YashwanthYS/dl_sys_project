"""
Needle-based copy task demo:
- Synthetic dataset: sequences of the form <bos> x1..xn # x1..xn
- Tiny GPT-style model using Needle's Transformer
- Training with masked cross-entropy (ignoring PAD)
- Greedy generation evaluation on the copy task

Run: python apps/copy_task_needle.py
"""

import sys
import math
import numpy as np

sys.path.append("python/")
import needle as ndl
import needle.nn as nn
from needle import ops


# ============================================
# 1. Vocabulary & helpers
# ============================================
ALPHABET = list("abcdef")  # 6 symbols to copy
SPECIAL = ["<pad>", "<bos>", "#"]

itos = SPECIAL + ALPHABET
stoi = {ch: i for i, ch in enumerate(itos)}

PAD_ID = stoi["<pad>"]
BOS_ID = stoi["<bos>"]
SEP_ID = stoi["#"]
VOCAB_SIZE = len(itos)


def decode_ids(ids):
    return " ".join(itos[int(i)] for i in ids)


# ============================================
# 2. Synthetic copy dataset
# ============================================
class CopyDataset:
    def __init__(self, num_examples, max_seq_len=32, seed=42):
        self.num_examples = num_examples
        self.max_seq_len = max_seq_len
        # <bos> + x1..xn + # + x1..xn  => length = 2n + 2
        self.max_copy_len = (max_seq_len - 2) // 2
        self.rng = np.random.default_rng(seed)
        self.data = [self._make_example() for _ in range(num_examples)]

    def _make_example(self):
        # Choose length of base pattern
        n = int(self.rng.integers(1, self.max_copy_len + 1))
        # Sample base tokens from ALPHABET (which start at index 3 in vocab)
        base_ids = self.rng.integers(low=3, high=VOCAB_SIZE, size=n, dtype=np.int64)
        # Build sequence: <bos> x1..xn # x1..xn
        seq = np.empty(2 * n + 2, dtype=np.int64)
        seq[0] = BOS_ID
        seq[1 : 1 + n] = base_ids
        seq[1 + n] = SEP_ID
        seq[2 + n :] = base_ids
        assert seq.shape[0] <= self.max_seq_len
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


class SimpleDataLoader:
    def __init__(self, dataset, batch_size=128, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        idxs = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idxs)
        for start in range(0, len(idxs), self.batch_size):
            batch_idx = idxs[start : start + self.batch_size]
            batch = [self.dataset[i] for i in batch_idx]
            yield collate_pad(batch)


# ============================================
# 3. Tiny GPT-style model (Needle Transformer)
# ============================================
class GPTTinyNeedle(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_head=4,
        n_layer=2,
        max_len=32,
        dropout=0.1,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.device = device
        self.dtype = dtype

        self.tok_emb = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # Use Needle's Transformer (adds learned positional embedding internally)
        # Set dim_head = d_model // n_head; hidden_size ~ 4*d_model
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
        """
        x_ids: (B, T) integer-like Tensor (float32 but with int values)
        returns: logits (B, T, vocab_size)
        """
        # Convert to (T, B) for Embedding and Transformer
        x_tb = ops.transpose(x_ids, axes=(1, 0))  # (T, B)
        h_tbd = self.tok_emb(x_tb)  # (T, B, d_model)
        y_tbd, _ = self.transformer(h_tbd)  # (T, B, d_model)
        y_btd = ops.transpose(y_tbd, axes=(1, 0))  # (B, T, d_model)
        B, T, D = y_btd.shape
        y2d = ops.reshape(y_btd, (B * T, D))
        logits2d = self.lm_head(y2d)  # (B*T, V)
        logits = ops.reshape(logits2d, (B, T, self.vocab_size))
        return logits


# ============================================
# 4. Masked cross-entropy (ignore PAD)
# ============================================
def masked_cross_entropy(logits, targets, pad_id=PAD_ID):
    """
    logits: (B, T, V)
    targets: (B, T) integer-like Tensor
    returns scalar loss (average over non-pad tokens)
    """
    B, T, V = logits.shape
    logits2d = ops.reshape(logits, (B * T, V))
    targets2d = ops.reshape(targets, (B * T,))

    log_probs = ops.logsoftmax(logits2d)
    one_hot = ndl.init.one_hot(V, targets2d, device=logits.device, dtype=logits.dtype)
    nll = -ops.summation(log_probs * one_hot, axes=(1,))  # (B*T,)

    # mask out PAD positions
    tgt_np = targets2d.numpy().astype(np.int32)
    mask_np = (tgt_np != pad_id).astype(np.float32)
    mask = ndl.Tensor(mask_np, device=logits.device, dtype=logits.dtype, requires_grad=False)

    masked_nll = nll * mask
    denom = float(mask.numpy().sum()) + 1e-8
    loss = ops.summation(masked_nll) / denom
    return loss


# ============================================
# 5. Training & evaluation
# ============================================
def train_one_epoch(model, loader, opt, device=None):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for step, batch_np in enumerate(loader):
        # Prepare inputs/targets
        inp_np = batch_np[:, :-1].astype(np.float32)
        tgt_np = batch_np[:, 1:].astype(np.float32)
        inp = ndl.Tensor(inp_np, device=device, dtype="float32")
        tgt = ndl.Tensor(tgt_np, device=device, dtype="float32")

        logits = model(inp)
        loss = masked_cross_entropy(logits, tgt, pad_id=PAD_ID)

        opt.reset_grad()
        loss.backward()
        opt.step()

        # stats
        with np.errstate(over="ignore"):
            tgt_int = batch_np[:, 1:]
            non_pad = int((tgt_int != PAD_ID).sum())
        total_loss += float(loss.numpy()) * non_pad
        total_tokens += non_pad

        if (step + 1) % 50 == 0:
            print(f"  [step {step+1}] loss: {float(loss.numpy()):.4f}")

    avg = total_loss / max(total_tokens, 1)
    print(f"  epoch avg token loss: {avg:.4f}, ppl≈{math.exp(avg):.2f}")


def greedy_generate(model, prefix_ids, max_new_tokens, device=None):
    """
    prefix_ids: 1D numpy array of token ids (no padding)
    returns: 1D numpy array of prefix + generated tokens
    """
    model.eval()
    gen = prefix_ids.astype(np.float32)[None, :]  # (1, L)
    for _ in range(max_new_tokens):
        x = ndl.Tensor(gen, device=device, dtype="float32")
        logits = model(x)  # (1, L, V)
        last = logits.numpy()[0, -1]
        next_id = int(last.argmax())
        gen = np.concatenate([gen, np.array([[next_id]], dtype=np.float32)], axis=1)
    return gen.astype(np.int64).squeeze(0)


def evaluate_copy_accuracy(model, num_examples=200, max_seq_len=32, seed=999, device=None):
    ds = CopyDataset(num_examples=num_examples, max_seq_len=max_seq_len, seed=seed)
    token_correct = 0
    token_total = 0
    seq_correct = 0

    for i in range(len(ds)):
        seq = ds[i]  # <bos> base # base
        L = seq.shape[0]
        n = (L - 2) // 2
        prefix = seq[: n + 2]
        true_tail = seq[n + 2 :]

        gen = greedy_generate(model, prefix, max_new_tokens=n, device=device)
        gen_tail = gen[-n:]

        token_correct += int((gen_tail == true_tail).sum())
        token_total += n
        if np.array_equal(gen_tail, true_tail):
            seq_correct += 1

        if i < 3:
            print("\nExample", i)
            print("Full seq:      ", decode_ids(seq))
            print("Prefix given:  ", decode_ids(prefix))
            print("True tail:     ", decode_ids(true_tail))
            print("Generated tail:", decode_ids(gen_tail))

    print("\n=== Copy task evaluation ===")
    print(f"Token-level accuracy:   {token_correct / max(token_total,1):.3f}")
    print(f"Sequence-level accuracy:{seq_correct / max(len(ds),1):.3f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Needle copy task trainer")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-examples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-examples", type=int, default=200)
    args = parser.parse_args()

    print("Using Needle backend:", ndl.backend_selection.BACKEND)
    device = ndl.default_device()
    print("Device:", device)

    # Data
    train_ds = CopyDataset(num_examples=args.train_examples, max_seq_len=args.max_len, seed=0)
    train_loader = SimpleDataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Model
    model = GPTTinyNeedle(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        max_len=args.max_len,
        dropout=args.dropout,
        device=device,
        dtype="float32",
    )

    nparams = sum(np.prod(p.shape) for p in model.parameters())
    print(f"Model parameters: {nparams/1e3:.1f} K")

    opt = ndl.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_one_epoch(model, train_loader, opt, device=device)

    evaluate_copy_accuracy(model, num_examples=args.eval_examples, max_seq_len=args.max_len, device=device)


if __name__ == "__main__":
    main()
