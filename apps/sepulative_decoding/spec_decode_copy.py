"""
Speculative decoding demo for Needle on synthetic copy task.

We train a small draft model and a larger verifier model, then decode with
speculative decoding (Big-Little style): the draft proposes K tokens, and the
verifier checks them in parallel; we accept the longest matching prefix. We
report tokens/sec, acceptance rate, and latency p50.

Run examples:
  - Train both models and run speculative decoding on GPU (if available):
    python apps/spec_decode_copy.py --device auto --train-examples 20000 --epochs-draft 5 --epochs-verifier 10 --draft-k 8

  - CPU quick test:
    python apps/spec_decode_copy.py --device cpu --train-examples 5000 --epochs-draft 2 --epochs-verifier 4 --d-draft 64 --layers-draft 1 --d-verifier 128 --layers-verifier 2
"""

import sys
import os
import time
import statistics
import math
import numpy as np

sys.path.append("python/")
import needle as ndl
from needle import ops

# add this folder to path for local imports when run as a script
sys.path.append(os.path.dirname(__file__))
from common import (
    CopyDataset,
    Loader,
    GPTTinyNeedle,
    masked_cross_entropy,
    PAD_ID,
    BOS_ID,
    SEP_ID,
    VOCAB_SIZE,
)


def decode_ids(ids):
    itos = ["<pad>", "<bos>", "#"] + list("abcdef")
    return " ".join(itos[int(i)] for i in ids)


def train_epoch(model, loader, opt, device, print_every=50, micro_batch_size=0):
    model.train()
    for step, batch_np in enumerate(loader):
        B = batch_np.shape[0]
        if micro_batch_size and micro_batch_size < B:
            # process in micro-batches to reduce peak memory
            for s in range(0, B, micro_batch_size):
                mb = batch_np[s : s + micro_batch_size]
                inp = ndl.Tensor(mb[:, :-1].astype(np.float32), device=device)
                tgt = ndl.Tensor(mb[:, 1:].astype(np.float32), device=device)
                logits = model(inp)
                loss = masked_cross_entropy(logits, tgt)
                opt.reset_grad()
                loss.backward()
                opt.step()
        else:
            inp = ndl.Tensor(batch_np[:, :-1].astype(np.float32), device=device)
            tgt = ndl.Tensor(batch_np[:, 1:].astype(np.float32), device=device)
            logits = model(inp)
            loss = masked_cross_entropy(logits, tgt)
            opt.reset_grad()
            loss.backward()
            opt.step()

        if (step + 1) % print_every == 0:
            try:
                val = float(loss.numpy().item())
            except Exception:
                val = float(loss.numpy())
            print(f"  [step {step+1}] loss: {val:.4f}")


def greedy_next(model, prefix_ids_np, device):
    x = ndl.Tensor(prefix_ids_np.astype(np.float32)[None, :], device=device)
    logits = model(x)
    last = logits.numpy()[0, -1]
    return int(last.argmax())


def draft_propose_k(draft_model, prefix_np, k, device):
    seq = prefix_np.astype(np.int64)
    for _ in range(k):
        nxt = greedy_next(draft_model, seq, device)
        seq = np.concatenate([seq, np.array([nxt], dtype=np.int64)], axis=0)
    return seq[len(prefix_np) :]


def verify_chunk(verifier_model, prefix_np, drafted_np, device):
    # Build combined: prefix + drafted, single forward
    combined = np.concatenate([prefix_np, drafted_np], axis=0).astype(np.float32)[None, :]
    logits = verifier_model(ndl.Tensor(combined, device=device))
    logits_np = logits.numpy()[0]  # (T,V)
    T0 = len(prefix_np)
    accepts = 0
    verified_next = None
    for i, tok in enumerate(drafted_np):
        pos = T0 + i - 1  # next-token logits for token at T0+i use position T0+i-1
        if pos < 0 or pos >= logits_np.shape[0]:
            break
        pred = int(np.argmax(logits_np[pos]))
        if pred == int(tok):
            accepts += 1
        else:
            verified_next = pred
            break
    return accepts, verified_next


def speculative_generate(draft_model, verifier_model, prefix_np, total_tokens, k, device):
    generated = prefix_np.astype(np.int64)
    accepted = 0
    drafted_total = 0
    while total_tokens > 0:
        step_k = min(k, total_tokens)
        drafted = draft_propose_k(draft_model, generated, step_k, device)
        drafted_total += len(drafted)
        accepts, verified_next = verify_chunk(verifier_model, generated, drafted, device)
        if accepts == len(drafted):
            generated = np.concatenate([generated, drafted], axis=0)
            accepted += accepts
            total_tokens -= accepts
        else:
            # accept prefix of drafted and take verifier token at mismatch
            if accepts > 0:
                generated = np.concatenate([generated, drafted[:accepts]], axis=0)
                accepted += accepts
                total_tokens -= accepts
            # take verifier token as next
            if total_tokens > 0 and verified_next is not None:
                generated = np.concatenate([generated, np.array([verified_next], dtype=np.int64)], axis=0)
                total_tokens -= 1
            # else, stop if cannot proceed
    return generated, accepted, drafted_total


def evaluate_speculative(draft_model, verifier_model, num_examples=200, max_seq_len=32, k=8, device=None):
    ds = CopyDataset(num_examples=num_examples, max_seq_len=max_seq_len, seed=999)
    latencies = []
    total_gen_tokens = 0
    total_time = 0.0
    total_accepted = 0
    total_drafted = 0
    correct_tokens = 0
    total_tokens = 0
    for i in range(len(ds)):
        seq = ds[i]
        L = seq.shape[0]
        n = (L - 2) // 2
        prefix = seq[: n + 2]
        true_tail = seq[n + 2 :]
        t0 = time.time()
        gen, acc, drafted = speculative_generate(draft_model, verifier_model, prefix, n, k, device)
        dt = time.time() - t0
        latencies.append(dt)
        total_gen_tokens += n
        total_time += dt
        total_accepted += acc
        total_drafted += drafted
        gen_tail = gen[-n:]
        correct_tokens += int((gen_tail == true_tail).sum())
        total_tokens += n
    tokens_per_sec = total_gen_tokens / max(total_time, 1e-8)
    acceptance_rate = total_accepted / max(total_drafted, 1)
    p50 = statistics.median(latencies)
    acc = correct_tokens / max(total_tokens, 1)
    return {
        "tokens_per_sec": tokens_per_sec,
        "acceptance_rate": acceptance_rate,
        "latency_p50": p50,
        "token_accuracy": acc,
    }


def greedy_generate_verifier(model, prefix_np, total_tokens, device):
    gen = prefix_np.astype(np.int64)
    for _ in range(total_tokens):
        nxt = greedy_next(model, gen, device)
        gen = np.concatenate([gen, np.array([nxt], dtype=np.int64)], axis=0)
    return gen


def evaluate_baseline(verifier_model, num_examples=200, max_seq_len=32, device=None):
    ds = CopyDataset(num_examples=num_examples, max_seq_len=max_seq_len, seed=1001)
    latencies = []
    total_gen_tokens = 0
    total_time = 0.0
    correct_tokens = 0
    total_tokens = 0
    for i in range(len(ds)):
        seq = ds[i]
        L = seq.shape[0]
        n = (L - 2) // 2
        prefix = seq[: n + 2]
        true_tail = seq[n + 2 :]
        t0 = time.time()
        gen = greedy_generate_verifier(verifier_model, prefix, n, device)
        dt = time.time() - t0
        latencies.append(dt)
        total_gen_tokens += n
        total_time += dt
        gen_tail = gen[-n:]
        correct_tokens += int((gen_tail == true_tail).sum())
        total_tokens += n
    tokens_per_sec = total_gen_tokens / max(total_time, 1e-8)
    p50 = statistics.median(latencies)
    acc = correct_tokens / max(total_tokens, 1)
    return {
        "tokens_per_sec": tokens_per_sec,
        "latency_p50": p50,
        "token_accuracy": acc,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Speculative decoding on copy task")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--train-examples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=0, help="Optional micro-batch size to reduce peak memory (0=disabled)")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--epochs-draft", type=int, default=5)
    parser.add_argument("--epochs-verifier", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    # make draft bigger and verifier even bigger by default for higher acceptance
    parser.add_argument("--d-draft", type=int, default=128)
    parser.add_argument("--heads-draft", type=int, default=4)
    parser.add_argument("--layers-draft", type=int, default=2)
    parser.add_argument("--d-verifier", type=int, default=256)
    parser.add_argument("--heads-verifier", type=int, default=4)
    parser.add_argument("--layers-verifier", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--draft-k", type=int, default=8)
    parser.add_argument("--eval-examples", type=int, default=200)
    args = parser.parse_args()

    print("Using Needle backend:", ndl.backend_selection.BACKEND)
    if args.device == "cpu":
        device = ndl.cpu()
    elif args.device == "cuda":
        device = ndl.cuda() if ndl.cuda().enabled() else ndl.cpu()
        if device.__repr__() == "cpu()":
            print("CUDA requested but not available; falling back to CPU.")
    else:
        device = ndl.cuda() if ndl.cuda().enabled() else ndl.cpu()
    print("Device:", device)

    # Data
    train_ds = CopyDataset(args.train_examples, max_seq_len=args.max_len, seed=0)
    loader = Loader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Models
    draft = GPTTinyNeedle(VOCAB_SIZE, d_model=args.d_draft, n_head=args.heads_draft, n_layer=args.layers_draft, max_len=args.max_len, dropout=args.dropout, device=device)
    verifier = GPTTinyNeedle(VOCAB_SIZE, d_model=args.d_verifier, n_head=args.heads_verifier, n_layer=args.layers_verifier, max_len=args.max_len, dropout=args.dropout, device=device)

    print(f"Draft params: {sum(np.prod(p.shape) for p in draft.parameters())/1e3:.1f} K  | Verifier params: {sum(np.prod(p.shape) for p in verifier.parameters())/1e3:.1f} K")

    # Train draft
    opt_d = ndl.optim.Adam(draft.parameters(), lr=args.lr)
    for e in range(args.epochs_draft):
        t0 = time.time()
        train_epoch(draft, loader, opt_d, device, print_every=args.print_every, micro_batch_size=args.micro_batch_size)
        print(f"Draft epoch {e+1}/{args.epochs_draft}  time={time.time()-t0:.1f}s")

    # Train verifier
    opt_v = ndl.optim.Adam(verifier.parameters(), lr=args.lr)
    for e in range(args.epochs_verifier):
        t0 = time.time()
        train_epoch(verifier, loader, opt_v, device, print_every=args.print_every, micro_batch_size=args.micro_batch_size)
        print(f"Verifier epoch {e+1}/{args.epochs_verifier}  time={time.time()-t0:.1f}s")

    # Baseline (verifier-only greedy) and speculative decoding
    print("\nEvaluating baseline (verifier-only)...")
    base = evaluate_baseline(verifier, num_examples=args.eval_examples, max_seq_len=args.max_len, device=device)
    print(f"baseline tokens/sec: {base['tokens_per_sec']:.2f}")
    print(f"baseline latency p50: {base['latency_p50']:.4f} s")
    print(f"baseline token accuracy: {base['token_accuracy']:.3f}")

    print("\nEvaluating speculative decoding...")
    spec = evaluate_speculative(draft, verifier, num_examples=args.eval_examples, max_seq_len=args.max_len, k=args.draft_k, device=device)
    print(f"spec tokens/sec: {spec['tokens_per_sec']:.2f}")
    print(f"acceptance rate: {spec['acceptance_rate']:.2f}")
    print(f"spec latency p50: {spec['latency_p50']:.4f} s")
    print(f"spec token accuracy: {spec['token_accuracy']:.3f}")

    speedup = spec['tokens_per_sec'] / max(base['tokens_per_sec'], 1e-8)
    print(f"\nSpeedup (spec vs baseline): {speedup:.2f}x")


if __name__ == "__main__":
    main()
