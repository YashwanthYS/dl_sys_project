"""
Analysis utilities for speculative decoding on the synthetic copy task.

Features:
- Train draft and verifier models; record per-epoch loss curves.
- Sweep K (draft proposal length) and measure baseline vs. speculative tokens/sec.
- Plot loss curves and speedup vs. K.

Example:
  python apps/sepulative_decoding/analysis.py \
    --device auto \
    --train-examples 20000 \
    --epochs-draft 5 --epochs-verifier 10 \
    --d-draft 128 --layers-draft 2 --d-verifier 256 --layers-verifier 3 \
    --k-values 2,4,6,8,12,16 \
    --output-dir analysis_out
"""

import os
import sys
import time
import math
import statistics
import numpy as np

import matplotlib.pyplot as plt

sys.path.append("python/")
import needle as ndl
from needle import ops

# local imports
sys.path.append(os.path.dirname(__file__))
from common import (
    CopyDataset,
    Loader,
    GPTTinyNeedle,
    masked_cross_entropy,
    PAD_ID,
    VOCAB_SIZE,
)


def train_model(model, loader, opt, device, epochs=1, print_every=50, micro_batch_size=0):
    """Train and return list of per-epoch average token loss."""
    loss_curve = []
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()
        for step, batch_np in enumerate(loader):
            B = batch_np.shape[0]
            if micro_batch_size and micro_batch_size < B:
                for s in range(0, B, micro_batch_size):
                    mb = batch_np[s : s + micro_batch_size]
                    inp = ndl.Tensor(mb[:, :-1].astype(np.float32), device=device)
                    tgt = ndl.Tensor(mb[:, 1:].astype(np.float32), device=device)
                    logits = model(inp)
                    loss = masked_cross_entropy(logits, tgt)
                    opt.reset_grad()
                    loss.backward()
                    opt.step()

                    non_pad = int((mb[:, 1:] != PAD_ID).sum())
                    total_loss += float(loss.numpy().item()) * non_pad
                    total_tokens += non_pad
            else:
                inp = ndl.Tensor(batch_np[:, :-1].astype(np.float32), device=device)
                tgt = ndl.Tensor(batch_np[:, 1:].astype(np.float32), device=device)
                logits = model(inp)
                loss = masked_cross_entropy(logits, tgt)
                opt.reset_grad()
                loss.backward()
                opt.step()

                non_pad = int((batch_np[:, 1:] != PAD_ID).sum())
                total_loss += float(loss.numpy().item()) * non_pad
                total_tokens += non_pad

            if (step + 1) % print_every == 0:
                try:
                    val = float(loss.numpy().item())
                except Exception:
                    val = float(loss.numpy())
                print(f"  [epoch {ep+1}/{epochs} step {step+1}] loss: {val:.4f}")

        avg = total_loss / max(total_tokens, 1)
        loss_curve.append(avg)
        print(f"Epoch {ep+1}/{epochs} done in {time.time()-t0:.1f}s. avg token loss: {avg:.4f}, ppl≈{math.exp(avg):.2f}")
    return loss_curve


def greedy_next(model, prefix_ids_np, device):
    x = ndl.Tensor(prefix_ids_np.astype(np.float32)[None, :], device=device)
    logits = model(x)
    last = logits.numpy()[0, -1]
    return int(last.argmax())


def greedy_generate_verifier(model, prefix_np, total_tokens, device):
    gen = prefix_np.astype(np.int64)
    for _ in range(total_tokens):
        nxt = greedy_next(model, gen, device)
        gen = np.concatenate([gen, np.array([nxt], dtype=np.int64)], axis=0)
    return gen


def evaluate_baseline(verifier_model, num_examples=200, max_seq_len=32, device=None):
    ds = CopyDataset(num_examples=num_examples, max_seq_len=max_seq_len, seed=1001)
    latencies, total_gen_tokens, total_time = [], 0, 0.0
    correct, total = 0, 0
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
        correct += int((gen_tail == true_tail).sum())
        total += n
    return {
        "tokens_per_sec": total_gen_tokens / max(total_time, 1e-8),
        "latency_p50": statistics.median(latencies),
        "token_accuracy": correct / max(total, 1),
    }


def draft_propose_k(draft_model, prefix_np, k, device):
    seq = prefix_np.astype(np.int64)
    for _ in range(k):
        nxt = greedy_next(draft_model, seq, device)
        seq = np.concatenate([seq, np.array([nxt], dtype=np.int64)], axis=0)
    return seq[len(prefix_np) :]


def verify_chunk(verifier_model, prefix_np, drafted_np, device):
    combined = np.concatenate([prefix_np, drafted_np], axis=0).astype(np.float32)[None, :]
    logits = verifier_model(ndl.Tensor(combined, device=device))
    logits_np = logits.numpy()[0]
    T0 = len(prefix_np)
    accepts = 0
    verified_next = None
    for i, tok in enumerate(drafted_np):
        pos = T0 + i - 1
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
            if accepts > 0:
                generated = np.concatenate([generated, drafted[:accepts]], axis=0)
                accepted += accepts
                total_tokens -= accepts
            if total_tokens > 0 and verified_next is not None:
                generated = np.concatenate([generated, np.array([verified_next], dtype=np.int64)], axis=0)
                total_tokens -= 1
    return generated, accepted, drafted_total


def evaluate_speculative(draft_model, verifier_model, num_examples=200, max_seq_len=32, k=8, device=None):
    ds = CopyDataset(num_examples=num_examples, max_seq_len=max_seq_len, seed=999)
    latencies, total_gen_tokens, total_time = [], 0, 0.0
    total_accepted, total_drafted = 0, 0
    correct, total = 0, 0
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
        correct += int((gen_tail == true_tail).sum())
        total += n
    return {
        "tokens_per_sec": total_gen_tokens / max(total_time, 1e-8),
        "acceptance_rate": total_accepted / max(total_drafted, 1),
        "latency_p50": statistics.median(latencies),
        "token_accuracy": correct / max(total, 1),
    }


def speculative_generate_timed(draft_model, verifier_model, prefix_np, total_tokens, k, device):
    """Speculative generate with timing breakdown per step."""
    generated = prefix_np.astype(np.int64)
    accepted = 0
    drafted_total = 0
    t_draft = 0.0
    t_verify = 0.0
    while total_tokens > 0:
        step_k = min(k, total_tokens)
        t0 = time.time()
        drafted = draft_propose_k(draft_model, generated, step_k, device)
        t_draft += time.time() - t0

        drafted_total += len(drafted)
        t1 = time.time()
        accepts, verified_next = verify_chunk(verifier_model, generated, drafted, device)
        t_verify += time.time() - t1

        if accepts == len(drafted):
            generated = np.concatenate([generated, drafted], axis=0)
            accepted += accepts
            total_tokens -= accepts
        else:
            if accepts > 0:
                generated = np.concatenate([generated, drafted[:accepts]], axis=0)
                accepted += accepts
                total_tokens -= accepts
            if total_tokens > 0 and verified_next is not None:
                generated = np.concatenate([generated, np.array([verified_next], dtype=np.int64)], axis=0)
                total_tokens -= 1
    return generated, accepted, drafted_total, t_draft, t_verify


def evaluate_speculative_timing(draft_model, verifier_model, num_examples=200, max_seq_len=32, k=8, device=None):
    ds = CopyDataset(num_examples=num_examples, max_seq_len=max_seq_len, seed=999)
    total_time = 0.0
    total_draft = 0.0
    total_verify = 0.0
    total_accepted = 0
    total_drafted = 0
    total_gen_tokens = 0
    for i in range(len(ds)):
        seq = ds[i]
        L = seq.shape[0]
        n = (L - 2) // 2
        prefix = seq[: n + 2]
        t0 = time.time()
        gen, acc, drafted, t_d, t_v = speculative_generate_timed(draft_model, verifier_model, prefix, n, k, device)
        dt = time.time() - t0
        total_time += dt
        total_draft += t_d
        total_verify += t_v
        total_accepted += acc
        total_drafted += drafted
        total_gen_tokens += n

    timing = {
        "total_time": total_time,
        "t_draft": total_draft,
        "t_verify": total_verify,
        "t_overhead": max(total_time - total_draft - total_verify, 0.0),
        "tokens_per_sec": total_gen_tokens / max(total_time, 1e-8),
        "acceptance_rate": total_accepted / max(total_drafted, 1),
    }
    return timing


def plot_loss_curves(draft_curve, verifier_curve, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(draft_curve)+1), draft_curve, label="draft")
    plt.plot(range(1, len(verifier_curve)+1), verifier_curve, label="verifier")
    plt.xlabel("Epoch")
    plt.ylabel("Avg token loss")
    plt.title("Training loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_speedup_vs_k(ks, speedups, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(ks, speedups, marker="o")
    plt.xlabel("K (draft tokens)")
    plt.ylabel("Speedup (spec/baseline)")
    plt.title("Speculative decoding: speedup vs K")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_timing_breakdown(ks, drafts, verifies, overheads, out_path):
    plt.figure(figsize=(7,4))
    ind = np.arange(len(ks))
    width = 0.6
    p1 = plt.bar(ind, drafts, width, label='draft')
    p2 = plt.bar(ind, verifies, width, bottom=drafts, label='verify')
    bottom2 = [d + v for d, v in zip(drafts, verifies)]
    p3 = plt.bar(ind, overheads, width, bottom=bottom2, label='overhead')
    plt.xticks(ind, ks)
    plt.xlabel('K (draft tokens)')
    plt.ylabel('Total time (s)')
    plt.title('Speculative decoding timing breakdown')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Speculative decoding analysis")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--train-examples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=0)
    parser.add_argument("--epochs-draft", type=int, default=5)
    parser.add_argument("--epochs-verifier", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-draft", type=int, default=128)
    parser.add_argument("--heads-draft", type=int, default=4)
    parser.add_argument("--layers-draft", type=int, default=2)
    parser.add_argument("--d-verifier", type=int, default=256)
    parser.add_argument("--heads-verifier", type=int, default=4)
    parser.add_argument("--layers-verifier", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--k-values", type=str, default="2,4,6,8,12,16")
    parser.add_argument("--eval-examples", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="analysis_out")
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

    os.makedirs(args.output_dir, exist_ok=True)

    # Data and loaders
    train_ds = CopyDataset(args.train_examples, max_seq_len=args.max_len, seed=0)
    loader = Loader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Models
    draft = GPTTinyNeedle(VOCAB_SIZE, d_model=args.d_draft, n_head=args.heads_draft, n_layer=args.layers_draft, max_len=args.max_len, dropout=args.dropout, device=device)
    verifier = GPTTinyNeedle(VOCAB_SIZE, d_model=args.d_verifier, n_head=args.heads_verifier, n_layer=args.layers_verifier, max_len=args.max_len, dropout=args.dropout, device=device)

    print(f"Draft params: {sum(np.prod(p.shape) for p in draft.parameters())/1e3:.1f} K | Verifier params: {sum(np.prod(p.shape) for p in verifier.parameters())/1e3:.1f} K")

    # Train and record loss curves
    opt_d = ndl.optim.Adam(draft.parameters(), lr=args.lr)
    draft_curve = train_model(draft, loader, opt_d, device, epochs=args.epochs_draft, micro_batch_size=args.micro_batch_size)

    opt_v = ndl.optim.Adam(verifier.parameters(), lr=args.lr)
    verifier_curve = train_model(verifier, loader, opt_v, device, epochs=args.epochs_verifier, micro_batch_size=args.micro_batch_size)

    # Save loss curves plot
    loss_plot = os.path.join(args.output_dir, "loss_curves.png")
    plot_loss_curves(draft_curve, verifier_curve, loss_plot)
    print("Saved:", loss_plot)

    # Baseline
    base = evaluate_baseline(verifier, num_examples=args.eval_examples, max_seq_len=args.max_len, device=device)
    print(f"Baseline tokens/sec: {base['tokens_per_sec']:.2f}, p50: {base['latency_p50']:.4f}s, acc: {base['token_accuracy']:.3f}")

    # Sweep over K
    ks = [int(x) for x in args.k_values.split(',')]

    speedups = []
    t_drafts, t_verifies, t_overheads = [], [], []
    for k in ks:
        spec = evaluate_speculative(draft, verifier, num_examples=args.eval_examples, max_seq_len=args.max_len, k=k, device=device)
        sp = spec['tokens_per_sec'] / max(base['tokens_per_sec'], 1e-8)
        speedups.append(sp)
        print(f"K={k:>2d}  spec t/s={spec['tokens_per_sec']:.2f}  speedup={sp:.2f}  acc_rate={spec['acceptance_rate']:.2f}")

        # timing breakdown
        tb = evaluate_speculative_timing(draft, verifier, num_examples=args.eval_examples, max_seq_len=args.max_len, k=k, device=device)
        t_drafts.append(tb['t_draft'])
        t_verifies.append(tb['t_verify'])
        t_overheads.append(tb['t_overhead'])

    # Save speedup plot
    sp_plot = os.path.join(args.output_dir, "speedup_vs_k.png")
    plot_speedup_vs_k(ks, speedups, sp_plot)
    print("Saved:", sp_plot)

    tb_plot = os.path.join(args.output_dir, "timing_breakdown.png")
    plot_timing_breakdown(ks, t_drafts, t_verifies, t_overheads, tb_plot)
    print("Saved:", tb_plot)


if __name__ == "__main__":
    main()
