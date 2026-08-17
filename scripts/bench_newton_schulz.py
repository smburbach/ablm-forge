"""Benchmark Muon's Newton-Schulz orthogonalization across the preset ladder.

Answers "when does the orthogonalizer start to matter?" -- the question behind any
decision to swap it (Gram Newton-Schulz, a fused kernel, an external optimizer). NS
cost grows cubically in width while the rest of the step grows quadratically, so the
share it takes is a function of scale, and the crossover is worth measuring rather
than assuming: Dion3 (arXiv:2608.11612) reports NS taking anywhere from 2% to 17% of
end-to-end training time depending on setup.

Reports per-shape timings and the per-optimizer-step total for each preset, split by
attention (square) and MLP (rectangular) weights, since alternatives like Gram
Newton-Schulz only help on rectangular matrices and fall back to standard NS when
m == n.

    python scripts/bench_newton_schulz.py                 # whole ladder, bf16
    python scripts/bench_newton_schulz.py --world-size 2  # per-rank, sharded over DDP
"""

from __future__ import annotations

import argparse

import torch
from torch.optim._muon import DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_NS_STEPS, EPS
from torch.optim._muon import _zeropower_via_newtonschulz as zeropower

from ablm import PRESETS, from_preset

_COEFFS = (DEFAULT_A, DEFAULT_B, DEFAULT_C)


def time_ns(rows: int, cols: int, dtype: torch.dtype, device: str, iters: int) -> float:
    """Median ms for one Newton-Schulz call on a rows x cols matrix."""
    x = torch.randn(rows, cols, device=device, dtype=dtype)
    for _ in range(3):
        zeropower(x, _COEFFS, DEFAULT_NS_STEPS, EPS)
    times = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        zeropower(x, _COEFFS, DEFAULT_NS_STEPS, EPS)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--world-size", type=int, default=1, help="divide totals across DDP ranks")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU: Newton-Schulz timings on CPU do not transfer")
    dtype = getattr(torch, args.dtype)
    print(f"{torch.cuda.get_device_name(0)} | {args.dtype} | ns_steps={DEFAULT_NS_STEPS}")
    print(f"{'preset':>7} {'hidden':>7} {'layers':>7} {'attn ms':>9} {'mlp ms':>9} {'step ms':>9}")

    for name in PRESETS:
        cfg = from_preset(name)
        h, ffn, layers = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
        attn = time_ns(h, h, dtype, "cuda", args.iters)
        mlp = time_ns(h, ffn, dtype, "cuda", args.iters)
        # per layer: 4 square attention projections, 3 rectangular SwiGLU matrices
        step = layers * (4 * attn + 3 * mlp) / args.world_size
        print(f"{name:>7} {h:7} {layers:7} {attn:9.3f} {mlp:9.3f} {step:9.1f}")

    print(f"\nstep ms = per optimizer step, all 2D body weights, over {args.world_size} rank(s).")
    print("attn is square (alpha=1): Gram Newton-Schulz falls back to standard NS there.")


if __name__ == "__main__":
    main()
