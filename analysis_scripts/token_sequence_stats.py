"""
Token Sequence Statistics for OATTok Tokenizer Comparison.

Computes codebook utilization, transition smoothness, consecutive action distances,
entropy rate, and per-token reconstruction error across FSQ / A2Lex / ZHill tokenizers.

Usage:
    python analysis_scripts/token_sequence_stats.py \
        --tokenizer_dirs output/.../fsq,output/.../a2lex,output/.../zhill \
        --tokenizer_names FSQ,A2Lex,ZHill \
        --data_dir data/libero/libero10_N500.zarr \
        --output_dir ./analysis_results
"""

import argparse
import json
import os
import sys
import pathlib
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from tqdm import tqdm

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from oat.tokenizer.base_tokenizer import BaseTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tokenizer(ckpt_dir: str, device: torch.device):
    """Load tokenizer from a training output directory or a direct .ckpt path."""
    if ckpt_dir.endswith(".ckpt") and os.path.isfile(ckpt_dir):
        best = ckpt_dir
    else:
        ckpt_path = os.path.join(ckpt_dir, "checkpoints")
        ckpts = sorted(
            [f for f in os.listdir(ckpt_path) if f.startswith("ep-") and f.endswith(".ckpt")],
            key=lambda f: float(f.split("mse-")[1].replace(".ckpt", ""))
        )
        if not ckpts:
            raise FileNotFoundError(f"No ep-*.ckpt found in {ckpt_path}")
        best = os.path.join(ckpt_path, ckpts[0])
    print(f"  Loading checkpoint: {best}")
    tok = BaseTokenizer.from_checkpoint(best)
    tok.eval()
    tok.to(device)
    return tok


def extract_action_chunks(zarr_path: str, horizon: int = 32, stride: int = 1):
    """Extract action chunks from the Zarr dataset.

    Returns:
        actions: np.ndarray of shape (N, horizon, action_dim)
        episode_ids: np.ndarray of shape (N,) — which episode each chunk belongs to
    """
    import zarr
    z = zarr.open(zarr_path, "r")
    all_actions = z["data/action"][:]          # (T_total, 7)
    episode_ends = z["meta/episode_ends"][:]   # (num_episodes,)

    chunks = []
    ep_ids = []
    starts = np.concatenate([[0], episode_ends[:-1]])
    for ep_idx, (s, e) in enumerate(zip(starts, episode_ends)):
        ep_actions = all_actions[s:e]  # (T_ep, 7)
        for t in range(0, len(ep_actions) - horizon + 1, stride):
            chunks.append(ep_actions[t : t + horizon])
            ep_ids.append(ep_idx)
    return np.stack(chunks).astype(np.float32), np.array(ep_ids)


@torch.inference_mode()
def tokenize_all(tokenizer, actions: np.ndarray, device: torch.device, batch_size: int = 512):
    """Tokenize all action chunks. Returns token indices (N, H_l)."""
    N = len(actions)
    all_tokens = []
    for i in tqdm(range(0, N, batch_size), desc="    tokenizing"):
        batch = torch.from_numpy(actions[i : i + batch_size]).to(device)
        tokens = tokenizer.tokenize(batch)  # (B, H_l)
        all_tokens.append(tokens.cpu())
    return torch.cat(all_tokens, dim=0)  # (N, H_l)


@torch.inference_mode()
def detokenize_all(tokenizer, tokens: torch.Tensor, device: torch.device, batch_size: int = 512):
    """Detokenize all tokens back to actions. Returns np.ndarray (N, H_a, action_dim)."""
    N = len(tokens)
    all_recon = []
    for i in range(0, N, batch_size):
        batch = tokens[i : i + batch_size].to(device)
        recon = tokenizer.detokenize(batch)  # (B, H_a, action_dim)
        all_recon.append(recon.cpu().numpy())
    return np.concatenate(all_recon, axis=0)


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def compute_codebook_utilization(tokens: torch.Tensor, K: int):
    """Metric 1: codebook utilization statistics."""
    flat = tokens.reshape(-1).numpy()
    unique = np.unique(flat)
    num_unique = len(unique)
    util_rate = num_unique / K

    counts = np.bincount(flat, minlength=K).astype(np.float64)
    probs = counts / counts.sum()
    probs_nz = probs[probs > 0]
    entropy = -np.sum(probs_nz * np.log2(probs_nz))
    effective_size = 2 ** entropy

    return {
        "num_unique": int(num_unique),
        "K": int(K),
        "utilization_rate": float(util_rate),
        "frequency_entropy_bits": float(entropy),
        "effective_codebook_size": float(effective_size),
        "counts": counts,  # for plotting
    }


def compute_transition_smoothness(tokens: torch.Tensor, tokenizer, device: torch.device, episode_ids: np.ndarray):
    """Metric 2: transition smoothness on real demo consecutive pairs.

    For consecutive action chunks (t, t+1) within the same episode, compute:
      - |v_t - v_{t+1}| per latent position (token index difference)
      - Euclidean action-space distance between decoded action chunks
      - Spearman correlation between index diff and action distance
    """
    N, H_l = tokens.shape

    # Build consecutive pairs within same episode
    mask = episode_ids[:-1] == episode_ids[1:]
    idx_curr = np.where(mask)[0]
    idx_next = idx_curr + 1

    if len(idx_curr) == 0:
        return {"spearman_r": float("nan"), "spearman_p": float("nan"),
                "mean_index_diff": float("nan"), "mean_action_dist": float("nan")}

    tok_curr = tokens[idx_curr]  # (P, H_l)
    tok_next = tokens[idx_next]

    # Per-position index difference
    index_diff = (tok_curr.long() - tok_next.long()).abs().float()  # (P, H_l)

    # Action-space distance: detokenize full chunks and compute Euclidean distance
    recon_curr = detokenize_all(tokenizer, tok_curr, device)  # (P, H_a, 7)
    recon_next = detokenize_all(tokenizer, tok_next, device)
    action_dist = np.sqrt(((recon_curr - recon_next) ** 2).sum(axis=(1, 2)))  # (P,)

    # Aggregate index diff across positions
    mean_index_diff_per_pair = index_diff.mean(dim=1).numpy()  # (P,)

    # Spearman correlation
    rho, pval = scipy_stats.spearmanr(mean_index_diff_per_pair, action_dist)

    return {
        "spearman_r": float(rho),
        "spearman_p": float(pval),
        "mean_index_diff": float(mean_index_diff_per_pair.mean()),
        "std_index_diff": float(mean_index_diff_per_pair.std()),
        "mean_action_dist": float(action_dist.mean()),
        "std_action_dist": float(action_dist.std()),
        "index_diffs": mean_index_diff_per_pair,  # for plotting
        "action_dists": action_dist,
    }


def compute_consecutive_action_distances(tokens: torch.Tensor, tokenizer, device: torch.device, episode_ids: np.ndarray):
    """Metric 3: distribution of Euclidean action distances between consecutive token sequences.

    Computes both latent-space embedding distance (per position) and full action-space distance.
    """
    N, H_l = tokens.shape
    quantizer = tokenizer.quantizer

    mask = episode_ids[:-1] == episode_ids[1:]
    idx_curr = np.where(mask)[0]
    idx_next = idx_curr + 1

    tok_curr = tokens[idx_curr]
    tok_next = tokens[idx_next]

    # Per-position: codebook embedding distance (latent space)
    emb_curr = quantizer.indices_to_embedding(tok_curr.to(device))  # (P, H_l, dim)
    emb_next = quantizer.indices_to_embedding(tok_next.to(device))
    latent_dist_per_pos = (emb_curr - emb_next).pow(2).sum(dim=-1).sqrt().cpu().numpy()  # (P, H_l)
    latent_dist_flat = latent_dist_per_pos.ravel()

    # Full chunk action distance
    recon_curr = detokenize_all(tokenizer, tok_curr, device)
    recon_next = detokenize_all(tokenizer, tok_next, device)
    action_dist = np.sqrt(((recon_curr - recon_next) ** 2).sum(axis=(1, 2)))

    thresholds = [1.0, 2.0, 3.0, 5.0]
    frac_large = {f"frac_gt_{t}": float((action_dist > t).mean()) for t in thresholds}

    return {
        "mean": float(action_dist.mean()),
        "median": float(np.median(action_dist)),
        "p95": float(np.percentile(action_dist, 95)),
        "p99": float(np.percentile(action_dist, 99)),
        "max": float(action_dist.max()),
        **frac_large,
        "action_dists": action_dist,
        "latent_dists_flat": latent_dist_flat,
    }


def compute_entropy_rate(tokens: torch.Tensor, K: int, episode_ids: np.ndarray):
    """Metric 4: bigram conditional entropy per latent position."""
    N, H_l = tokens.shape
    tokens_np = tokens.numpy()

    mask = episode_ids[:-1] == episode_ids[1:]
    idx_curr = np.where(mask)[0]
    idx_next = idx_curr + 1

    entropies = []
    for h in range(H_l):
        bigram = np.zeros((K, K), dtype=np.float64)
        curr_h = tokens_np[idx_curr, h]
        next_h = tokens_np[idx_next, h]
        np.add.at(bigram, (curr_h, next_h), 1.0)

        # Conditional entropy H(next | curr) = sum_v P(v) * H(next | curr=v)
        row_sums = bigram.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        cond_probs = bigram / row_sums
        marginal = row_sums.ravel() / row_sums.sum()

        h_cond = 0.0
        for v in range(K):
            if marginal[v] < 1e-12:
                continue
            probs_v = cond_probs[v]
            probs_v = probs_v[probs_v > 0]
            h_cond += marginal[v] * (-np.sum(probs_v * np.log2(probs_v)))
        entropies.append(h_cond)

    return {
        "per_position_entropy": [float(e) for e in entropies],
        "mean_entropy": float(np.mean(entropies)),
    }


def compute_per_token_recon_error(tokens: torch.Tensor, actions: np.ndarray, tokenizer, device: torch.device, K: int):
    """Metric 5: per-token reconstruction error distribution.

    For each token appearing in the dataset, compute the MSE of the full action chunks
    that contain it (averaged over token positions).
    """
    N, H_l = tokens.shape
    tokens_np = tokens.numpy()

    recon = detokenize_all(tokenizer, tokens, device)  # (N, H_a, 7)
    per_chunk_mse = ((recon - actions) ** 2).mean(axis=(1, 2))  # (N,)

    # For each token, collect MSE of chunks containing it
    token_mses = defaultdict(list)
    for i in range(N):
        mse_i = per_chunk_mse[i]
        for h in range(H_l):
            token_mses[int(tokens_np[i, h])].append(mse_i)

    per_token_mean_mse = {}
    for t_id, mses in token_mses.items():
        per_token_mean_mse[t_id] = float(np.mean(mses))

    all_mse_vals = list(per_token_mean_mse.values())
    return {
        "num_tokens_with_data": len(per_token_mean_mse),
        "mean_per_token_mse": float(np.mean(all_mse_vals)) if all_mse_vals else float("nan"),
        "std_per_token_mse": float(np.std(all_mse_vals)) if all_mse_vals else float("nan"),
        "max_per_token_mse": float(np.max(all_mse_vals)) if all_mse_vals else float("nan"),
        "per_token_mse_dict": per_token_mean_mse,
        "per_chunk_mse": per_chunk_mse,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_codebook_utilization(results: dict, output_dir: str):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4), squeeze=False)
    for ax, (name, r) in zip(axes[0], results.items()):
        counts = r["codebook"]["counts"]
        ax.bar(range(len(counts)), counts, width=1.0, color="steelblue", alpha=0.7)
        ax.set_title(f"{name}\nUnique={r['codebook']['num_unique']}/{r['codebook']['K']}  "
                     f"Eff={r['codebook']['effective_codebook_size']:.0f}")
        ax.set_xlabel("Token index")
        ax.set_ylabel("Frequency")
        ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "codebook_utilization.png"), dpi=150)
    plt.close()


def plot_consecutive_action_distances(results: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name, r in results.items():
        dists = r["consecutive"]["action_dists"]
        ax.hist(dists, bins=100, alpha=0.5, label=name, density=True)
    ax.set_xlabel("Action-space Euclidean distance")
    ax.set_ylabel("Density")
    ax.set_title("Consecutive Chunk Action Distance Distribution")
    ax.legend()

    ax = axes[1]
    for name, r in results.items():
        dists = np.sort(r["consecutive"]["action_dists"])
        cdf = np.arange(1, len(dists) + 1) / len(dists)
        ax.plot(dists, cdf, label=name)
    ax.set_xlabel("Action-space Euclidean distance")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Consecutive Action Distances")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "consecutive_action_distances.png"), dpi=150)
    plt.close()


def plot_transition_smoothness(results: dict, output_dir: str):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
    for ax, (name, r) in zip(axes[0], results.items()):
        tr = r["transition"]
        if "index_diffs" not in tr:
            continue
        ax.scatter(tr["index_diffs"], tr["action_dists"], alpha=0.02, s=1, rasterized=True)
        ax.set_xlabel("Mean |token index diff|")
        ax.set_ylabel("Action-space distance")
        ax.set_title(f"{name}  ρ={tr['spearman_r']:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transition_smoothness.png"), dpi=150)
    plt.close()


def plot_per_token_recon_error(results: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        mse_vals = sorted(r["recon_error"]["per_token_mse_dict"].values())
        ax.plot(mse_vals, label=f"{name} ({r['recon_error']['num_tokens_with_data']} tokens)")
    ax.set_xlabel("Token rank (sorted by MSE)")
    ax.set_ylabel("Mean reconstruction MSE")
    ax.set_title("Per-Token Reconstruction Error (sorted)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_token_recon_error.png"), dpi=150)
    plt.close()


def plot_entropy_rate(results: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = None
    width = 0.8 / len(results)
    for i, (name, r) in enumerate(results.items()):
        ent = r["entropy"]["per_position_entropy"]
        positions = np.arange(len(ent))
        if x is None:
            x = positions
        ax.bar(x + i * width, ent, width=width, label=name, alpha=0.8)
    ax.set_xlabel("Latent position h")
    ax.set_ylabel("Conditional entropy (bits)")
    ax.set_title("Bigram Conditional Entropy per Latent Position")
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(len(x))])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "entropy_rate.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict):
    names = list(results.keys())
    sep = "-" * 90
    print(f"\n{'='*90}")
    print("TOKEN SEQUENCE STATISTICS — SUMMARY COMPARISON")
    print(f"{'='*90}\n")

    header = f"{'':20s}" + "".join(f"{n:>20s}" for n in names)

    # Codebook
    print("METRIC 1: Codebook Utilization")
    print(sep)
    print(header)
    for key in ["num_unique", "utilization_rate", "frequency_entropy_bits", "effective_codebook_size"]:
        row = f"{key:20s}"
        for n in names:
            v = results[n]["codebook"][key]
            if isinstance(v, float):
                row += f"{v:20.3f}"
            else:
                row += f"{v:20d}"
        print(row)
    print()

    # Transition smoothness
    print("METRIC 2: Transition Smoothness (Spearman ρ: index diff vs action dist)")
    print(sep)
    print(header)
    for key in ["spearman_r", "spearman_p", "mean_index_diff", "mean_action_dist"]:
        row = f"{key:20s}"
        for n in names:
            v = results[n]["transition"][key]
            row += f"{v:20.4f}"
        print(row)
    print()

    # Consecutive distances
    print("METRIC 3: Consecutive Action Distances")
    print(sep)
    print(header)
    for key in ["mean", "median", "p95", "p99", "max", "frac_gt_1.0", "frac_gt_2.0", "frac_gt_3.0", "frac_gt_5.0"]:
        row = f"{key:20s}"
        for n in names:
            v = results[n]["consecutive"][key]
            row += f"{v:20.4f}"
        print(row)
    print()

    # Entropy rate
    print("METRIC 4: Token Sequence Entropy Rate (bits)")
    print(sep)
    print(header)
    row = f"{'mean_entropy':20s}"
    for n in names:
        v = results[n]["entropy"]["mean_entropy"]
        row += f"{v:20.4f}"
    print(row)
    for h in range(len(results[names[0]]["entropy"]["per_position_entropy"])):
        row = f"{f'position_{h}':20s}"
        for n in names:
            v = results[n]["entropy"]["per_position_entropy"][h]
            row += f"{v:20.4f}"
        print(row)
    print()

    # Recon error
    print("METRIC 5: Per-Token Reconstruction Error")
    print(sep)
    print(header)
    for key in ["num_tokens_with_data", "mean_per_token_mse", "std_per_token_mse", "max_per_token_mse"]:
        row = f"{key:20s}"
        for n in names:
            v = results[n]["recon_error"][key]
            if isinstance(v, float):
                row += f"{v:20.6f}"
            else:
                row += f"{v:20d}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Token Sequence Statistics for OATTok Comparison")
    parser.add_argument("--tokenizer_dirs", type=str, required=True,
                        help="Comma-separated list of tokenizer training output directories")
    parser.add_argument("--tokenizer_names", type=str, required=True,
                        help="Comma-separated names for each tokenizer (e.g., FSQ,A2Lex,ZHill)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to LIBERO-10 Zarr dataset")
    parser.add_argument("--output_dir", type=str, default="./analysis_results/token_seq_stats",
                        help="Directory to save plots and metrics")
    parser.add_argument("--horizon", type=int, default=32, help="Action chunk horizon")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride for extracting action chunks (1=all, higher=subsample)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for tokenization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tok_dirs = [d.strip() for d in args.tokenizer_dirs.split(",")]
    tok_names = [n.strip() for n in args.tokenizer_names.split(",")]
    assert len(tok_dirs) == len(tok_names), "Number of dirs must match number of names"

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # --- Load data ---
    print("Extracting action chunks from Zarr dataset ...")
    actions, episode_ids = extract_action_chunks(args.data_dir, horizon=args.horizon, stride=args.stride)
    print(f"  Total action chunks: {len(actions)}  (horizon={args.horizon}, stride={args.stride})")

    # --- Run analysis per tokenizer ---
    results = {}
    for name, tok_dir in zip(tok_names, tok_dirs):
        print(f"\n{'='*60}")
        print(f"Analyzing tokenizer: {name}")
        print(f"{'='*60}")

        tok = load_tokenizer(tok_dir, device)
        K = tok.quantizer.codebook_size

        print("  Tokenizing all action chunks ...")
        tokens = tokenize_all(tok, actions, device, batch_size=args.batch_size)
        print(f"  Token tensor shape: {tokens.shape}")

        print("  Computing codebook utilization ...")
        codebook = compute_codebook_utilization(tokens, K)

        print("  Computing transition smoothness ...")
        transition = compute_transition_smoothness(tokens, tok, device, episode_ids)

        print("  Computing consecutive action distances ...")
        consecutive = compute_consecutive_action_distances(tokens, tok, device, episode_ids)

        print("  Computing entropy rate ...")
        entropy = compute_entropy_rate(tokens, K, episode_ids)

        print("  Computing per-token reconstruction error ...")
        recon_error = compute_per_token_recon_error(tokens, actions, tok, device, K)

        results[name] = {
            "codebook": codebook,
            "transition": transition,
            "consecutive": consecutive,
            "entropy": entropy,
            "recon_error": recon_error,
        }

        # Free GPU memory
        del tok
        torch.cuda.empty_cache()

    # --- Print summary ---
    print_summary(results)

    # --- Plots ---
    print("Generating plots ...")
    plot_codebook_utilization(results, args.output_dir)
    plot_consecutive_action_distances(results, args.output_dir)
    plot_transition_smoothness(results, args.output_dir)
    plot_per_token_recon_error(results, args.output_dir)
    plot_entropy_rate(results, args.output_dir)

    # --- Save JSON metrics (strip numpy arrays) ---
    json_results = {}
    for name, r in results.items():
        jr = {}
        for metric_name, metric_data in r.items():
            jr[metric_name] = {
                k: v for k, v in metric_data.items()
                if not isinstance(v, np.ndarray)
            }
            # Convert per_token_mse_dict keys to strings for JSON
            if "per_token_mse_dict" in jr[metric_name]:
                jr[metric_name]["per_token_mse_dict"] = {
                    str(k): v for k, v in jr[metric_name]["per_token_mse_dict"].items()
                }
        json_results[name] = jr

    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nMetrics saved to {json_path}")
    print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
