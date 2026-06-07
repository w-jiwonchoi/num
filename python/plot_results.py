"""
plot_results.py
===============
Generates Figures 1–5 from benchmark CSV files.

Usage:
    python3 plot_results.py                          # auto-detect latest CSVs
    python3 plot_results.py --spmv path/to/spmv.csv --cg path/to/cg.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PLOTS_DIR = Path(__file__).parent.parent / "benchmark" / "plots"
RESULTS_DIR = Path(__file__).parent.parent / "benchmark" / "results"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
STYLE = {
    'scipy_cpu':           dict(color='#888888', ls='-',  marker='o', label='SciPy (CPU)'),
    'jax_1gpu':            dict(color='#2196F3', ls='-',  marker='s', label='JAX BCOO (1 GPU)'),
    'kokkos_kk_1gpu':      dict(color='#4CAF50', ls='-',  marker='^', label='KokkosKernels (1 GPU)'),
    'kokkos_custom_1gpu':  dict(color='#FF9800', ls='--', marker='D', label='Custom TeamPolicy (1 GPU)'),
    'kokkos_dist_2gpu':    dict(color='#9C27B0', ls='-',  marker='v', label='Kokkos Dist (2 GPU)'),
    'kokkos_dist_4gpu':    dict(color='#E91E63', ls='-',  marker='*', label='Kokkos Dist (4 GPU)'),
}

BACKEND_ORDER = list(STYLE)

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':       11,
    'axes.labelsize':  12,
    'axes.titlesize':  13,
    'legend.fontsize': 9,
    'figure.dpi':      150,
})


def latest_csv(prefix: str) -> Path | None:
    csvs = sorted(RESULTS_DIR.glob(f"{prefix}_*.csv"))
    return csvs[-1] if csvs else None


# ─── Fig 1: SpMV Bandwidth vs N ──────────────────────────────────────────────

def fig1_bw_vs_n(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    for backend, grp in df.groupby('backend'):
        s = STYLE.get(backend, dict(label=backend))
        grp_s = grp.sort_values('N')
        ax.plot(grp_s['N'], grp_s['bw_GB_s'], **s)

    ax.set_xscale('log')
    ax.set_xlabel('Matrix size N')
    ax.set_ylabel('Memory Bandwidth (GB/s)')
    ax.set_title('Fig 1 — SpMV Throughput vs Matrix Size')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)

    # Reference lines
    ax.axhline(2000, ls=':', color='red', alpha=0.5, label='A100 peak (2 TB/s)')
    ax.axhline(50,   ls=':', color='gray', alpha=0.4, label='CPU peak (~50 GB/s)')

    fig.tight_layout()
    path = PLOTS_DIR / "fig1_bw_vs_n.png"
    fig.savefig(path, bbox_inches='tight')
    print(f"  Saved {path}")
    plt.close(fig)


# ─── Fig 2: Batch SpMV Throughput vs k ───────────────────────────────────────

def fig2_batch_vs_k(df: pd.DataFrame):
    """Expects df with columns: backend, k, gflops."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for backend, grp in df.groupby('backend'):
        if 'k' not in grp.columns:
            continue
        grp_s = grp.dropna(subset=['k']).sort_values('k')
        s = STYLE.get(backend, dict(label=backend))
        ax.plot(grp_s['k'], grp_s['gflops'], **s)

    ax.set_xlabel('Batch size k (# RHS vectors)')
    ax.set_ylabel('Throughput (GFLOPS)')
    ax.set_title('Fig 2 — Batch SpMV Throughput vs Batch Size')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.legend(framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / "fig2_batch_vs_k.png"
    fig.savefig(path, bbox_inches='tight')
    print(f"  Saved {path}")
    plt.close(fig)


# ─── Fig 3: Multi-GPU SpMV Scaling Efficiency ────────────────────────────────

def fig3_spmv_scaling(df: pd.DataFrame):
    """Expects 'n_ranks' column and backend like 'kspmv_spmv_Ngpu'."""
    spmv_rows = df[df['backend'].str.contains('spmv', na=False)].copy()
    if spmv_rows.empty:
        print("  [SKIP] Fig 3 — no distributed SpMV data")
        return

    spmv_rows['n_ranks'] = spmv_rows['n_ranks'].fillna(1).astype(int)
    base = spmv_rows[spmv_rows['n_ranks'] == 1]['elapsed_s'].mean()
    if np.isnan(base):
        print("  [SKIP] Fig 3 — missing single-GPU baseline")
        return

    spmv_rows['efficiency'] = (base / spmv_rows['elapsed_s'] / spmv_rows['n_ranks']) * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    n = spmv_rows['n_ranks'].values
    eff = spmv_rows['efficiency'].values
    ax.plot(n, eff, 'o-', color='#4CAF50', label='SpMV scaling efficiency')
    ax.axhline(100, ls='--', color='gray', alpha=0.6, label='Ideal (100%)')

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Scaling Efficiency (%)')
    ax.set_title('Fig 3 — Multi-GPU SpMV Scaling Efficiency')
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / "fig3_spmv_scaling.png"
    fig.savefig(path, bbox_inches='tight')
    print(f"  Saved {path}")
    plt.close(fig)


# ─── Fig 4: CG Scaling vs SpMV Scaling ───────────────────────────────────────

def fig4_cg_vs_spmv_scaling(cg_df: pd.DataFrame, spmv_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for (df, label, color) in [
        (spmv_df, 'SpMV (linear expected)',  '#4CAF50'),
        (cg_df,   'CG (allreduce penalty)',  '#E91E63'),
    ]:
        sub = df[df.get('n_ranks', pd.Series([1] * len(df))).notna()].copy()
        if sub.empty:
            continue
        if 'n_ranks' not in sub.columns:
            sub['n_ranks'] = 1
        sub['n_ranks'] = sub['n_ranks'].fillna(1).astype(int)
        base = sub[sub['n_ranks'] == 1]['elapsed_s'].mean()
        if np.isnan(base):
            continue
        sub['speedup'] = base / sub['elapsed_s']
        sub_s = sub.sort_values('n_ranks')
        ax.plot(sub_s['n_ranks'], sub_s['speedup'], 'o-', color=color, label=label)

    # Ideal line
    gpus = np.array([1, 2, 4, 8])
    ax.plot(gpus, gpus, 'k--', alpha=0.4, label='Ideal speedup')

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Speedup vs single GPU')
    ax.set_title('Fig 4 — Scaling: CG vs SpMV')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / "fig4_cg_vs_spmv_scaling.png"
    fig.savefig(path, bbox_inches='tight')
    print(f"  Saved {path}")
    plt.close(fig)


# ─── Fig 5: Convergence history ──────────────────────────────────────────────

def fig5_convergence(cg_df: pd.DataFrame):
    """Plot residual vs iteration (if 'history' column present)."""
    if 'history' not in cg_df.columns:
        print("  [SKIP] Fig 5 — no convergence history in CSV")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for backend, grp in cg_df.groupby('backend'):
        for _, row in grp.iterrows():
            hist = [float(v) for v in str(row['history']).split(';') if v]
            if hist:
                ax.semilogy(range(len(hist)), hist,
                            label=f"{backend} ({row.get('n_ranks', 1)} GPU)")

    ax.set_xlabel('CG Iteration')
    ax.set_ylabel('Relative Residual ||r||/||b||')
    ax.set_title('Fig 5 — CG Convergence History')
    ax.axhline(1e-8, ls='--', color='gray', alpha=0.6, label='tol=1e-8')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / "fig5_convergence.png"
    fig.savefig(path, bbox_inches='tight')
    print(f"  Saved {path}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--spmv', type=Path, default=None)
    p.add_argument('--cg',   type=Path, default=None)
    return p.parse_args()


def load_df(path: Path | None, prefix: str) -> pd.DataFrame:
    if path is None:
        path = latest_csv(prefix)
    if path is None or not path.exists():
        print(f"  [WARN] No {prefix} CSV found — skipping related figures")
        return pd.DataFrame()
    print(f"  Loading {path}")
    return pd.read_csv(path)


def main():
    args = parse_args()
    print("\nGenerating plots from benchmark results...")
    print(f"  Output directory: {PLOTS_DIR}\n")

    spmv_df = load_df(args.spmv, 'spmv')
    cg_df   = load_df(args.cg,   'cg')

    if not spmv_df.empty:
        print("Generating Fig 1 ...")
        fig1_bw_vs_n(spmv_df)
        print("Generating Fig 2 ...")
        fig2_batch_vs_k(spmv_df)
        print("Generating Fig 3 ...")
        fig3_spmv_scaling(spmv_df)

    if not cg_df.empty:
        print("Generating Fig 4 ...")
        fig4_cg_vs_spmv_scaling(cg_df, spmv_df if not spmv_df.empty else pd.DataFrame())
        print("Generating Fig 5 ...")
        fig5_convergence(cg_df)

    print("\nDone.")


if __name__ == '__main__':
    main()
