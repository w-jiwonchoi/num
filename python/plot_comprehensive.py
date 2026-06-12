"""
plot_comprehensive.py  (v0.3 — fixed ls= kwarg conflict)
=========================================================
Fix: style_of() 가 ls 키를 포함할 때 grid_lines 의 ls='-' 와 충돌하던 버그 수정.
     → plot() 호출 전에 style dict 에서 ls 를 꺼내 별도 처리.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE        = Path(__file__).parent.parent / "benchmark"
RESULTS_DIR = BASE / "results"
PLOTS_DIR   = BASE / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    'scipy_cpu':         dict(color='#888888', marker='o',  label='SciPy (CPU)'),
    'jax_1gpu':          dict(color='#2196F3', marker='s',  label='JAX BCOO (1 GPU)'),
    'kk_1gpu':           dict(color='#4CAF50', marker='^',  label='KokkosKernels (1 GPU)'),
    'custom_1gpu':       dict(color='#FF9800', marker='D',  label='Custom kernel (1 GPU)'),
    'kk_2gpu':           dict(color='#26C6DA', marker='v',  label='KK dist (2 GPU)'),
    'custom_2gpu':       dict(color='#AB47BC', marker='v',  label='Custom dist (2 GPU)'),
    'kk_4gpu':           dict(color='#E91E63', marker='*',  label='KK dist (4 GPU)'),
    'custom_4gpu':       dict(color='#D32F2F', marker='*',  label='Custom dist (4 GPU)'),
    # noovl 계열은 ls='--' 를 style dict 안에 명시적으로 유지
    'kk_noovl_2gpu':     dict(color='#9E9E9E', marker='x',  ls='--', label='KK 2GPU (no-ovl)'),
    'kk_noovl_4gpu':     dict(color='#607D8B', marker='x',  ls='--', label='KK 4GPU (no-ovl)'),
    'jax_vmap':          dict(color='#2196F3', marker='s',  label='JAX vmap'),
    'kokkos_batch':      dict(color='#FF9800', marker='D',  label='Kokkos batch (kk)'),
    'custom_batch':      dict(color='#F44336', marker='P',  label='Custom batch'),
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 8, 'figure.dpi': 150,
})


def load_all(prefix):
    dfs = []
    for p in sorted(RESULTS_DIR.glob(f"{prefix}_*.csv")):
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"  [WARN] {p.name}: {e}", file=sys.stderr)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    keys = [c for c in ('backend', 'matrix', 'N', 'k', 'n_ranks', 'overlap')
            if c in df.columns]
    df = df.drop_duplicates(subset=keys, keep='last')
    if 'matrix' not in df.columns:
        df['matrix'] = 'laplacian_3d'
    return df


def style_of(b):
    """STYLE dict 복사본을 반환. ls 는 항상 포함 (없으면 '-' 기본값)."""
    s = dict(STYLE.get(b, dict(label=b)))
    if 'ls' not in s:
        s['ls'] = '-'
    return s


def grid_lines(df, ycol, ylabel, title, path, logy=True):
    mats = sorted(df['matrix'].dropna().unique())
    if not mats:
        return
    ncol = min(3, len(mats))
    nrow = int(np.ceil(len(mats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.8 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for idx, mat in enumerate(mats):
        ax = axes[idx // ncol][idx % ncol]
        ax.set_visible(True)
        sub = df[df['matrix'] == mat]
        for backend, grp in sub.groupby('backend'):
            g = grp.dropna(subset=['N', ycol]).sort_values('N')
            if len(g) == 0:
                continue
            s = style_of(backend)   # ls 는 여기에 이미 포함
            ax.plot(g['N'], g[ycol], **s)
        ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel('Matrix size N')
        ax.set_ylabel(ylabel)
        ax.set_title(mat)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(framealpha=0.85)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    print(f"  Saved {path}")
    plt.close(fig)


def fig_batch(df):
    sub = df.dropna(subset=['k']) if 'k' in df.columns else pd.DataFrame()
    sub = sub[sub['backend'].isin(['jax_vmap', 'kokkos_batch', 'custom_batch'])] \
          if not sub.empty else sub
    if sub.empty:
        print("  [SKIP] figC — no batch data")
        return
    mats = sorted(sub['matrix'].unique())
    ncol = min(3, len(mats))
    nrow = int(np.ceil(len(mats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.8 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for idx, mat in enumerate(mats):
        ax = axes[idx // ncol][idx % ncol]
        ax.set_visible(True)
        for backend, grp in sub[sub['matrix'] == mat].groupby('backend'):
            for Nval, g2 in grp.groupby('N'):
                g2 = g2.sort_values('k')
                s = style_of(backend)
                s['label'] = f"{s.get('label', backend)} (N={int(Nval):,})"
                ax.plot(g2['k'], g2['gflops'], **s)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Batch size k')
        ax.set_ylabel('GFLOPS')
        ax.set_title(mat)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(framealpha=0.85)
    fig.suptitle('Batch SpMV throughput vs k', y=1.0)
    fig.tight_layout()
    p = PLOTS_DIR / "figC_batch_sweep.png"
    fig.savefig(p, bbox_inches='tight')
    print(f"  Saved {p}")
    plt.close(fig)


def fig_scaling(df):
    pat = re.compile(r'^(kk|custom)_(\d)gpu$')
    sub  = df[df['backend'].str.match(pat, na=False)].copy()
    base = df[df['backend'].isin(['kk_1gpu', 'custom_1gpu'])].copy()
    if sub.empty or base.empty:
        print("  [SKIP] figD — need both 1-GPU and dist results")
        return
    base['family']  = base['backend'].str.replace('_1gpu', '', regex=False)
    base['n_ranks'] = 1
    sub['family']   = sub['backend'].str.extract(pat)[0]
    sub['n_ranks']  = sub['backend'].str.extract(pat)[1].astype(int)
    allrows = pd.concat([base, sub], ignore_index=True)

    mats = sorted(allrows['matrix'].unique())
    fig, axes = plt.subplots(1, len(mats), figsize=(4.6 * len(mats), 4),
                             squeeze=False)
    colors = {'kk': '#4CAF50', 'custom': '#FF9800'}
    for idx, mat in enumerate(mats):
        ax = axes[0][idx]
        m = allrows[allrows['matrix'] == mat]
        common_N = None
        for Nval in sorted(m['N'].unique(), reverse=True):
            if m[m['N'] == Nval]['n_ranks'].nunique() >= 2:
                common_N = Nval
                break
        if common_N is None:
            continue
        m = m[m['N'] == common_N]
        for fam, g in m.groupby('family'):
            g = g.sort_values('n_ranks')
            t1 = g[g['n_ranks'] == 1]['elapsed_s']
            ax.plot(g['n_ranks'], g['elapsed_us'], 'o-',
                    color=colors.get(fam, None),
                    label=f"{fam} (N={int(common_N):,})")
            if len(t1):
                ideal = float(t1.iloc[0]) * 1e6 / g['n_ranks'].values
                ax.plot(g['n_ranks'], ideal, ':',
                        color=colors.get(fam), alpha=0.5,
                        label=f"{fam} ideal")
        ax.set_xlabel('#GPUs')
        ax.set_ylabel('Time (μs)')
        ax.set_yscale('log')
        ax.set_xticks([1, 2, 4])
        ax.set_title(f"{mat}")
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.85)
    fig.suptitle('Strong scaling — distributed SpMV', y=1.02)
    fig.tight_layout()
    p = PLOTS_DIR / "figD_scaling.png"
    fig.savefig(p, bbox_inches='tight')
    print(f"  Saved {p}")
    plt.close(fig)


def fig_overlap(df):
    """noovl vs overlap 비교. noovl 계열 데이터가 있을 때만 생성."""
    if 'backend' not in df.columns:
        print("  [SKIP] figF — no backend column")
        return
    noovl = df[df['backend'].str.contains('noovl', na=False)]
    if noovl.empty:
        print("  [SKIP] figF — run with --no-overlap to collect data")
        return
    ovl = df[df['backend'].str.match(r'^kk_\dgpu$', na=False)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for tag, grp, ls in [('overlap ON', ovl, '-'), ('overlap OFF', noovl, '--')]:
        for mat, g2 in grp.groupby('matrix'):
            g2 = g2.sort_values('N')
            ax.plot(g2['N'], g2['elapsed_us'], ls,
                    label=f"{tag} ({mat})")
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('N'); ax.set_ylabel('Time (μs)')
    ax.set_title('Distributed SpMV — overlap ON vs OFF (4 GPU)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    p = PLOTS_DIR / "figF_overlap.png"
    fig.savefig(p, bbox_inches='tight')
    print(f"  Saved {p}")
    plt.close(fig)


def fig_cg(cg):
    if cg.empty:
        print("  [SKIP] figE — no CG data")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cmap = {'scipy_cg': '#888888', 'petsc_cg': '#2196F3',
            'kspmv_cg_1gpu': '#FF9800', 'kspmv_cg_2gpu': '#AB47BC',
            'kspmv_cg_4gpu': '#E91E63'}
    for backend, g in cg.groupby('backend'):
        if 'cg' not in str(backend):
            continue
        g = g.dropna(subset=['N', 'elapsed_s']).sort_values('N')
        if g.empty:
            continue
        ax.plot(g['N'], g['elapsed_s'], 'o-',
                color=cmap.get(backend), label=backend)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('N'); ax.set_ylabel('Solve time (s)')
    ax.set_title('CG solver — laplacian_3d, tol=1e-8')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    p = PLOTS_DIR / "figE_cg.png"
    fig.savefig(p, bbox_inches='tight')
    print(f"  Saved {p}")
    plt.close(fig)


def main():
    print(f"\nLoading CSVs from {RESULTS_DIR}")
    spmv = load_all('spmv')
    cg   = load_all('cg')
    print(f"  spmv rows: {len(spmv)}, cg rows: {len(cg)}\n")

    if not spmv.empty:
        single_and_dist = spmv[spmv.get('k', pd.Series(dtype=float)).isna()] \
                          if 'k' in spmv.columns else spmv
        grid_lines(single_and_dist, 'elapsed_us', 'Time (μs)',
                   'SpMV time vs matrix size (lower = better)',
                   PLOTS_DIR / "figA_time_vs_n.png")
        grid_lines(single_and_dist, 'bw_GB_s', 'Bandwidth (GB/s)',
                   'SpMV effective bandwidth vs matrix size',
                   PLOTS_DIR / "figB_bw_vs_n.png", logy=False)
        fig_batch(spmv)
        fig_scaling(single_and_dist)
        fig_overlap(single_and_dist)
    fig_cg(cg)
    print("\nDone — plots in", PLOTS_DIR)


if __name__ == '__main__':
    main()
