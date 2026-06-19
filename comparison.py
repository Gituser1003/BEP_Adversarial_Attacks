import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv('Data.csv')

# Reduce to only final iterations & verify stopping rule
def _verify_and_get_last_idx(grp):
    rid = grp.name
    grp = grp.sort_values('iteration')
    flip_iters = grp.index[grp['os_c'] != grp['paras_c']]
    if len(flip_iters) > 0:
        assert flip_iters[-1] == grp.index[-1], \
            f"rewrite_id {rid}: class flip found before final iteration"
        assert grp.loc[grp.index[-1], 'asr'] == 1, \
            f"rewrite_id {rid}: class flipped but asr != 1"
    else:
        assert grp.loc[grp.index[-1], 'asr'] == 0, \
            f"rewrite_id {rid}: no class flip but asr == 1"
    return grp.index[-1]

idx_last = df.groupby('rewrite_id', group_keys=False).apply(_verify_and_get_last_idx, include_groups=False)
attempts = df.loc[idx_last].copy()
attempts.rename(columns={'iteration': 'n_attempts', 'cp_delta_norm': 'best_cp_delta_norm'}, inplace=True)

human = attempts[attempts['source'] == 'human']
llm   = attempts[attempts['source'] == 'llm']

print("Dataset overview")
print(f"  Total sequences : {len(attempts)}")
print(f"  Human           : {len(human)}")
print(f"  LLM             : {len(llm)}")


# Helper: rank-biserial correlation effect size for Mann-Whitney U
def rank_biserial(u_stat, n1, n2):
    """r = 1 - (2U / (n1 * n2)). Range [-1, 1]; |r|: 0.1 small, 0.3 medium, 0.5 large."""
    return 1 - (2 * u_stat) / (n1 * n2)


# 1. Success rate
print("\n1. Success rate")

grouped = attempts.groupby('source')['asr'].agg(
    total='count',
    successful='sum'
)
grouped['success_rate_%'] = (grouped['successful'] / grouped['total'] * 100).round(2)
print(grouped.to_string())

contingency = np.array([
    [grouped.loc['human', 'successful'], grouped.loc['human', 'total'] - grouped.loc['human', 'successful']],
    [grouped.loc['llm',   'successful'], grouped.loc['llm',   'total'] - grouped.loc['llm',   'successful']]
])
chi2, p_asr, dof, _ = stats.chi2_contingency(contingency)

# Effect size: Cramer's V for chi-square
n_total = grouped['total'].sum()
cramers_v = np.sqrt(chi2 / n_total)
print(f"\nChi-square: {chi2:.4f}, df={dof}, p={p_asr:.4f}")
print(f"Cramer's V (effect size): {cramers_v:.4f}")
print("Significant (p<0.05):", p_asr < 0.05)


# 2. Number of attempts (successful sequences only)
print("\n2. Number of attempts (successful sequences only)")

human_succ = human[human['asr'] == 1]
llm_succ   = llm[llm['asr'] == 1]

for group, label in [(human_succ, 'Human'), (llm_succ, 'LLM')]:
    s = group['n_attempts']
    print(f"\n{label}:")
    print(f"  Mean   : {s.mean():.2f}")
    print(f"  Median : {s.median():.1f}")
    print(f"  Std    : {s.std():.2f}")
    print(f"  Min/Max: {s.min()} / {s.max()}")

u_stat_s, p_attempts_s = stats.mannwhitneyu(
    human_succ['n_attempts'], llm_succ['n_attempts'], alternative='two-sided'
)
r_attempts = rank_biserial(u_stat_s, len(human_succ), len(llm_succ))
print(f"\nMann-Whitney U: {u_stat_s:.1f}, p={p_attempts_s:.4f}")
print(f"Rank-biserial r (effect size): {r_attempts:.4f}")
print("Significant (p<0.05):", p_attempts_s < 0.05)


# 3. Change in classifier's prediction confidence (successful sequences only)
print("\n3. cp_delta_norm (successful sequences only)")

for group, label in [(human_succ, 'Human'), (llm_succ, 'LLM')]:
    s = group['best_cp_delta_norm']
    print(f"\n{label}:")
    print(f"  Mean   : {s.mean():.4f}")
    print(f"  Median : {s.median():.4f}")
    print(f"  Std    : {s.std():.4f}")
    print(f"  Min/Max: {s.min():.4f} / {s.max():.4f}")

u_stat_cp, p_cp = stats.mannwhitneyu(
    human_succ['best_cp_delta_norm'],
    llm_succ['best_cp_delta_norm'],
    alternative='two-sided'
)
r_cp = rank_biserial(u_stat_cp, len(human_succ), len(llm_succ))
print(f"\nMann-Whitney U: {u_stat_cp:.1f}, p={p_cp:.4f}")
print(f"Rank-biserial r (effect size): {r_cp:.4f}")
print("Significant (p<0.05):", p_cp < 0.05)


# 4. Summary
print("\n4. Summary")
summary = pd.DataFrame({
    'Metric'     : ['Success rate (%)', 'Mean attempts (successful only)', 'Mean cp_delta_norm (successful)'],
    'Human'      : [
        f"{grouped.loc['human', 'success_rate_%']}%",
        f"{human_succ['n_attempts'].mean():.2f}",
        f"{human_succ['best_cp_delta_norm'].mean():.4f}",
    ],
    'LLM'        : [
        f"{grouped.loc['llm', 'success_rate_%']}%",
        f"{llm_succ['n_attempts'].mean():.2f}",
        f"{llm_succ['best_cp_delta_norm'].mean():.4f}",
    ],
    'Statistic'  : [f"χ²(1) = {chi2:.4f}", f"U = {u_stat_s:.1f}", f"U = {u_stat_cp:.1f}"],
    'p-value'    : [f"{p_asr:.4f}", f"{p_attempts_s:.4f}", f"{p_cp:.4f}"],
    'Effect size': [f"V = {cramers_v:.4f}", f"r = {r_attempts:.4f}", f"r = {r_cp:.4f}"],
    'Significant': [p_asr < 0.05, p_attempts_s < 0.05, p_cp < 0.05]
})
print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Figure: Three-panel comparison of humans vs LLM
# ---------------------------------------------------------------------------

# APA-style colours: muted blue and orange
COLOR_HUMAN = '#4878CF'
COLOR_LLM   = '#E87820'

fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
fig.subplots_adjust(wspace=0.42)

# --- Panel 1: Success rate (bar chart) ---
ax1 = axes[0]
sr_human = grouped.loc['human', 'success_rate_%']
sr_llm   = grouped.loc['llm',   'success_rate_%']
bars = ax1.bar(['Human', 'LLM'], [sr_human, sr_llm],
               color=[COLOR_HUMAN, COLOR_LLM], width=0.5, edgecolor='white')
ax1.set_ylim(0, 85)
ax1.set_ylabel('Success rate (%)', fontsize=10)
ax1.set_title('(a) Attack success rate', fontsize=10, pad=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax1.set_axisbelow(True)
# Annotate bars with values
for bar, val in zip(bars, [sr_human, sr_llm]):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

# --- Panel 2: Number of attempts (boxplot) ---
ax2 = axes[1]
bp2 = ax2.boxplot(
    [human_succ['n_attempts'].values, llm_succ['n_attempts'].values],
    tick_labels=['Human', 'LLM'],
    patch_artist=True,
    medianprops=dict(color='white', linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker='o', markersize=4, linestyle='none', alpha=0.5)
)
for patch, color in zip(bp2['boxes'], [COLOR_HUMAN, COLOR_LLM]):
    patch.set_facecolor(color)
for flier, color in zip(bp2['fliers'], [COLOR_HUMAN, COLOR_LLM]):
    flier.set_markerfacecolor(color)
    flier.set_markeredgecolor(color)
ax2.set_ylabel('Number of attempts', fontsize=10)
ax2.set_title('(b) Number of attempts', fontsize=10, pad=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax2.set_axisbelow(True)

# --- Panel 3: cp_delta_norm (boxplot) ---
ax3 = axes[2]
bp3 = ax3.boxplot(
    [human_succ['best_cp_delta_norm'].values, llm_succ['best_cp_delta_norm'].values],
    tick_labels=['Human', 'LLM'],
    patch_artist=True,
    medianprops=dict(color='white', linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker='o', markersize=4, linestyle='none', alpha=0.5)
)
for patch, color in zip(bp3['boxes'], [COLOR_HUMAN, COLOR_LLM]):
    patch.set_facecolor(color)
for flier, color in zip(bp3['fliers'], [COLOR_HUMAN, COLOR_LLM]):
    flier.set_markerfacecolor(color)
    flier.set_markeredgecolor(color)
ax3.set_ylabel('Confidence shift', fontsize=10)
ax3.set_title('(c) Confidence shift', fontsize=10, pad=8)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax3.set_axisbelow(True)

plt.tight_layout()
plt.savefig('figure_comparison.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figure_comparison.png', bbox_inches='tight', dpi=300)
print("\nFigure saved as figure_comparison.pdf and figure_comparison.png")