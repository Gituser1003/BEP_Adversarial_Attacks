import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import KFold
from scipy import stats

import random
import numpy as np
import torch

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Load data ---
train_df = pd.read_csv('hippocorpus_training_truncated.csv')
train_df['label'] = (train_df['condition'] == 'deceptive').astype(int)

test_df = pd.read_csv('hippocorpus_test_truncated.csv')
test_df['label'] = (test_df['condition'] == 'deceptive').astype(int)

data_df  = pd.read_csv('Data.csv')
idx_last = data_df.groupby('rewrite_id')['iteration'].idxmax()
final    = data_df.loc[idx_last]

succ = final[final['asr'] == 1].copy()
succ['label'] = (succ['original_condition'] == 'deceptive').astype(int)

human_succ = succ[succ['source'] == 'human'].reset_index(drop=True)
llm_succ   = succ[succ['source'] == 'llm'].reset_index(drop=True)

print(f"Original training narratives : {len(train_df)}")
print(f"Successful human paraphrases : {len(human_succ)}")
print(f"Successful LLM paraphrases   : {len(llm_succ)}")

# Dataset
class NarrativeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.labels    = torch.tensor(list(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        loss   = loss_fn(logits, batch['labels'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_loader(model, loader):
    """
    Returns accuracy, weighted F1, ASR, ROC AUC, per-class metrics dict, and
    the raw per-item 'wrong' boolean array (True = misclassified).
    Per-class dict has keys 'precision', 'recall', 'f1' each as a list
    indexed by class label [0, 1].
    The 'wrong' array is needed for McNemar's test, which requires paired
    per-item outcomes between two models evaluated on the same items in the
    same order (guaranteed here since loaders are built once per fold and
    reused across the base/human/llm/combined models).
    """
    model.eval()
    preds, true_labels, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            softmax_probs = torch.softmax(logits, dim=1)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
            probs.extend(softmax_probs[:, 1].cpu().numpy())  # probability of positive class

    preds       = np.array(preds)
    true_labels = np.array(true_labels)
    probs       = np.array(probs)
    wrong       = preds != true_labels

    acc = accuracy_score(true_labels, preds)
    f1  = f1_score(true_labels, preds, average='weighted')
    asr = np.mean(wrong)

    # ROC AUC — guard against single-class edge case in small held-out splits
    n_unique = len(np.unique(true_labels))
    auc = roc_auc_score(true_labels, probs) if n_unique > 1 else float('nan')

    # Per-class precision, recall, F1
    report = classification_report(true_labels, preds, output_dict=True, zero_division=0)
    per_class = {
        'precision': [report.get('0', {}).get('precision', float('nan')),
                      report.get('1', {}).get('precision', float('nan'))],
        'recall':    [report.get('0', {}).get('recall',    float('nan')),
                      report.get('1', {}).get('recall',    float('nan'))],
        'f1':        [report.get('0', {}).get('f1-score',  float('nan')),
                      report.get('1', {}).get('f1-score',  float('nan'))],
    }

    return acc, f1, asr, auc, per_class, wrong

def cohen_h(p1, p2):
    """
    Cohen's h effect size for the difference between two proportions.
    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    Positive h means p1 > p2 (i.e. ASR went up relative to base).
    """
    p1 = np.clip(p1, 0.0, 1.0)
    p2 = np.clip(p2, 0.0, 1.0)
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

def mcnemar_test(wrong_a, wrong_b):
    """
    McNemar's test for paired binary outcomes (same items, two models).
    wrong_a, wrong_b: boolean arrays, same length and order, True = misclassified.

    Returns (statistic, p_value, b01, b10) where:
      b01 = items model A got right but model B got wrong
      b10 = items model A got wrong but model B got right
    Uses the continuity-corrected chi-square statistic when the number of
    discordant pairs is large enough (>= 25); otherwise falls back to the
    exact binomial test, as is standard practice for small discordant counts.
    """
    wrong_a = np.asarray(wrong_a, dtype=bool)
    wrong_b = np.asarray(wrong_b, dtype=bool)
    assert len(wrong_a) == len(wrong_b), "Paired arrays must be the same length"

    b01 = int(np.sum(~wrong_a & wrong_b))   # A correct, B wrong
    b10 = int(np.sum(wrong_a & ~wrong_b))   # A wrong, B correct
    n_discordant = b01 + b10

    if n_discordant == 0:
        return 0.0, 1.0, b01, b10

    if n_discordant < 25:
        p_value = stats.binomtest(min(b01, b10), n_discordant, 0.5, alternative='two-sided').pvalue
        statistic = float(min(b01, b10))
    else:
        statistic = (abs(b01 - b10) - 1) ** 2 / n_discordant
        p_value = stats.chi2.sf(statistic, df=1)

    return statistic, p_value, b01, b10

def cohens_g(b01, b10):
    """
    Cohen's g effect size for McNemar designs.
    g = proportion of discordant pairs favouring the larger side - 0.5
    Range [0, 0.5]. Benchmarks (Cohen, 1988): 0.05 small, 0.15 medium, 0.25 large.
    """
    n_discordant = b01 + b10
    if n_discordant == 0:
        return 0.0
    return max(b01, b10) / n_discordant - 0.5

def finetune_model(paras_train, tokenizer):
    combined_texts  = train_df['text_truncated'].tolist() + paras_train['paras'].tolist()
    combined_labels = train_df['label'].tolist()          + paras_train['label'].tolist()

    model = DistilBertForSequenceClassification.from_pretrained('base_model')
    model.to(device)

    dataset = NarrativeDataset(combined_texts, combined_labels, tokenizer)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True)

    class_counts  = np.bincount(combined_labels)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
    class_weights = class_weights / class_weights.sum()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn   = torch.nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(3):
        train_epoch(model, loader, optimizer, loss_fn)

    return model

# Standard test set loader (same for all folds)
tokenizer    = DistilBertTokenizer.from_pretrained('base_model')
std_dataset  = NarrativeDataset(test_df['text_truncated'].tolist(), test_df['label'].tolist(), tokenizer)
std_loader   = DataLoader(std_dataset, batch_size=16)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store results per fold
results = {
    'base':     {'acc': [], 'f1': [], 'auc': [],
                 'asr_human': [], 'asr_llm': [], 'asr_combined': [],
                 'per_class_std': [],
                 'wrong_human': [], 'wrong_llm': [], 'wrong_combined': []},
    'human':    {'acc': [], 'f1': [], 'auc': [],
                 'asr_human': [], 'asr_llm': [], 'asr_combined': [],
                 'per_class_std': [],
                 'wrong_human': [], 'wrong_llm': [], 'wrong_combined': []},
    'llm':      {'acc': [], 'f1': [], 'auc': [],
                 'asr_human': [], 'asr_llm': [], 'asr_combined': [],
                 'per_class_std': [],
                 'wrong_human': [], 'wrong_llm': [], 'wrong_combined': []},
    'combined': {'acc': [], 'f1': [], 'auc': [],
                 'asr_human': [], 'asr_llm': [], 'asr_combined': [],
                 'per_class_std': [],
                 'wrong_human': [], 'wrong_llm': [], 'wrong_combined': []},
}

n_folds  = 5
human_kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
llm_kf   = KFold(n_splits=n_folds, shuffle=True, random_state=42)

human_folds = list(human_kf.split(human_succ))
llm_folds   = list(llm_kf.split(llm_succ))

for fold in range(n_folds):
    print(f"\nFold {fold + 1}/5")

    h_train_idx, h_test_idx = human_folds[fold]
    l_train_idx, l_test_idx = llm_folds[fold]

    human_train = human_succ.iloc[h_train_idx]
    human_test  = human_succ.iloc[h_test_idx]
    llm_train   = llm_succ.iloc[l_train_idx]
    llm_test    = llm_succ.iloc[l_test_idx]

    combined_train = pd.concat([human_train, llm_train])
    combined_test  = pd.concat([human_test, llm_test])

    human_test_loader    = DataLoader(NarrativeDataset(human_test['paras'].tolist(),    human_test['label'].tolist(),    tokenizer), batch_size=16)
    llm_test_loader      = DataLoader(NarrativeDataset(llm_test['paras'].tolist(),      llm_test['label'].tolist(),      tokenizer), batch_size=16)
    combined_test_loader = DataLoader(NarrativeDataset(combined_test['paras'].tolist(), combined_test['label'].tolist(), tokenizer), batch_size=16)

    # Base model
    base_model = DistilBertForSequenceClassification.from_pretrained('base_model').to(device)
    acc, f1, _, auc, pc, _        = evaluate_loader(base_model, std_loader)
    _, _, asr_h, _, _, wrong_h    = evaluate_loader(base_model, human_test_loader)
    _, _, asr_l, _, _, wrong_l    = evaluate_loader(base_model, llm_test_loader)
    _, _, asr_c, _, _, wrong_c    = evaluate_loader(base_model, combined_test_loader)
    results['base']['acc'].append(acc);   results['base']['f1'].append(f1)
    results['base']['auc'].append(auc)
    results['base']['asr_human'].append(asr_h); results['base']['asr_llm'].append(asr_l); results['base']['asr_combined'].append(asr_c)
    results['base']['per_class_std'].append(pc)
    results['base']['wrong_human'].append(wrong_h); results['base']['wrong_llm'].append(wrong_l); results['base']['wrong_combined'].append(wrong_c)

    # Human fine-tuned
    print("  Training: human condition")
    human_model = finetune_model(human_train, tokenizer)
    acc, f1, _, auc, pc, _      = evaluate_loader(human_model, std_loader)
    _, _, asr_h, _, _, wrong_h  = evaluate_loader(human_model, human_test_loader)
    _, _, asr_l, _, _, wrong_l  = evaluate_loader(human_model, llm_test_loader)
    _, _, asr_c, _, _, wrong_c  = evaluate_loader(human_model, combined_test_loader)
    results['human']['acc'].append(acc);  results['human']['f1'].append(f1)
    results['human']['auc'].append(auc)
    results['human']['asr_human'].append(asr_h); results['human']['asr_llm'].append(asr_l); results['human']['asr_combined'].append(asr_c)
    results['human']['per_class_std'].append(pc)
    results['human']['wrong_human'].append(wrong_h); results['human']['wrong_llm'].append(wrong_l); results['human']['wrong_combined'].append(wrong_c)

    # LLM fine-tuned
    print("  Training: LLM condition")
    llm_model = finetune_model(llm_train, tokenizer)
    acc, f1, _, auc, pc, _      = evaluate_loader(llm_model, std_loader)
    _, _, asr_h, _, _, wrong_h  = evaluate_loader(llm_model, human_test_loader)
    _, _, asr_l, _, _, wrong_l  = evaluate_loader(llm_model, llm_test_loader)
    _, _, asr_c, _, _, wrong_c  = evaluate_loader(llm_model, combined_test_loader)
    results['llm']['acc'].append(acc);    results['llm']['f1'].append(f1)
    results['llm']['auc'].append(auc)
    results['llm']['asr_human'].append(asr_h); results['llm']['asr_llm'].append(asr_l); results['llm']['asr_combined'].append(asr_c)
    results['llm']['per_class_std'].append(pc)
    results['llm']['wrong_human'].append(wrong_h); results['llm']['wrong_llm'].append(wrong_l); results['llm']['wrong_combined'].append(wrong_c)

    # Combined fine-tuned
    print("  Training: combined condition")
    combined_model = finetune_model(combined_train, tokenizer)
    acc, f1, _, auc, pc, _      = evaluate_loader(combined_model, std_loader)
    _, _, asr_h, _, _, wrong_h  = evaluate_loader(combined_model, human_test_loader)
    _, _, asr_l, _, _, wrong_l  = evaluate_loader(combined_model, llm_test_loader)
    _, _, asr_c, _, _, wrong_c  = evaluate_loader(combined_model, combined_test_loader)
    results['combined']['acc'].append(acc); results['combined']['f1'].append(f1)
    results['combined']['auc'].append(auc)
    results['combined']['asr_human'].append(asr_h); results['combined']['asr_llm'].append(asr_l); results['combined']['asr_combined'].append(asr_c)
    results['combined']['per_class_std'].append(pc)
    results['combined']['wrong_human'].append(wrong_h); results['combined']['wrong_llm'].append(wrong_l); results['combined']['wrong_combined'].append(wrong_c)

# ---------------------------------------------------------------------------
# Summary table 1 — overall metrics
# ---------------------------------------------------------------------------
print("\n" + "="*100)
print("Results (averaged over 5 folds):")
print(f"  {'Model':<22} {'Accuracy':>10} {'F1 (wtd)':>10} {'AUC':>8} {'ASR (human)':>13} {'ASR (LLM)':>11} {'ASR (combined)':>16}")
print(f"  {'-'*92}")
for name, key in [('Base model', 'base'), ('Human fine-tuned', 'human'), ('LLM fine-tuned', 'llm'), ('Combined', 'combined')]:
    r = results[key]
    print(f"  {name:<22} {np.mean(r['acc']):>10.4f} {np.mean(r['f1']):>10.4f} {np.nanmean(r['auc']):>8.4f}"
          f" {np.mean(r['asr_human']):>13.4f} {np.mean(r['asr_llm']):>11.4f} {np.mean(r['asr_combined']):>16.4f}")

print("\nNote: ASR = proportion of held-out paraphrases that still fool the model. Lower = more robust.")

# ---------------------------------------------------------------------------
# Summary table 2 — per-class precision / recall / F1 on standard test set
# ---------------------------------------------------------------------------
print("\n" + "="*100)
print("Per-class metrics on standard test set (averaged over 5 folds):")
print(f"  {'Model':<22} {'Class':>6} {'Precision':>11} {'Recall':>9} {'F1':>9}")
print(f"  {'-'*60}")
for name, key in [('Base model', 'base'), ('Human fine-tuned', 'human'), ('LLM fine-tuned', 'llm'), ('Combined', 'combined')]:
    pc_list = results[key]['per_class_std']   # list of dicts, one per fold
    for cls_idx, cls_name in [(0, 'truthful'), (1, 'deceptive')]:
        mean_p = np.mean([pc['precision'][cls_idx] for pc in pc_list])
        mean_r = np.mean([pc['recall'][cls_idx]    for pc in pc_list])
        mean_f = np.mean([pc['f1'][cls_idx]        for pc in pc_list])
        row_label = name if cls_idx == 0 else ''
        print(f"  {row_label:<22} {cls_name:>9} {mean_p:>11.4f} {mean_r:>9.4f} {mean_f:>9.4f}")
    print(f"  {'':<22}")   # blank separator between models

# ---------------------------------------------------------------------------
# Summary table 3 — Cohen's h effect sizes for ASR reductions vs base model
# ---------------------------------------------------------------------------
print("="*100)
print("Cohen's h effect sizes for ASR differences (fine-tuned vs base model):")
print("  Negative h = lower ASR than base (more robust). |h|: 0.2 small, 0.5 medium, 0.8 large.")
print(f"\n  {'Model':<22} {'h (human ASR)':>15} {'h (LLM ASR)':>13} {'h (combined ASR)':>18}")
print(f"  {'-'*70}")

base_asr_human    = np.mean(results['base']['asr_human'])
base_asr_llm      = np.mean(results['base']['asr_llm'])
base_asr_combined = np.mean(results['base']['asr_combined'])

for name, key in [('Human fine-tuned', 'human'), ('LLM fine-tuned', 'llm'), ('Combined', 'combined')]:
    r = results[key]
    h_human    = cohen_h(np.mean(r['asr_human']),    base_asr_human)
    h_llm      = cohen_h(np.mean(r['asr_llm']),      base_asr_llm)
    h_combined = cohen_h(np.mean(r['asr_combined']), base_asr_combined)
    print(f"  {name:<22} {h_human:>+15.4f} {h_llm:>+13.4f} {h_combined:>+18.4f}")

print()

# ---------------------------------------------------------------------------
# Summary table 4 — McNemar's test for ASR differences vs base model
# ---------------------------------------------------------------------------
# Per-item 'wrong' arrays are pooled across the 5 folds before testing, since
# each fold holds out a different (non-overlapping) set of paraphrases; the
# pairing required by McNemar's test (same items, two models) only needs to
# hold within each fold, so concatenating across folds yields one valid long
# paired vector per comparison. Cohen's g is reported alongside as the
# corresponding effect size for this paired design.
print("="*100)
print("McNemar's test for ASR differences (fine-tuned vs base model, pooled across folds):")
print("  Tests whether a fine-tuned model is significantly more/less often fooled than the")
print("  base model on the same held-out paraphrases. Cohen's g: 0.05 small, 0.15 medium, 0.25 large.")
print(f"\n  {'Comparison':<28} {'Test set':<10} {'Statistic':>10} {'p':>8} {'g':>8}")
print(f"  {'-'*70}")

for name, key in [('Human fine-tuned', 'human'), ('LLM fine-tuned', 'llm'), ('Combined', 'combined')]:
    r = results[key]
    for test_label, test_key in [('Human', 'wrong_human'), ('LLM', 'wrong_llm'), ('Combined', 'wrong_combined')]:
        wrong_base  = np.concatenate(results['base'][test_key])
        wrong_model = np.concatenate(r[test_key])
        stat, p, b01, b10 = mcnemar_test(wrong_base, wrong_model)
        g = cohens_g(b01, b10)
        print(f"  {name:<28} {test_label:<10} {stat:>10.4f} {p:>8.4f} {g:>8.4f}")
    print()

# ---------------------------------------------------------------------------
# Figure: grouped bar chart of ASR by model and paraphrase type
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

models_order = ['Base model', 'Human fine-tuned', 'LLM fine-tuned', 'Combined']
keys_order   = ['base', 'human', 'llm', 'combined']

asr_human_means    = [np.mean(results[k]['asr_human'])    for k in keys_order]
asr_llm_means      = [np.mean(results[k]['asr_llm'])      for k in keys_order]
asr_combined_means = [np.mean(results[k]['asr_combined']) for k in keys_order]

COLOR_HUMAN    = '#4878CF'
COLOR_LLM      = '#E87820'
COLOR_COMBINED = '#6FAE5C'

x = np.arange(len(models_order))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width, asr_human_means,    width, label='Human paraphrases',    color=COLOR_HUMAN,    edgecolor='white')
bars2 = ax.bar(x,         asr_llm_means,      width, label='LLM paraphrases',      color=COLOR_LLM,      edgecolor='white')
bars3 = ax.bar(x + width, asr_combined_means, width, label='Combined paraphrases', color=COLOR_COMBINED, edgecolor='white')

ax.set_ylabel('Attack success rate (ASR)', fontsize=10)
ax.set_title('Attack success rate by model and paraphrase type', fontsize=11, pad=10)
ax.set_xticks(x)
ax.set_xticklabels(models_order, fontsize=9)
ax.set_ylim(0, max(asr_human_means + asr_llm_means + asr_combined_means) + 0.12)
ax.legend(frameon=False, fontsize=9, loc='upper right')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.015,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig('figure_asr.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figure_asr.png', bbox_inches='tight', dpi=300)
print("\nFigure saved as figure_asr.pdf and figure_asr.png")