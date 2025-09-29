# %%
# =============================
# 1. Imports
# =============================
import pandas as pd
import numpy as np
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.cluster import AgglomerativeClustering
from pathlib import Path

# NEW: fast Levenshtein
from rapidfuzz.distance import Levenshtein as rf_lev

# -----------------------------
# Config
# -----------------------------
# Cap pairwise distance computation for large logs (O(N^2))
MAX_TRACES = 600          # reduce if still slow (e.g., 300-400)
RNG_SEED = 42             # reproducible sampling
TOPK_ACTS_HEATMAP = 30    # transition heatmap limited to top activities

BASE_DIR = Path(__file__).parent
xes_path = BASE_DIR / "dataset" / "Untitled.xes"
assert xes_path.exists(), f"File not found: {xes_path}"

# =============================
# 2. Load Event Log
# =============================
# Use pathlib path (Windows-safe)
log = xes_importer.apply(str(xes_path))
df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

# Keep only required columns
df = df[["case:concept:name", "concept:name", "time:timestamp"]].copy()

# Ensure proper timestamp, then sort
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")
df = df.dropna(subset=["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)

# =============================
# 3. Create Sequences per Case
# =============================
case_sequences = df.groupby("case:concept:name")["concept:name"].apply(list).to_dict()
sequences = list(case_sequences.values())
case_ids = list(case_sequences.keys())

print(f"Total cases: {len(sequences)} | Mean length: {np.mean([len(s) for s in sequences]):.1f}")

# =============================
# 4. Transition Matrix
# =============================
transitions = Counter()
for seq in sequences:
    for i in range(len(seq) - 1):
        transitions[(seq[i], seq[i + 1])] += 1

transition_df = pd.DataFrame(
    [(a, b, c) for (a, b), c in transitions.items()],
    columns=["From", "To", "Count"]
).sort_values("Count", ascending=False)

if not transition_df.empty:
    total_from = transition_df.groupby("From")["Count"].transform("sum")
    transition_df["Probability"] = transition_df["Count"] / total_from
else:
    transition_df["Probability"] = []

print("\n--- Transition Matrix (head) ---")
print(transition_df.head(15))

# =============================
# 5. Distance Matrix (Fast Edit Distance + Sampling)
# =============================

# Use RapidFuzz (C-accelerated)
def levenshtein(seq1, seq2):
    # rapidfuzz is faster on tuples
    return rf_lev.distance(tuple(seq1), tuple(seq2))

# Sample if too many cases (O(N^2) behavior)
N = len(sequences)
if N > MAX_TRACES:
    rng = np.random.default_rng(RNG_SEED)
    keep = rng.choice(N, size=MAX_TRACES, replace=False)
    sequences_s = [sequences[i] for i in keep]
    case_ids_s = [case_ids[i] for i in keep]
    print(f"\nSampling {len(sequences_s)} / {N} cases for distance-based clustering.")
else:
    sequences_s = sequences
    case_ids_s = case_ids

Ns = len(sequences_s)
dist_matrix = np.zeros((Ns, Ns), dtype=np.float32)

print("\nComputing pairwise Levenshtein distances...")
for i in range(Ns):
    if i % 50 == 0 or i == Ns - 1:
        print(f"  row {i+1}/{Ns}")
    si = sequences_s[i]
    for j in range(i + 1, Ns):
        d = levenshtein(si, sequences_s[j])
        dist_matrix[i, j] = dist_matrix[j, i] = d

# =============================
# 6. Sequence Clustering
# =============================
clustering = AgglomerativeClustering(n_clusters=3, metric="precomputed", linkage="average")
labels = clustering.fit_predict(dist_matrix)

case_cluster = pd.DataFrame({
    "Case": case_ids_s,
    "Cluster": labels
})
print("\n--- Cluster Assignment (head) ---")
print(case_cluster.head())

# =============================
# 7. Visualizations
# =============================

# (a) Sequence length distribution (all cases)
seq_lengths = [len(s) for s in sequences]
plt.figure()
plt.hist(seq_lengths, bins=30, color="skyblue")
plt.title("Sequence Length Distribution")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# (b) Heatmap of Transition Probabilities
# Limit to top-k activities (keeps matrix readable & faster)
if not transition_df.empty:
    act_freq = df["concept:name"].value_counts()
    top_acts = set(act_freq.head(TOPK_ACTS_HEATMAP).index)
    trans_small = transition_df[transition_df["From"].isin(top_acts) & transition_df["To"].isin(top_acts)]
    if not trans_small.empty:
        pivot = trans_small.pivot(index="From", columns="To", values="Probability").fillna(0)
        plt.figure(figsize=(min(12, 0.5 + 0.35 * len(pivot.columns)), min(8, 0.5 + 0.35 * len(pivot.index))))
        sns.heatmap(pivot, cmap="Blues", annot=False)
        plt.title("Transition Probability Matrix (Top Activities)")
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping heatmap: no transitions among top activities.")
else:
    print("Skipping heatmap: no transitions computed.")

# (c) Cluster size distribution (on sampled set)
plt.figure()
uniq, counts = np.unique(labels, return_counts=True)
plt.bar(uniq, counts)
plt.title("Cluster Sizes (on sampled set)" if N > MAX_TRACES else "Cluster Sizes")
plt.xlabel("Cluster")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.show()

# =============================
# 8. SSA-style Plots
# =============================

# Pad sequences so they all have equal length (use all cases for these)
max_len = max(len(s) for s in sequences)
padded_seqs = [s + ["-"] * (max_len - len(s)) for s in sequences]  # "-" = missing
padded_df = pd.DataFrame(padded_seqs, index=case_ids)

# --- (i) Sequence Index Plot ---
plt.figure(figsize=(14, 6))
for i, seq in enumerate(padded_seqs[:100]):  # first 100 cases for clarity
    plt.plot(range(max_len), [hash(a) % 20 for a in seq], lw=2)
plt.title("Sequence Index Plot (first 100 cases)")
plt.xlabel("Time position")
plt.ylabel("Activity (hashed)")
plt.tight_layout()
plt.show()

# --- (ii) State Distribution Plot ---
state_dist = []
for pos in range(max_len):
    col = padded_df.iloc[:, pos]
    freq = col.value_counts(normalize=True)
    state_dist.append(freq)

state_dist_df = pd.DataFrame(state_dist).fillna(0)

plt.figure(figsize=(14, 6))
state_dist_df.plot.area(colormap="tab20", alpha=0.8, linewidth=0)
plt.title("State Distribution Plot")
plt.xlabel("Time position")
plt.ylabel("Proportion of cases")
plt.tight_layout()
plt.show()

# --- (iii) Entropy Plot ---
entropy = -np.nansum(state_dist_df.values * np.log2(state_dist_df.values + 1e-9), axis=1)

plt.figure(figsize=(10, 5))
plt.plot(range(max_len), entropy, marker="o", color="red")
plt.title("Entropy Plot (uncertainty across positions)")
plt.xlabel("Time position")
plt.ylabel("Entropy")
plt.tight_layout()
plt.show()

# =============================
# END
# =============================

# %%
