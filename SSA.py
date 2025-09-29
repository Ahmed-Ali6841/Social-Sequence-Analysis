# %%
!pip install pm4py


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

# =============================
# 2. Load Event Log
# =============================

# Option A: Load from XES
log = xes_importer.apply("/content/your_file_fixed.xes")
df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)


# Keep only required columns
df = df[["case:concept:name", "concept:name", "time:timestamp"]]
df = df.sort_values(by=["case:concept:name", "time:timestamp"])

# =============================
# 3. Create Sequences per Case
# =============================
case_sequences = df.groupby("case:concept:name")["concept:name"].apply(list).to_dict()
sequences = list(case_sequences.values())
case_ids = list(case_sequences.keys())

# =============================
# 4. Transition Matrix
# =============================
transitions = Counter()
for seq in sequences:
    for i in range(len(seq)-1):
        transitions[(seq[i], seq[i+1])] += 1

transition_df = pd.DataFrame(
    [(a, b, c) for (a, b), c in transitions.items()],
    columns=["From", "To", "Count"]
)
total_from = transition_df.groupby("From")["Count"].transform("sum")
transition_df["Probability"] = transition_df["Count"] / total_from

print("\n--- Transition Matrix ---")
print(transition_df.head(15))

# =============================
# 5. Distance Matrix (Edit Distance)
# =============================
def levenshtein(seq1, seq2):
    """Compute Levenshtein edit distance between two sequences"""
    n, m = len(seq1), len(seq2)
    if n > m:
        seq1, seq2 = seq2, seq1
        n, m = m, n

    current = range(n+1)
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if seq1[j-1] != seq2[i-1]:
                change += 1
            current[j] = min(add, delete, change)
    return current[n]

N = len(sequences)
dist_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(i+1, N):
        d = levenshtein(sequences[i], sequences[j])
        dist_matrix[i, j] = dist_matrix[j, i] = d

# =============================
# 6. Sequence Clustering
# =============================
clustering = AgglomerativeClustering(n_clusters=3, metric="precomputed", linkage="average")
labels = clustering.fit_predict(dist_matrix)

case_cluster = pd.DataFrame({
    "Case": case_ids,
    "Cluster": labels
})
print("\n--- Cluster Assignment ---")
print(case_cluster.head())

# =============================
# 7. Visualizations
# =============================

# (a) Sequence length distribution
seq_lengths = [len(s) for s in sequences]
plt.hist(seq_lengths, bins=30, color="skyblue")
plt.title("Sequence Length Distribution")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()

# (b) Heatmap of Transition Probabilities
pivot = transition_df.pivot(index="From", columns="To", values="Probability").fillna(0)

plt.figure(figsize=(12,8))
sns.heatmap(pivot, cmap="Blues", annot=False)
plt.title("Transition Probability Matrix")
plt.show()

# (c) Cluster size distribution
plt.bar(*np.unique(labels, return_counts=True))
plt.title("Cluster Sizes")
plt.xlabel("Cluster")
plt.ylabel("Number of Cases")
plt.show()

# =============================
# 8. SSA-style Plots
# =============================

# Pad sequences so they all have equal length
max_len = max(len(s) for s in sequences)
padded_seqs = [s + ["-"]*(max_len-len(s)) for s in sequences]  # "-" = missing

padded_df = pd.DataFrame(padded_seqs, index=case_ids)

# --- (i) Sequence Index Plot ---
plt.figure(figsize=(14,6))
for i, seq in enumerate(padded_seqs[:100]):  # first 100 cases for clarity
    plt.plot(range(max_len), [hash(a)%20 for a in seq], lw=2)
plt.title("Sequence Index Plot (first 100 cases)")
plt.xlabel("Time position")
plt.ylabel("Activity (hashed)")
plt.show()

# --- (ii) State Distribution Plot ---
state_dist = []
for pos in range(max_len):
    col = padded_df.iloc[:, pos]
    freq = col.value_counts(normalize=True)
    state_dist.append(freq)

state_dist_df = pd.DataFrame(state_dist).fillna(0)

plt.figure(figsize=(14,6))
state_dist_df.plot.area(colormap="tab20", alpha=0.8, linewidth=0)
plt.title("State Distribution Plot")
plt.xlabel("Time position")
plt.ylabel("Proportion of cases")
plt.show()

# --- (iii) Entropy Plot ---
entropy = -np.nansum(state_dist_df.values * np.log2(state_dist_df.values+1e-9), axis=1)

plt.figure(figsize=(10,5))
plt.plot(range(max_len), entropy, marker="o", color="red")
plt.title("Entropy Plot (uncertainty across positions)")
plt.xlabel("Time position")
plt.ylabel("Entropy")
plt.show()

# =============================
# END
# =============================




# %%
