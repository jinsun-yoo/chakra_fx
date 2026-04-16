# %%
import matplotlib.pyplot as plt
import numpy as np

# %%

# %%
figure_name = [
    "Llama 8B (TP 8)",
    "Llama 8B (TP 8 + FSDP 2)",
    "Llama 8B (TP 2 + FSDP 2 + DP 2)",
    "Llama 8B (TP 4 + FSDP 4)",
]

figure_value_ground = [
    # Llama 8B (TP 8)
    {"GeMM": 675, "Attn": 64, "ElemWise": 1191, "Others": 260, "AllReduce": 224, "AllGather": 1, "ReduceScatter": 1},
    # Llama 8B (TP 8 + FSDP 2)
    {"GeMM": 675, "Attn": 64, "ElemWise": 1613, "Others": 228, "AllReduce": 225, "AllGather": 292, "ReduceScatter": 292},
    # Llama 8B (TP 2 + FSDP 2 + DP 2)
    {"GeMM": 675, "Attn": 64, "ElemWise": 1677, "Others": 164, "AllReduce": 516, "AllGather": 292, "ReduceScatter": 292},
    # Llama 8B (TP 2 + FSDP 2 + DP 2)
    {"GeMM": 675, "Attn": 64, "ElemWise": 1677, "Others": 228, "AllReduce": 225, "AllGather": 292, "ReduceScatter": 292},
]

# Desired value with the 'optimization'
figure_value_flint = [
    # Llama 8B (TP 8)
    {"GeMM": 675, "Attn": 64, "ElemWise": 1907, "Others": 197, "AllReduce": 224, "AllGather": 1, "ReduceScatter": 1},
    # Llama 8B (TP 8 + FSDP 2)
    {"GeMM": 675, "Attn": 64, "ElemWise": 1973, "Others": 199, "AllReduce": 225, "AllGather": 292, "ReduceScatter": 292},
    # Llama 8B (TP 2 + FSDP 2 + DP 2)
    {"GeMM": 675, "Attn": 64, "ElemWise": 1973, "Others": 199, "AllReduce": 516, "AllGather": 292, "ReduceScatter": 292},
    # Llama 8B (TP 2 + FSDP 2 + DP 2)
    {"GeMM": 675, "Attn": 64, "ElemWise": 1973, "Others": 199, "AllReduce": 225, "AllGather": 292, "ReduceScatter": 292},
]
# True figure value
# figure_value_flint = [
#     # Llama 8B (TP 8)
#     {"GeMM": 675, "Attn": 64, "ElemWise": 1907, "Others": 197, "AllReduce": 224, "AllGather": 1, "ReduceScatter": 1},
#     # Llama 8B (TP 8 + FSDP 2)
#     {"GeMM": 675, "Attn": 64, "ElemWise": 1973, "Others": 199, "AllReduce": 225, "AllGather": 582, "ReduceScatter": 292},
#     # Llama 8B (TP 2 + FSDP 2 + DP 2)
#     {"GeMM": 675, "Attn": 64, "ElemWise": 1973, "Others": 199, "AllReduce": 516, "AllGather": 582, "ReduceScatter": 292},
#     # Llama 8B (TP 2 + FSDP 2 + DP 2)
#     {"GeMM": 675, "Attn": 64, "ElemWise": 1973, "Others": 199, "AllReduce": 225, "AllGather": 582, "ReduceScatter": 292},
# ]

# %%

keys = ["GeMM", "Attn", "ElemWise", "Others", "AllReduce", "AllGather", "ReduceScatter"]
keys_shortened = ["MM", "Attn", "Elem", "Other", "AR", "AG", "RS"]
n = len(keys)

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ground = figure_value_ground[i]
    flint = figure_value_flint[i]

    g_vals = np.array([ground[k] for k in keys], dtype=float)
    f_vals = np.array([flint[k] for k in keys], dtype=float)

    # Normalize by the sum to get proportions
    g_norm = [1 for _ in g_vals]
    f_norm = f_vals / g_vals

    ind = np.arange(n)
    width = 0.35

    ax.bar(ind - width / 2, g_norm, width, label="Post-exec", color="#564BDE", alpha=0.9)
    ax.bar(ind + width / 2, f_norm, width, label="Flint", facecolor="none", edgecolor="#E7124D", hatch="//")

    ax.set_xticks(ind)
    # ax.set_xticklabels(keys, rotation=60, ha="right")
    ax.set_xticklabels(keys_shortened)
    ax.set_title(figure_name[i])
    ax.set_ylabel("Normalized")
    ax.set_ylim(0, 2.1)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Set final figure size and resolution
fig.legend(labels=["Post-exec", "Flint"], loc="upper right", ncol=1, frameon=True, bbox_to_anchor=(1, 0.90))
fig.set_size_inches(8.5, 2.8)
fig.set_dpi(600)
plt.tight_layout(rect=[0, 0, 0.87, 1])
plt.savefig("tmp.pdf", bbox_inches="tight", dpi=600, pad_inches=0.1)

# %%
