# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
config_names = [
    "Ring+Ring",
    "DBT+DBT",
    "TACOS+Ring",
    "Ring+TACOS",  # Comment to keep format
    # "Llama 405B\n(FSDP 64)",
    # "Llama 405B\n(TP 8 + FSDP 8 +\nDP 8)",
]


figure_value_latency = [28559054200, 104300344603, 0, 0]  # , 395219461, 0]
figure_value_latency = np.array(figure_value_latency) / 1_000_000_000  # Convert ns to second

# figure_value_allfront_latency = [17923331972, 8971908428, 4496198896, 458052424]  # , 0, 0]

# figure_value_allfront_latency = [af / o for af, o in zip(figure_value_allfront_latency, figure_value_ordered_latency, strict=False)]
# figure_value_ordered_latency = [1 for _ in figure_value_ordered_latency]
# %%
# assume config_names, figure_value_ordered_memory, figure_value_allfront_memory,
# figure_value_ordered_latency, figure_value_allfront_latency are defined above

x = np.arange(len(config_names))
width = 0.35

fig, ax_lat = plt.subplots(figsize=(10, 5))


# Create a second y-axis for latency
# Lines: latency on right y-axis
# bars1 = ax_lat.bar(x - width / 2, figure_value_ordered_latency, width, label="Default FSDP (latency)", color="#4C72B0")
# bars2 = ax_lat.bar(x + width / 2, figure_value_allfront_latency, width, label="Reordered AG (latency)", color="#55A868")
bars1 = ax_lat.bar(x, figure_value_latency, width, color="#4C72B0")
ax_lat.set_ylabel("Duration (s)", fontsize=18)
ax_lat.set_xlabel("Collective Algorithm (TP + FSDP)", fontsize=18)
ax_lat.tick_params(axis="y", labelsize=16)
# ax_lat.legend(loc="upper left", fontsize=14)
# ax_lat.set_ylim(0, 1.2)
ax_lat.set_xticks(x)
ax_lat.set_xticklabels(config_names, fontsize=16)

# Set final figure size and resolution
fig.set_size_inches(10, 5)
# fig.savefig("fig2_fsdp.png", dpi=300, bbox_inches="tight")
fig.set_dpi(600)
plt.tight_layout()
plt.savefig("tmp.pdf", bbox_inches="tight", dpi=600, pad_inches=0.1)
plt.show()

# %%
