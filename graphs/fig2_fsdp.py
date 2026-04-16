# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
config_names = [
    "Llama 8B\n(FSDP 8)",  # Comment to keep format
    "Llama 8B\n(FSDP 64)",
    "Llama 70B\n(FSDP 8)",
    "Llama 70B\n(FSDP 64)",
    # "Llama 405B\n(FSDP 64)",
    # "Llama 405B\n(TP 8 + FSDP 8 +\nDP 8)",
]

figure_value_ordered_memory = [6.09, 3.28, 37.39, 9.71]  # , 35.01, 0]
figure_value_allfront_memory = [6.30, 3.5, 38.26, 10.58]  # , 0, 0]


figure_value_ordered_latency = [63692734, 28532739, 493433816, 95173393]  # , 395219461, 0]
figure_value_allfront_latency = [54420524, 14007333, 458052424, 76454148]  # , 0, 0]

figure_value_ordered_latency = np.array(figure_value_ordered_latency) / 1000_000  # convert to milliseconds
figure_value_allfront_latency = np.array(figure_value_allfront_latency) / 1000_000  # convert to milliseconds

# %%
# assume config_names, figure_value_ordered_memory, figure_value_allfront_memory,
# figure_value_ordered_latency, figure_value_allfront_latency are defined above

x = np.arange(len(config_names))
width = 0.35

fig, ax_lat = plt.subplots(figsize=(10, 5))

# Bars: memory on left y-axis
bars1 = ax_lat.bar(x - width / 2, figure_value_ordered_latency, width, label="Default FSDP (latency)", color="#4C72B0")
bars2 = ax_lat.bar(x + width / 2, figure_value_allfront_latency, width, label="Reordered AG (latency)", color="#55A868")
ax_lat.set_ylabel("Duration (milliseconds)", fontsize=18)
ax_lat.set_xticks(x)
ax_lat.set_xticklabels(config_names, fontsize=16)
ax_lat.set_ylim(0, 600)
ax_lat.legend(loc="upper left", fontsize=14)
ax_lat.tick_params(axis="y", labelsize=16)


# Create a second y-axis for latency
ax_mem = ax_lat.twinx()
# Lines: latency on right y-axis
(line1,) = ax_mem.plot(x - width / 2, figure_value_ordered_memory, label="Default FSDP (memory)", color="#C44E52", marker="o")
(line2,) = ax_mem.plot(x + width / 2, figure_value_allfront_memory, label="Reordered AG (memory)", color="#8172B3", marker="o")
ax_mem.tick_params(axis="y", labelsize=16)
ax_mem.legend(loc="upper right", fontsize=14)
ax_mem.set_ylabel("Peak Memory (GB)", fontsize=18)
ax_mem.set_ylim(0, 50)

# Set final figure size and resolution
fig.set_size_inches(10, 5)
# fig.savefig("fig2_fsdp.png", dpi=300, bbox_inches="tight")
fig.set_dpi(600)
plt.tight_layout()
plt.savefig("tmp.pdf", bbox_inches="tight", dpi=600, pad_inches=0.1)
plt.show()

# %%
config_names = [
    "12.5GB/s",
    "25GB/s",
    "50GB/s",
    "500GB/s",  # Comment to keep format
    # "Llama 405B\n(FSDP 64)",
    # "Llama 405B\n(TP 8 + FSDP 8 +\nDP 8)",
]


figure_value_ordered_latency = [17946545445, 8995158978, 4519641120, 493433816]  # , 395219461, 0]
figure_value_allfront_latency = [17923331972, 8971908428, 4496198896, 458052424]  # , 0, 0]

figure_value_allfront_latency = [af / o for af, o in zip(figure_value_allfront_latency, figure_value_ordered_latency, strict=False)]
figure_value_ordered_latency = [1 for _ in figure_value_ordered_latency]
# %%
# %%
# assume config_names, figure_value_ordered_memory, figure_value_allfront_memory,
# figure_value_ordered_latency, figure_value_allfront_latency are defined above

x = np.arange(len(config_names))
width = 0.35

fig, ax_lat = plt.subplots(figsize=(10, 5))


# Create a second y-axis for latency
# Lines: latency on right y-axis
bars1 = ax_lat.bar(x - width / 2, figure_value_ordered_latency, width, label="Default FSDP (latency)", color="#4C72B0")
bars2 = ax_lat.bar(x + width / 2, figure_value_allfront_latency, width, label="Reordered AG (latency)", color="#55A868")
ax_lat.set_ylabel("Normalized Duration", fontsize=18)
ax_lat.set_xlabel("Physical Bandwidth (GB/s)", fontsize=18)
ax_lat.tick_params(axis="y", labelsize=16)
ax_lat.legend(loc="upper left", fontsize=14)
ax_lat.set_ylim(0, 1.2)
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
