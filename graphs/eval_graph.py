# %%


import matplotlib.pyplot as plt
import numpy as np
import random

# %%

xticks = ["Flint", "Flint+Cache", "Flint+Multinode", "Flint+Cache+Multinode"]
x = [0.3, 1, 1.7, 2.4]

y1 = [30.766, 69.4241]
y2 = [31.7937, 1.3316]
y3 = [5.3827, 65.4241]
y4 = [5.3827, 0.4048]

bar_width = 0.25

plt.bar(x[0], y1[0], bar_width, color="C0")
plt.bar(x[0], y1[1], bar_width, bottom=y1[0], color="C1")

plt.bar(x[1], y2[0], bar_width, color="C0")
plt.bar(x[1], y2[1], bar_width, bottom=y2[0], color="C1")

plt.bar(x[2], y3[0], bar_width, color="C0")
plt.bar(x[2], y3[1], bar_width, bottom=y3[0], color="C1")

plt.bar(x[3], y4[0], bar_width, color="C0")
plt.bar(x[3], y4[1], bar_width, bottom=y4[0], color="C1")

plt.bar(0, 0, bar_width, color="C0", label="FX graph capture")
plt.bar(0, 0, bar_width, color="C1", label="Conversion to Chakra graph")

plt.xticks(x, xticks, fontsize=12, rotation=20, ha="right")
plt.legend(fontsize=12)
plt.xlim(0, 2.7)
plt.xlabel("Runtime Configurations", fontsize=14)
plt.ylabel("Average Duration (s)", fontsize=14)
plt.title("[Cartoon Graph] Scalability", fontsize=16)
plt.tight_layout()
plt.savefig("scalability.pdf")
# %%
plt.figure()


xticks = [32, 64, 128, 256]
x = [0.3, 1, 1.7, 2.4]
pllz = ["TP=8, FSDP=4", "TP=8, FSDP=2, PP=4", "TP=8, FSDP=8, PP=2", "TP=8, FSDP=8, PP=2, CP=2"]

y1 = [30.766]
y2 = [16.7937]
y3 = [12.3827]
y4 = [11.3827]

bar_width = 0.25

plt.bar(x[0], y1[0], bar_width, color="C0")
plt.text(x[0], y1[0], f"{pllz[0]}", ha="center", va="bottom", fontsize=10)

plt.bar(x[1], y2[0], bar_width, color="C0")
plt.text(x[1], y2[0], f"{pllz[1]}", ha="center", va="bottom", fontsize=10)

plt.bar(x[2], y3[0], bar_width, color="C0")
plt.text(x[2], y3[0], f"{pllz[2]}", ha="center", va="bottom", fontsize=10)

plt.bar(x[3], y4[0], bar_width, color="C0")
plt.text(x[3], y4[0], f"{pllz[3]}", ha="center", va="bottom", fontsize=10)

plt.xticks(x, xticks, fontsize=12, ha="right")
plt.legend(fontsize=12)
plt.xlim(0, 2.7)
plt.xlabel("Number of GPUs", fontsize=14)
plt.ylabel("Average Duration (s)", fontsize=14)
plt.title("[Cartoon Graph] Different # GPUs", fontsize=16)
plt.tight_layout()
plt.savefig("physical.pdf")


# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# First subplot: Scalability
xticks1 = ["1F1B", "Interleaved1F1B", "GPipe", "ZBVZeroBubble", "InterleavedZeroBubble"]
x1 = [0.3, 1, 1.7, 2.4, 3.0]
y1 = [30.766, 69.4241, 31.7937, 40.0, 42.2]
bar_width = 0.25

for idx, bar in enumerate(xticks1):
    axs[0].bar(x1[idx], y1[idx], bar_width, color="C0")
axs[0].set_xticks(x1)
axs[0].set_xticklabels(xticks1, fontsize=12, rotation=20, ha="right")
axs[0].set_xlim(0, 3.4)
axs[0].set_xlabel("PP Schedules", fontsize=14)
axs[0].set_ylabel("Average Duration (s)", fontsize=14)
axs[0].set_title("[Cartoon Graph] Llama3-8B", fontsize=16)

# Second subplot: Different # GPUs
xticks2 = ["1F1B", "Interleaved1F1B", "GPipe", "ZBVZeroBubble", "InterleavedZeroBubble"]
x2 = [0.3, 1, 1.7, 2.4, 3.0]
y2 = [44.8, 31.3, 48.8, 47.1, 55.3]

for idx, bar in enumerate(xticks2):
    axs[1].bar(x2[idx], y2[idx], bar_width, color="C0")

axs[1].set_xticks(x2)
axs[1].set_xticklabels(xticks2, fontsize=12, rotation=20, ha="right")
axs[1].set_xlim(0, 3.4)
axs[1].set_xlabel("PP Schedules", fontsize=14)
axs[1].set_ylabel("Average Duration (s)", fontsize=14)
axs[1].set_title("[Cartoon Graph] Llama3-70B", fontsize=16)

plt.tight_layout()
plt.savefig("pp.pdf")
plt.show()
# %%
